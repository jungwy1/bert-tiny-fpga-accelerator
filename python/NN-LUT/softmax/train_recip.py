"""
LUT for reciprocal (deferred-softmax normalization).  16-segment PWL, ANALYTIC fit.

Deferred softmax divides by rowsum AFTER P'@V:
    ctx = ctx_unnorm_int * S_v / (rowsum * S_ctx)  quantized to int8 (scale S_ctx)
        = ctx_unnorm_int * M(rowsum),   M(rowsum) = C / rowsum,   C = S_v / S_ctx
This LUT gives the per-row multiplier M(rowsum) as a FRAC-bit fixed-point number:
    M_fixed = (s[i]*rowsum + t[i]) >>> sh[i]  ~= round(C * 2^FRAC / rowsum)
and the HW then applies:  ctx_int8 = clamp( (ctx_unnorm_int * M_fixed) >>> FRAC, +-127 ).

1/rowsum is a known convex function over a 64x range, so a learned NN-LUT drifts and
wastes segments; instead we place 16 FIXED log-spaced segments over [ROWSUM_LO,ROWSUM_HI]
(constant relative error for 1/x) and fit each segment's line by RELATIVE least-squares.
Per layer only the constant C = S_v/S_ctx scales s,t (breakpoints d are shared).
rowsum in [127, 127*S]: max P' element is always exp(0)->127, S = seq len (<=64).

Run (from python/NN-LUT/softmax/):  python train_recip.py
Output: recip_fit.png, lut_recip.txt, lut_recip.json.
"""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

QP_PATH   = "../../quant_params.pt"
N_SEG     = 16               # 16 segments -> 15 breakpoints
W_S       = 18               # slope width  (signed)
W_T       = 48               # offset width (signed)
W_SH      = 6                # shamt width  -> sh in 0..63
FRAC      = 23               # output fractional bits: M ~= M_fixed / 2^FRAC (R:S18, matches VFU 2^23/L)
S_MAX     = (1 << (W_S - 1)) - 1
T_MAX     = (1 << (W_T - 1)) - 1
SH_MAX    = (1 << W_SH) - 1
ROWSUM_LO = 127              # min rowsum (fully peaked row: one key at 127, rest 0)
ROWSUM_HI = 127 * 64         # max rowsum (uniform row, S = MAX_LEN = 64)


def pick_shamt(s, t):
    """Largest sh in [0,SH_MAX] with round(s*2^sh) in 18b and round(t*2^sh) in 48b."""
    for sh in range(SH_MAX, -1, -1):
        sf, tf = round(s * (1 << sh)), round(t * (1 << sh))
        if abs(sf) <= S_MAX and abs(tf) <= T_MAX:
            return 0 if (sf == 0 and tf == 0) else sh
    return 0


def fit_segment(lo, hi, scale):
    """Relative least-squares line s*x + t ~= scale*2^FRAC / x over [lo,hi] (minimizes rel error)."""
    x = np.geomspace(lo, hi, 128)
    objective = scale * (1 << FRAC) / x
    X = np.stack([x / objective, 1.0 / objective], axis=1)     # rows scaled by 1/objective -> relative metric
    s_int, t_int = np.linalg.lstsq(X, np.ones_like(x), rcond=None)[0]
    return s_int, t_int


def convert_layer(L, qp, edges):
    """Fixed log-spaced segments + per-segment relative fit -> integer LUT + eval."""
    C = qp["act"][f"L{L}.v"]["scale"] / qp["act"][f"L{L}.ctx"]["scale"]   # M = C/rowsum

    d_fix = np.round(edges[1:-1]).astype(np.int64)       # 15 interior breakpoints (int32)
    s_int = np.empty(N_SEG); t_int = np.empty(N_SEG)
    for i in range(N_SEG):
        s_int[i], t_int[i] = fit_segment(edges[i], edges[i + 1], C)

    # single per-page shamt (6b): largest shift keeping every ACTIVE segment's M in 18b.
    per_seg = [pick_shamt(si, ti) for si, ti in zip(s_int, t_int)]
    sh = min(p for p, si, ti in zip(per_seg, s_int, t_int) if si != 0 or ti != 0)  # single int
    s_fix = np.round(s_int * (1 << sh)).astype(np.int64)
    t_fix = np.round(t_int * (1 << sh)).astype(np.int64)

    # evaluate the TRUE fixed-point LUT over the full rowsum range; error is RELATIVE
    # (M multiplies ctx_unnorm -> rel error in M = rel error in ctx; worst ctx LSB = 127*rel)
    rs   = np.arange(ROWSUM_LO, ROWSUM_HI + 1, dtype=np.int64)
    idx  = np.searchsorted(d_fix, rs, side="right")      # 0..15
    Mfix = (s_fix[idx] * rs + t_fix[idx]) >> sh           # int64 fixed-point M (single-page shift)
    M_hat  = Mfix.astype(np.float64) / (1 << FRAC)
    M_true = C / rs.astype(np.float64)
    rel_err = np.abs(M_hat - M_true) / M_true
    lsb_err = rel_err * 127.0                             # worst-case ctx LSB (full-scale ctx=127)

    return dict(L=L, C=C, x=rs.astype(np.float64), yhat=M_hat, ytrue=M_true,
                rel_err=rel_err, lsb_err=lsb_err, d_fix=d_fix, s_fix=s_fix, t_fix=t_fix, sh=sh)


def main():
    qp = torch.load(QP_PATH)
    edges = np.geomspace(ROWSUM_LO, ROWSUM_HI, N_SEG + 1)  # 17 edges -> 16 log-spaced segments
    res = [convert_layer(L, qp, edges) for L in (0, 1)]

    # ---- report + Verilog dump + JSON ------------------------------------
    lines = []
    report = {"function": "recip", "frac": FRAC,
              "widths": {"d": 32, "s": W_S, "t": W_T, "sh": W_SH},
              "n_entries": N_SEG, "rowsum_range": [ROWSUM_LO, ROWSUM_HI], "layers": {}}
    for r in res:
        me, mn = float(r["lsb_err"].max()), float(r["lsb_err"].mean())
        rel_mx = float(r["rel_err"].max())
        print(f"L{r['L']}: C={r['C']:.4f}  max_rel={rel_mx*100:.3f}%  "
              f"max_ctx_err={me:.3f} LSB  mean={mn:.3f}  shift={r['sh']}")
        assert np.all(np.abs(r["s_fix"]) <= S_MAX), "s_fix overflows 18-bit"
        assert np.all(np.abs(r["t_fix"]) <= T_MAX), "t_fix overflows 48-bit"
        assert 0 <= r["sh"] <= SH_MAX, "shift overflows 6-bit"
        lines.append(f"// ---- L{r['L']} recip LUT  (C={r['C']:.6f} FRAC={FRAC} shift={r['sh']}, log-spaced)  "
                     f"max_rel={rel_mx*100:.3f}%  max_ctx_err={me:.3f} LSB")
        lines += [f"d[{i}] = 32'sd{int(v)};" for i, v in enumerate(r["d_fix"])]
        lines += [f"s[{i}] = 18'sd{int(v)};" for i, v in enumerate(r["s_fix"])]
        lines += [f"t[{i}] = 48'sd{int(v)};" for i, v in enumerate(r["t_fix"])]
        lines.append(f"shift = 6'd{int(r['sh'])};")
        lines.append("")
        report["layers"][f"L{r['L']}"] = {
            "C": float(r["C"]), "max_rel_err": rel_mx, "max_ctx_err_lsb": me, "mean_ctx_err_lsb": mn,
            "shift": int(r["sh"]),
            "d": [int(v) for v in r["d_fix"]], "s": [int(v) for v in r["s_fix"]],
            "t": [int(v) for v in r["t_fix"]]}
    with open("lut_recip.txt", "w") as f:
        f.write("\n".join(lines))
    with open("lut_recip.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote lut_recip.txt, lut_recip.json")

    # ---- plot: per-layer M(rowsum) + ctx LSB error -----------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, r in enumerate(res):
        ax[0][k].plot(r["x"], r["ytrue"], "k", lw=1.2, label="C/rowsum")
        ax[0][k].plot(r["x"], r["yhat"], "C1--", lw=1.0, label="16-seg LUT")
        for dv in r["d_fix"]:
            ax[0][k].axvline(dv, color="C0", alpha=0.25, lw=0.6)
        ax[0][k].set_xscale("log")
        ax[0][k].set_title(f"L{r['L']} M(rowsum)  (max_rel={r['rel_err'].max()*100:.2f}%)")
        ax[0][k].set_xlabel("rowsum (int, log)"); ax[0][k].set_ylabel("M = C/rowsum")
        ax[0][k].legend(); ax[0][k].grid(alpha=0.3)
        ax[1][k].plot(r["x"], r["lsb_err"], "C3", lw=0.8)
        ax[1][k].set_xscale("log")
        ax[1][k].set_title(f"L{r['L']} worst-case ctx error (127*rel), LSB")
        ax[1][k].set_xlabel("rowsum (int, log)"); ax[1][k].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("recip_fit.png", dpi=120)
    print("wrote recip_fit.png")


if __name__ == "__main__":
    main()
