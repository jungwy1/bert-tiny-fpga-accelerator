"""
LUT for LayerNorm rsqrt -- GaugePack v1 frozen contract (U8 output, D_PRE_SHIFT=4, single shift).

    D    = 128*sum(z^2) - sum(z)^2                    # nqs2, on the int8 residual (== co-worker D)
    D27  = RNE_even(D / 16)                           # D_PRE_SHIFT=4, positive A27 input ("16x")
    rho  = 1 / sqrt(16*D27) = 1 / sqrt(D)             # real teacher
    R    = clamp( (s*D27 + t) >>> shift , 0, 255 )  ~= round(rho / s_rho)   # U8 [0,255]
    s_rho = max(rho)/255 = 1 / (255*sqrt(D_min))     # chosen so R hits 255 at D_min
    downstream:  T = (128*z - S) * R                 # = x_hat at scale s_rho (no intermediate shift)

16 fixed log-spaced segments over the D27 domain + per-segment relative fit, SINGLE per-page shift,
U8 clamp.  Per-LN s_rho.  Matches the co-worker's frozen arithmetic contract (D/16 in, U8 out); the
fit itself is analytic log-spaced (not their NN spline), so codes are close but not bit-identical.

Run (from python/NN-LUT/layernorm/):  python train_rsqrt.py
Output: rsqrt_fit.png, lut_rsqrt.txt, lut_rsqrt.json.
"""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

QP_PATH   = "../../quant_params.pt"
N_SEG     = 16               # 16 segments -> 15 breakpoints
W_S       = 18               # slope width  (signed)
W_T       = 48               # offset width (signed)
W_SH      = 6                # shamt width  -> shift in 0..63
D_PRE     = 4                # D_PRE_SHIFT: D27 = RNE(D / 16)
QMAX      = 255              # rho output U8 [0,255]
S_MAX     = (1 << (W_S - 1)) - 1
T_MAX     = (1 << (W_T - 1)) - 1
SH_MAX    = (1 << W_SH) - 1
N_FEAT    = 128              # LayerNorm feature dim -> |x_hat| <= sqrt(N-1)
XHAT_MAX  = (N_FEAT - 1) ** 0.5
LNS       = ["L0.ln1", "L0.ln2", "L1.ln1", "L1.ln2"]   # encoder LNs (emb_ln is host-side)


def pick_shamt(s, t):
    """Largest sh in [0,SH_MAX] with round(s*2^sh) in 18b and round(t*2^sh) in 48b."""
    for sh in range(SH_MAX, -1, -1):
        sf, tf = round(s * (1 << sh)), round(t * (1 << sh))
        if abs(sf) <= S_MAX and abs(tf) <= T_MAX:
            return 0 if (sf == 0 and tf == 0) else sh
    return 0


def rho_code(d27, s_rho):
    """Real teacher rho=1/sqrt(16*D27), scaled to U8 code (before rounding)."""
    return (1.0 / np.sqrt(16.0 * d27)) / s_rho


def fit_segment(lo, hi, s_rho):
    """Relative least-squares line s*D27 + t ~= rho_code(D27) over [lo,hi]."""
    x = np.geomspace(lo, hi, 128)
    objective = rho_code(x, s_rho)
    X = np.stack([x / objective, 1.0 / objective], axis=1)   # rows scaled by 1/objective -> relative
    s_int, t_int = np.linalg.lstsq(X, np.ones_like(x), rcond=None)[0]
    return s_int, t_int


def convert_ln(ln, qp):
    """GaugePack rsqrt page: D27 domain, U8 target, log-spaced 16-seg + single shift."""
    e = qp["weight"][ln]
    d_min, d_max = int(e["nqs2_int_min"]), int(e["nqs2_int_max"])
    # D27 = RNE(D/16); domain in D27 units
    lo = (d_min + (1 << (D_PRE - 1))) >> D_PRE
    hi = (d_max + (1 << (D_PRE - 1))) >> D_PRE
    s_rho = float(rho_code(lo, 1.0)) / QMAX          # = max(rho)/255 (rho largest at D27 min)

    edges = np.geomspace(lo, hi, N_SEG + 1)          # 17 edges -> 16 log-spaced segments
    d_fix = np.round(edges[1:-1]).astype(np.int64)   # 15 interior breakpoints (D27, int32)
    s_int = np.empty(N_SEG); t_int = np.empty(N_SEG)
    for i in range(N_SEG):
        s_int[i], t_int[i] = fit_segment(edges[i], edges[i + 1], s_rho)

    # single per-page shift = min active pick_shamt (dead segments excluded)
    per_seg = [pick_shamt(si, ti) for si, ti in zip(s_int, t_int)]
    sh = min(p for p, si, ti in zip(per_seg, s_int, t_int) if si != 0 or ti != 0)
    s_fix = np.round(s_int * (1 << sh)).astype(np.int64)
    t_fix = np.round(t_int * (1 << sh)).astype(np.int64)

    # evaluate the TRUE fixed-point LUT over the D27 domain; error is U8 CODE error
    d27  = np.unique(np.round(np.geomspace(lo, hi, 200_000)).astype(np.int64))
    idx  = np.searchsorted(d_fix, d27, side="right")     # 0..15
    R_hat  = np.clip((s_fix[idx] * d27 + t_fix[idx]) >> sh, 0, QMAX).astype(np.float64)
    R_true = np.clip(np.round(rho_code(d27, s_rho)), 0, QMAX)
    code_err = np.abs(R_hat - R_true)
    rel_err  = code_err / np.maximum(R_true, 1.0)         # rel error on rho (= on x_hat)
    xhat_err = rel_err * XHAT_MAX                          # worst-case abs x_hat error

    return dict(ln=ln, lo=lo, hi=hi, d_min=d_min, d_max=d_max, s_rho=s_rho, shift=sh,
                x=d27.astype(np.float64), yhat=R_hat, ytrue=R_true,
                code_err=code_err, rel_err=rel_err, xhat_err=xhat_err,
                d_fix=d_fix, s_fix=s_fix, t_fix=t_fix)


def main():
    qp = torch.load(QP_PATH)
    res = [convert_ln(ln, qp) for ln in LNS]

    # ---- report + Verilog dump + JSON ------------------------------------
    lines = []
    report = {"function": "rsqrt", "d_pre_shift": D_PRE, "qmax": QMAX,
              "widths": {"d": 32, "s": W_S, "t": W_T, "sh": W_SH},
              "n_entries": N_SEG, "n_feat": N_FEAT, "layers": {}}
    for r in res:
        me = float(r["code_err"].max()); mn = float(r["code_err"].mean())
        rel_mx = float(r["rel_err"].max()); xe = float(r["xhat_err"].max())
        print(f"{r['ln']:8s}: D27=[{r['lo']},{r['hi']}]  s_rho={r['s_rho']:.4e}  shift={r['shift']}  "
              f"max_code_err={me:.2f} mean={mn:.3f}  max_rel={rel_mx*100:.3f}%  xhat_err={xe:.4f}")
        assert np.all(np.abs(r["s_fix"]) <= S_MAX), "s_fix overflows 18-bit"
        assert np.all(np.abs(r["t_fix"]) <= T_MAX), "t_fix overflows 48-bit"
        assert 0 <= r["shift"] <= SH_MAX, "shift overflows 6-bit"
        lines.append(f"// ---- {r['ln']} rsqrt LUT  (D27=[{r['lo']},{r['hi']}] D_PRE=4 U8 shift={r['shift']} "
                     f"s_rho={r['s_rho']:.6e})  max_code_err={me:.2f}")
        lines += [f"d[{i}] = 32'sd{int(v)};" for i, v in enumerate(r["d_fix"])]
        lines += [f"s[{i}] = 18'sd{int(v)};" for i, v in enumerate(r["s_fix"])]
        lines += [f"t[{i}] = 48'sd{int(v)};" for i, v in enumerate(r["t_fix"])]
        lines.append(f"shift = 6'd{int(r['shift'])};")
        lines.append("")
        report["layers"][r["ln"]] = {
            "d_min": r["d_min"], "d_max": r["d_max"], "d27_min": r["lo"], "d27_max": r["hi"],
            "s_rho": r["s_rho"], "shift": int(r["shift"]),
            "max_code_err": me, "max_rel_err": rel_mx, "max_xhat_err": xe,
            "d": [int(v) for v in r["d_fix"]], "s": [int(v) for v in r["s_fix"]],
            "t": [int(v) for v in r["t_fix"]]}
    with open("lut_rsqrt.txt", "w") as f:
        f.write("\n".join(lines))
    with open("lut_rsqrt.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote lut_rsqrt.txt, lut_rsqrt.json")

    # ---- plot: per-LN R(D27) + U8 code error -----------------------------
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for k, r in enumerate(res):
        ax[0][k].plot(r["x"], r["ytrue"], "k", lw=1.2, label="R = rho/s_rho (U8)")
        ax[0][k].plot(r["x"], r["yhat"], "C1--", lw=1.0, label="16-seg LUT")
        for dv in r["d_fix"]:
            ax[0][k].axvline(dv, color="C0", alpha=0.25, lw=0.6)
        ax[0][k].set_xscale("log")
        ax[0][k].set_title(f"{r['ln']}  (max_code_err={r['code_err'].max():.1f})")
        ax[0][k].set_xlabel("D27 = RNE(D/16) (log)"); ax[0][k].set_ylabel("R (U8)")
        ax[0][k].legend(); ax[0][k].grid(alpha=0.3)
        ax[1][k].plot(r["x"], r["code_err"], "C3", lw=0.8)
        ax[1][k].set_xscale("log")
        ax[1][k].set_title(f"{r['ln']} U8 code error")
        ax[1][k].set_xlabel("D27 (log)"); ax[1][k].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("rsqrt_fit.png", dpi=120)
    print("wrote rsqrt_fit.png")


if __name__ == "__main__":
    main()
