"""
NN-LUT for exp (deferred-softmax numerator).  1-hidden ReLU net = 16-entry LUT.

Softmax is computed deferred:  P' = exp(x - rowmax),  divide by rowsum after P'@V.
The exp input is z = scores_int32 - rowmax_int32  (<= 0, accumulator domain), so we
fit exp(u) on real u in [-8,0] (exp(-6)*127 < 0.5 -> 0 below that) and fold per-layer
scales in as constants (u = S_x*z,  P' = exp(u)/S_y):
    d_int = d_u/S_x,   s_int = s_u*S_x/S_y,   t_int = t_u/S_y
    S_x = S_q*S_k/sqrt(d)  (Q@K^T accumulator scale, folds the 1/sqrt(d) attn scaling)
    S_y = probs scale (fixed 1/127)
Far-negative z falls in the flat left segment -> 0 (handled by the ReLUs themselves).

HW eval per segment i:  P' = (s[i]*z + t[i]) >>> sh[i], clamped to [0,127].
Quantization widths: slope s=18b, offset t=48b, shift sh=6b (0..63), per-segment shamt.

Run (from python/NN-LUT/softmax/):  python train_exp.py
Output: exp_u.png, fit.png, lut_exp.txt, lut_exp.json.
"""
import math
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

QP_PATH   = "../../quant_params.pt"
D_HEAD    = 64               # attention head_dim -> 1/sqrt(d) folded into S_x
N_NEURON  = 15               # -> 16 intervals
W_S       = 18               # slope width  (signed)
W_T       = 48               # offset width (signed)
W_SH      = 6                # shamt width  -> sh in 0..63
S_MAX     = (1 << (W_S - 1)) - 1
T_MAX     = (1 << (W_T - 1)) - 1
SH_MAX    = (1 << W_SH) - 1
CLAMP     = 127.0            # int8 output clamp (P' in [0,127])
U_LO, U_HI = -8.0, 0.0       # exp training domain (z <= 0; exp negligible below ~-6)
N_SAMPLES = 100_000
EPOCHS    = 8000
LR        = 5e-3
torch.manual_seed(42)

class LutNet(nn.Module):
    """1-hidden ReLU net (paper Eq. 5):  NN(u) = sum_i m_i * relu(n_i*u + b_i).
    No output bias -- exp's left tail -> 0 is realized by the ReLUs themselves."""
    def __init__(self, bp0):
        super().__init__()
        k = len(bp0)
        self.n = nn.Parameter(torch.ones(k) + torch.randn(k) * 0.05)        # n_i > 0 (exp increasing)
        self.b = nn.Parameter(-torch.as_tensor(bp0, dtype=torch.float32))   # bp = -b_i/n_i ~ bp0
        self.m = nn.Parameter(torch.randn(k) * 0.05)

    def forward(self, u):                                # u: [N,1]
        return torch.relu(u * self.n + self.b) @ self.m


def train_exp():
    """Fit exp on u in (U_LO,U_HI); return u-domain LUT params (d_u,s_u,t_u)."""
    u = torch.empty(N_SAMPLES, 1).uniform_(U_LO, U_HI)
    y = torch.exp(u).squeeze(1)
    # init breakpoints one-sided, denser near 0 (where exp bends hardest)
    lin = np.linspace(0.0, 1.0, N_NEURON)
    bp0 = -(lin ** 1.5) * abs(U_LO)                       # in (U_LO,0], dense at 0
    net = LutNet(bp0)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[4000, 6000], gamma=0.2)
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = (net(u) - y).abs().mean()                  # L1 loss (paper)
        loss.backward(); opt.step(); sched.step()

    # ---- transform net -> LUT (Eq. 6-7) ----------------------------------
    n = net.n.detach().double().numpy(); b = net.b.detach().double().numpy()
    m = net.m.detach().double().numpy()
    d_u = np.sort(-b / n)                                  # breakpoints -b_i/n_i (u-domain) [15]

    edges = np.concatenate([[-1e9], d_u, [1e9]])
    u_mid = 0.5 * (edges[:-1] + edges[1:])                # per-interval representative u
    s_u, t_u = [], []
    for uc in u_mid:
        act = (n * uc + b) > 0                            # active neurons at u=uc
        s_u.append(float((m * n * act).sum()))            # s_i = sum active m*n   (Eq. 7)
        t_u.append(float((m * b * act).sum()))            # t_i = sum active m*b   (Eq. 7)
    return d_u, np.array(s_u), np.array(t_u), loss.item()


def pick_shamt(s, t):
    """Largest sh in [0,SH_MAX] with round(s*2^sh) in 18b and round(t*2^sh) in 48b.
    Dead segments (s==t==0, flat tail) -> sh=0 (shift is irrelevant on zero)."""
    for sh in range(SH_MAX, -1, -1):
        sf, tf = round(s * (1 << sh)), round(t * (1 << sh))
        if abs(sf) <= S_MAX and abs(tf) <= T_MAX:
            return 0 if (sf == 0 and tf == 0) else sh
    return 0                                             # s or t already overflows @sh=0


def convert_layer(L, qp, d_u, s_u, t_u):
    """Fold layer scales into the shared u-domain LUT -> integer LUT + eval."""
    s_q = qp["act"][f"L{L}.q"]["scale"]
    s_k = qp["act"][f"L{L}.k"]["scale"]
    s_x = s_q * s_k / math.sqrt(D_HEAD)                  # LUT input scale (folds 1/sqrt(d))
    s_y = qp["act"][f"L{L}.probs"]["scale"]              # LUT output scale (1/127)
    # z = score - rowmax spans [-(max-min), 0]; bound by 2*|scores|_max
    sc_absmax = qp["act"][f"L{L}.scores"]["observed_absmax"]
    z_max = int(math.ceil(2.0 * sc_absmax / s_x))

    d_int = d_u / s_x                                     # breakpoints in integer z
    s_int = s_u * s_x / s_y
    t_int = t_u / s_y

    # single per-page shamt (6b): largest shift keeping every ACTIVE segment's M in 18b.
    # dead segments (s=t=0) impose no constraint -> excluded (else they'd force sh=0).
    per_seg = [pick_shamt(si, ti) for si, ti in zip(s_int, t_int)]
    sh = min(p for p, si, ti in zip(per_seg, s_int, t_int) if si != 0 or ti != 0)  # single int
    d_fix = np.round(d_int).astype(np.int64)             # int32
    s_fix = np.round(s_int * (1 << sh)).astype(np.int64)
    t_fix = np.round(t_int * (1 << sh)).astype(np.int64)

    # evaluate the TRUE fixed-point LUT (arith shift = floor) over z in [-z_max, 0]
    z   = np.arange(-z_max, 1, dtype=np.int64)
    idx = np.searchsorted(d_fix, z, side="right")        # 0..15
    raw = s_fix[idx] * z + t_fix[idx]                    # int64: |s|<2^17, |t|<2^47 -> no overflow
    yq  = raw >> sh                                       # single-page arithmetic right shift
    yhat  = np.clip(yq, 0.0, CLAMP).astype(np.float64)   # P' >= 0
    ytrue = np.clip(np.exp(z.astype(np.float64) * s_x) / s_y, 0.0, CLAMP)
    abs_err = np.abs(yhat - ytrue)

    return dict(L=L, s_x=s_x, s_y=s_y, x_max=z_max, x=z.astype(np.float64), yhat=yhat,
                ytrue=ytrue, abs_err=abs_err, d_int=d_int, s_int=s_int, t_int=t_int,
                d_fix=d_fix, s_fix=s_fix, t_fix=t_fix, sh=sh)


def main():
    qp = torch.load(QP_PATH)
    d_u, s_u, t_u, l1 = train_exp()
    print(f"u-domain exp fit: L1={l1:.4e}  breakpoints(u)="
          f"[{d_u.min():.2f}, {d_u.max():.2f}]")

    res = [convert_layer(L, qp, d_u, s_u, t_u) for L in (0, 1)]

    # ---- report + Verilog dump + JSON ------------------------------------
    lines = []
    report = {"function": "exp",
              "widths": {"d": 32, "s": W_S, "t": W_T, "sh": W_SH},
              "n_entries": 16, "domain_u": [U_LO, U_HI], "clamp": int(CLAMP),
              "u_fit_l1": l1, "layers": {}}
    for r in res:
        me, mn = float(r["abs_err"].max()), float(r["abs_err"].mean())
        rms = math.sqrt((r["abs_err"] ** 2).mean())
        print(f"L{r['L']}: z_range=[-{r['x_max']},0]  S_x={r['s_x']:.4e} S_y={r['s_y']:.4e}  "
              f"max_err={me:.3f} LSB  mean={mn:.3f}  rms={rms:.3f}  shift={r['sh']}")
        assert np.all(np.abs(r["s_fix"]) <= S_MAX), "s_fix overflows 18-bit"
        assert np.all(np.abs(r["t_fix"]) <= T_MAX), "t_fix overflows 48-bit"
        assert 0 <= r["sh"] <= SH_MAX, "shift overflows 6-bit"
        lines.append(f"// ---- L{r['L']} exp LUT  (S_x={r['s_x']:.6e} S_y={r['s_y']:.6e}, "
                     f"shift={r['sh']})  max_err={me:.3f} LSB")
        lines += [f"d[{i}] = 32'sd{int(v)};" for i, v in enumerate(r["d_fix"])]
        lines += [f"s[{i}] = 18'sd{int(v)};" for i, v in enumerate(r["s_fix"])]
        lines += [f"t[{i}] = 48'sd{int(v)};" for i, v in enumerate(r["t_fix"])]
        lines.append(f"shift = 6'd{int(r['sh'])};")
        lines.append("")
        report["layers"][f"L{r['L']}"] = {
            "s_x": float(r["s_x"]), "s_y": float(r["s_y"]), "x_max": int(r["x_max"]),
            "max_err_lsb": me, "mean_err_lsb": mn, "rms_err_lsb": rms, "shift": int(r["sh"]),
            "d": [int(v) for v in r["d_fix"]], "s": [int(v) for v in r["s_fix"]],
            "t": [int(v) for v in r["t_fix"]]}
    with open("lut_exp.txt", "w") as f:
        f.write("\n".join(lines))
    with open("lut_exp.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote lut_exp.txt, lut_exp.json")

    # ---- plot: shared u-domain fit ---------------------------------------
    u = np.linspace(U_LO, U_HI, 1000)
    idx = np.searchsorted(d_u, u, side="right")
    fig1, a = plt.subplots(1, 2, figsize=(11, 4))
    a[0].plot(u, np.exp(u), "k", lw=1.4, label="exp")
    a[0].plot(u, s_u[idx] * u + t_u[idx], "C1--", lw=1.0, label="16-entry LUT")
    for dv in d_u:
        a[0].axvline(dv, color="C0", alpha=0.25, lw=0.6)
    a[0].set_title("shared u-domain fit"); a[0].set_xlabel("u (real)"); a[0].legend(); a[0].grid(alpha=0.3)
    a[1].plot(u, (s_u[idx] * u + t_u[idx]) - np.exp(u), "C3", lw=0.9)
    a[1].set_title("u-domain error (LUT - exp)"); a[1].set_xlabel("u (real)"); a[1].grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig("exp_u.png", dpi=120)

    # ---- plot: per-layer integer LUT (valid range only) ------------------
    fig2, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, r in enumerate(res):
        ax[0][k].plot(r["x"], r["ytrue"], "k", lw=1.2, label="target")
        ax[0][k].plot(r["x"], r["yhat"], "C1--", lw=1.0, label="16-entry LUT")
        for dv in r["d_int"]:
            if -r["x_max"] <= dv <= 0:
                ax[0][k].axvline(dv, color="C0", alpha=0.25, lw=0.6)
        ax[0][k].set_xlim(-r["x_max"], 0)
        ax[0][k].set_title(f"L{r['L']} integer LUT  (max_err={r['abs_err'].max():.2f} LSB)")
        ax[0][k].set_xlabel("z = score-rowmax (int32)"); ax[0][k].set_ylabel("P' (int8)")
        ax[0][k].legend(); ax[0][k].grid(alpha=0.3)
        ax[1][k].plot(r["x"], r["yhat"] - r["ytrue"], "C3", lw=0.8)
        ax[1][k].set_xlim(-r["x_max"], 0)
        ax[1][k].set_title(f"L{r['L']} error (LUT - target), LSB")
        ax[1][k].set_xlabel("z (int32)"); ax[1][k].grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig("exp_fit.png", dpi=120)
    print("wrote exp_u.png, exp_fit.png")


if __name__ == "__main__":
    main()
