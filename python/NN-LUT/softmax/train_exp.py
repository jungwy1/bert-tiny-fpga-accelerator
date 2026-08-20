import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QP_PATH   = "../../quant_params.pt"
N_NEURON  = 15               # -> 16 intervals
W_S       = 18               # slope width  (signed)
W_T       = 48               # offset width (signed)
W_SH      = 6                # shamt width  -> sh in 0..63
S_MAX     = (1 << (W_S - 1)) - 1
T_MAX     = (1 << (W_T - 1)) - 1
SH_MAX    = (1 << W_SH) - 1
CLAMP     = 127.0            # int8 output clamp
U_LO, U_HI = -5.0, 5.0       # GELU training domain (paper Table 1)
N_SAMPLES = 100_000
EPOCHS    = 8000
LR        = 5e-3
torch.manual_seed(0)

class LutNet(nn.Module):
    def __init__(self, bp0):
        super().__init__()
        k = len(bp0)
        self.n = nn.Parameter(torch.ones(k) + torch.randn(k) * 0.05)        # n_i > 0 (GELU increasing)
        self.b = nn.Parameter(-torch.as_tensor(bp0, dtype=torch.float32))   # bp = -b_i/n_i ~ bp0
        self.m = nn.Parameter(torch.randn(k) * 0.05)

    def forward(self, u):                                # u: [N,1]
        return torch.relu(u * self.n + self.b) @ self.m


def train_gelu():
    u = torch.empty(N_SAMPLES, 1).uniform_(U_LO, U_HI)
    y = F.gelu(u).squeeze(1)
    lin = np.linspace(-1.0, 1.0, N_NEURON)
    bp0 = np.sign(lin) * (np.abs(lin) ** 1.5) * U_HI      # in (U_LO,U_HI), dense at 0
    net = LutNet(bp0)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[4000, 6000], gamma=0.2)
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = (net(u) - y).abs().mean()                  # L1 loss (paper)
        loss.backward(); opt.step(); sched.step()

    # ---- transform net -> LUT ----------------------------------
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
    s_x = qp["weight"][f"L{L}.W_1"]["bias_scale"]         # LUT input scale
    s_y = qp["act"][f"L{L}.ffn_act"]["scale"]             # LUT output scale
    x_max = int(math.ceil(qp["act"][f"L{L}.ffn_mid"]["observed_absmax"] / s_x))

    d_int = d_u / s_x                                     # breakpoints in integer x
    s_int = s_u * s_x / s_y
    t_int = t_u / s_y

    # per-segment shamt (6b) + fixed-point slope(18b)/offset(48b)
    sh    = np.array([pick_shamt(si, ti) for si, ti in zip(s_int, t_int)], dtype=np.int64)
    d_fix = np.round(d_int).astype(np.int64)             # int32
    s_fix = np.array([round(si * (1 << shi)) for si, shi in zip(s_int, sh)], dtype=np.int64)
    t_fix = np.array([round(ti * (1 << shi)) for ti, shi in zip(t_int, sh)], dtype=np.int64)

    # evaluate the TRUE fixed-point LUT (arith shift = floor) over the valid range
    x   = np.arange(-x_max, x_max + 1, dtype=np.int64)
    idx = np.searchsorted(d_fix, x, side="right")        # 0..15
    raw = s_fix[idx] * x + t_fix[idx]                    # int64: |s|<2^17, |t|<2^47 -> no overflow
    yq  = raw >> sh[idx]                                  # arithmetic right shift
    yhat  = np.clip(yq, -CLAMP, CLAMP).astype(np.float64)
    ytrue = np.clip(F.gelu(torch.from_numpy(x.astype(np.float64) * s_x)).numpy() / s_y,
                    -CLAMP, CLAMP)
    abs_err = np.abs(yhat - ytrue)

    return dict(L=L, s_x=s_x, s_y=s_y, x_max=x_max, x=x.astype(np.float64), yhat=yhat,
                ytrue=ytrue, abs_err=abs_err, d_int=d_int, s_int=s_int, t_int=t_int,
                d_fix=d_fix, s_fix=s_fix, t_fix=t_fix, sh=sh)


def main():
    qp = torch.load(QP_PATH)
    d_u, s_u, t_u, l1 = train_gelu()
    print(f"u-domain GELU fit: L1={l1:.4e}  breakpoints(u)="
          f"[{d_u.min():.2f}, {d_u.max():.2f}]")

    res = [convert_layer(L, qp, d_u, s_u, t_u) for L in (0, 1)]

    # ---- report + Verilog dump + JSON ------------------------------------
    lines = []
    report = {"function": "gelu",
              "widths": {"d": 32, "s": W_S, "t": W_T, "sh": W_SH},
              "n_entries": 16, "domain_u": [U_LO, U_HI], "clamp": int(CLAMP),
              "u_fit_l1": l1, "layers": {}}
    for r in res:
        me, mn = float(r["abs_err"].max()), float(r["abs_err"].mean())
        rms = math.sqrt((r["abs_err"] ** 2).mean())
        print(f"L{r['L']}: x_range=+-{r['x_max']}  S_x={r['s_x']:.4e} S_y={r['s_y']:.4e}  "
              f"max_err={me:.3f} LSB  mean={mn:.3f}  rms={rms:.3f}  sh={r['sh'].min()}..{r['sh'].max()}")
        assert np.all(np.abs(r["s_fix"]) <= S_MAX), "s_fix overflows 18-bit"
        assert np.all(np.abs(r["t_fix"]) <= T_MAX), "t_fix overflows 48-bit"
        assert np.all((r["sh"] >= 0) & (r["sh"] <= SH_MAX)), "sh overflows 6-bit"
        lines.append(f"// ---- L{r['L']} GELU LUT  (S_x={r['s_x']:.6e} S_y={r['s_y']:.6e}, "
                     f"per-seg shamt)  max_err={me:.3f} LSB")
        lines += [f"d[{i}] = 32'sd{int(v)};" for i, v in enumerate(r["d_fix"])]
        lines += [f"s[{i}] = 18'sd{int(v)};" for i, v in enumerate(r["s_fix"])]
        lines += [f"t[{i}] = 48'sd{int(v)};" for i, v in enumerate(r["t_fix"])]
        lines += [f"sh[{i}] = 6'd{int(v)};" for i, v in enumerate(r["sh"])]
        lines.append("")
        report["layers"][f"L{r['L']}"] = {
            "s_x": float(r["s_x"]), "s_y": float(r["s_y"]), "x_max": int(r["x_max"]),
            "max_err_lsb": me, "mean_err_lsb": mn, "rms_err_lsb": rms,
            "d": [int(v) for v in r["d_fix"]], "s": [int(v) for v in r["s_fix"]],
            "t": [int(v) for v in r["t_fix"]], "sh": [int(v) for v in r["sh"]]}
    with open("lut_gelu.txt", "w") as f:
        f.write("\n".join(lines))
    with open("lut_gelu.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote lut_gelu.txt, lut_gelu.json")

    # ---- plot: shared u-domain fit ---------------------------------------
    u = np.linspace(U_LO, U_HI, 1000)
    idx = np.searchsorted(d_u, u, side="right")
    fig1, a = plt.subplots(1, 2, figsize=(11, 4))
    a[0].plot(u, F.gelu(torch.from_numpy(u)).numpy(), "k", lw=1.4, label="GELU")
    a[0].plot(u, s_u[idx] * u + t_u[idx], "C1--", lw=1.0, label="16-entry LUT")
    for dv in d_u:
        a[0].axvline(dv, color="C0", alpha=0.25, lw=0.6)
    a[0].set_title("shared u-domain fit"); a[0].set_xlabel("u (real)"); a[0].legend(); a[0].grid(alpha=0.3)
    a[1].plot(u, (s_u[idx] * u + t_u[idx]) - F.gelu(torch.from_numpy(u)).numpy(), "C3", lw=0.9)
    a[1].set_title("u-domain error (LUT - GELU)"); a[1].set_xlabel("u (real)"); a[1].grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig("gelu_u.png", dpi=120)

    # ---- plot: per-layer integer LUT (valid range only) ------------------
    fig2, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, r in enumerate(res):
        ax[0][k].plot(r["x"], r["ytrue"], "k", lw=1.2, label="target")
        ax[0][k].plot(r["x"], r["yhat"], "C1--", lw=1.0, label="16-entry LUT")
        for dv in r["d_int"]:
            if -r["x_max"] <= dv <= r["x_max"]:
                ax[0][k].axvline(dv, color="C0", alpha=0.25, lw=0.6)
        ax[0][k].set_xlim(-r["x_max"], r["x_max"])
        ax[0][k].set_title(f"L{r['L']} integer LUT  (max_err={r['abs_err'].max():.2f} LSB)")
        ax[0][k].set_xlabel("x (int32)"); ax[0][k].set_ylabel("y (int8)")
        ax[0][k].legend(); ax[0][k].grid(alpha=0.3)
        ax[1][k].plot(r["x"], r["yhat"] - r["ytrue"], "C3", lw=0.8)
        ax[1][k].set_xlim(-r["x_max"], r["x_max"])
        ax[1][k].set_title(f"L{r['L']} error (LUT - target), LSB")
        ax[1][k].set_xlabel("x (int32)"); ax[1][k].grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig("fit.png", dpi=120)
    print("wrote gelu_u.png, fit.png")


if __name__ == "__main__":
    main()
