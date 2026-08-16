"""
NN-LUT for the softmax EXP, per NN-LUT (Yu et al., 2021), Table 1 (Exp).

Softmax uses the max-subtraction trick: exp(scores_i - max), so the exp argument
is ALWAYS <= 0 -- a one-sided domain u in (U_LO, 0].  Fit exp(u) with L1 loss,
then fold per-layer scales in as constants (u = S_x*x, y = exp(u)/S_y):
    d_int = d_u/S_x,   s_int = s_u*S_x/S_y,   t_int = t_u/S_y
    S_x = S_scores      = qp["act"]["L{i}.scores"]["scale"]   (exp input = scores-max)
    S_y = EXP_OUT_SCALE = free choice -- it CANCELS in softmax's exp/sum(exp);
          1/127 chosen so exp(0)=1 -> 127 (int8 HW).
One shared exp curve -> per-layer integer LUT.  Left tail -> 0 handled by the
outer interval; output clamped to [0,127] (exp >= 0).

Run (from python/softmax/):  python train_exp.py
Output: exp_u.png, fit.png, lut_exp.txt (Verilog d/s/t, SH=16).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

QP_PATH   = "../quant_params.pt"
N_NEURON  = 15
SH        = 13              # < GELU's 16: exp has a steep slope (s_int ~ S_scores*127 ~ 10),
                            #   so s_fix must stay in 18b -> 2^SH <= 2^17/s_int_max ~ 1e4 -> SH<=13
U_LO      = -9.0             # train domain (U_LO, 0];  exp(-9)=1.2e-4 << 1 LSB
EVAL_LO   = -16.0            # honest error range (scores-max can go this low; exp underflows)
EXP_OUT_SCALE = 1.0 / 127    # free (cancels in normalization); exp(0)=1 -> 127
N_SAMPLES = 100_000
EPOCHS    = 8000
LR        = 5e-3
torch.manual_seed(0)


class LutNet(nn.Module):
    """1-hidden ReLU net (Eq. 5):  NN(u) = sum_i m_i * relu(n_i*u + b_i)."""
    def __init__(self, bp0):
        super().__init__()
        k = len(bp0)
        self.n = nn.Parameter(torch.ones(k) + torch.randn(k) * 0.05)
        self.b = nn.Parameter(-torch.as_tensor(bp0, dtype=torch.float32))   # bp = -b/n ~ bp0
        self.m = nn.Parameter(torch.randn(k) * 0.05)

    def forward(self, u):
        return torch.relu(u * self.n + self.b) @ self.m


def train_exp():
    """Fit exp on u in (U_LO, 0]; return u-domain LUT params (d_u, s_u, t_u)."""
    u = torch.empty(N_SAMPLES, 1).uniform_(U_LO, 0.0)
    y = torch.exp(u).squeeze(1)

    # one-sided breakpoint init in (U_LO,0), dense near 0 (where exp bends fastest)
    lin = np.linspace(0.0, 1.0, N_NEURON)
    bp0 = -(lin ** 2.5) * abs(U_LO)                       # 0 -> 0 (dense), 1 -> U_LO (exp steep near 0)
    net = LutNet(bp0)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[4000, 6000], gamma=0.2)
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = (net(u) - y).abs().mean()                  # L1 loss
        loss.backward(); opt.step(); sched.step()

    n = net.n.detach().double().numpy(); b = net.b.detach().double().numpy()
    m = net.m.detach().double().numpy()
    d_u = np.sort(-b / n)

    edges = np.concatenate([[-1e9], d_u, [1e9]])
    u_mid = 0.5 * (edges[:-1] + edges[1:])
    s_u, t_u = [], []
    for uc in u_mid:
        act = (n * uc + b) > 0
        s_u.append(float((m * n * act).sum()))
        t_u.append(float((m * b * act).sum()))
    return d_u, np.array(s_u), np.array(t_u), loss.item()


def convert_layer(L, qp, d_u, s_u, t_u):
    s_x = qp["act"][f"L{L}.scores"]["scale"]              # exp input scale (= S_scores)
    s_y = EXP_OUT_SCALE
    x_lo = int(math.floor(EVAL_LO / s_x))                 # most-negative integer input evaluated

    d_int = d_u / s_x
    s_int = s_u * s_x / s_y
    t_int = t_u / s_y

    x = np.arange(x_lo, 1, dtype=np.float64)              # x in [x_lo, 0]  (u <= 0)
    idx = np.searchsorted(d_int, x, side="right")
    yhat = np.clip(s_int[idx] * x + t_int[idx], 0.0, 127.0)          # exp >= 0
    ytrue = np.clip(np.exp(x * s_x) / s_y, 0.0, 127.0)
    abs_err = np.abs(yhat - ytrue)

    d_fix = np.round(d_int).astype(np.int64)
    s_fix = np.round(s_int * (1 << SH)).astype(np.int64)
    t_fix = np.round(t_int * (1 << SH)).astype(np.int64)
    return dict(L=L, s_x=s_x, s_y=s_y, x_lo=x_lo, x=x, yhat=yhat, ytrue=ytrue,
                abs_err=abs_err, d_int=d_int, s_int=s_int, t_int=t_int,
                d_fix=d_fix, s_fix=s_fix, t_fix=t_fix)


def main():
    qp = torch.load(QP_PATH)
    d_u, s_u, t_u, l1 = train_exp()
    print(f"u-domain exp fit: L1={l1:.4e}  breakpoints(u)=[{d_u.min():.2f}, {d_u.max():.2f}]")

    res = [convert_layer(L, qp, d_u, s_u, t_u) for L in (0, 1)]

    lines = []
    for r in res:
        me, mn = r["abs_err"].max(), r["abs_err"].mean()
        rms = math.sqrt((r["abs_err"] ** 2).mean())
        print(f"L{r['L']}: x in [{r['x_lo']},0]  S_x={r['s_x']:.4e} S_y={r['s_y']:.4e}  "
              f"max_err={me:.3f} LSB  mean={mn:.3f}  rms={rms:.3f}")
        assert np.all(np.abs(r["s_fix"]) <= (1 << 17) - 1), "s_fix overflows 18-bit"
        lines.append(f"// ---- L{r['L']} EXP LUT  (S_x={r['s_x']:.6e} S_y={r['s_y']:.6e} "
                     f"SH={SH})  max_err={me:.3f} LSB")
        lines += [f"d[{i}] = 32'sd{int(v)};" for i, v in enumerate(r["d_fix"])]
        lines += [f"s[{i}] = 18'sd{int(v)};" for i, v in enumerate(r["s_fix"])]
        lines += [f"t[{i}] = 32'sd{int(v)};" for i, v in enumerate(r["t_fix"])]
        lines.append("")
    with open("lut_exp.txt", "w") as f:
        f.write("\n".join(lines))
    print("wrote lut_exp.txt")

    # ---- plot: shared u-domain fit ---------------------------------------
    u = np.linspace(U_LO, 0.0, 1000)
    idx = np.searchsorted(d_u, u, side="right")
    fig1, a = plt.subplots(1, 2, figsize=(11, 4))
    a[0].plot(u, np.exp(u), "k", lw=1.4, label="exp")
    a[0].plot(u, s_u[idx] * u + t_u[idx], "C1--", lw=1.0, label="16-entry LUT")
    for dv in d_u:
        a[0].axvline(dv, color="C0", alpha=0.25, lw=0.6)
    a[0].set_title("shared u-domain fit"); a[0].set_xlabel("u = scores-max (<=0)"); a[0].legend(); a[0].grid(alpha=0.3)
    a[1].plot(u, (s_u[idx] * u + t_u[idx]) - np.exp(u), "C3", lw=0.9)
    a[1].set_title("u-domain error (LUT - exp)"); a[1].set_xlabel("u"); a[1].grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig("exp_u.png", dpi=120)

    # ---- plot: per-layer integer LUT -------------------------------------
    fig2, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, r in enumerate(res):
        ax[0][k].plot(r["x"], r["ytrue"], "k", lw=1.2, label="target")
        ax[0][k].plot(r["x"], r["yhat"], "C1--", lw=1.0, label="16-entry LUT")
        for dv in r["d_int"]:
            if r["x_lo"] <= dv <= 0:
                ax[0][k].axvline(dv, color="C0", alpha=0.25, lw=0.6)
        ax[0][k].set_xlim(r["x_lo"], 0)
        ax[0][k].set_title(f"L{r['L']} integer EXP LUT  (max_err={r['abs_err'].max():.2f} LSB)")
        ax[0][k].set_xlabel("x = scores_int - max_int"); ax[0][k].set_ylabel("y (int8)")
        ax[0][k].legend(); ax[0][k].grid(alpha=0.3)
        ax[1][k].plot(r["x"], r["yhat"] - r["ytrue"], "C3", lw=0.8)
        ax[1][k].set_xlim(r["x_lo"], 0)
        ax[1][k].set_title(f"L{r['L']} error (LUT - target), LSB")
        ax[1][k].set_xlabel("x"); ax[1][k].grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig("fit.png", dpi=120)
    print("wrote exp_u.png, fit.png")


if __name__ == "__main__":
    main()
