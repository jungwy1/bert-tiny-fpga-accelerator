"""
Plot the GELU NN-LUT target functions vs. plain GELU.

For each FFN layer the LUT must realize (int32 x -> int8 y):
    y(x) = gelu(S_x * x) / S_y
with
    S_x = qp["weight"]["L{i}.W_1"]["bias_scale"]   # W_1 accumulator scale
    S_y = qp["act"]["L{i}.ffn_act"]["scale"]       # W_2 input scale

Three curves are drawn so we can eyeball whether the integer-domain target is
any easier / harder to piecewise-linear-fit than plain GELU:
    (1) plain GELU(u) over the real pre-activation range
    (2) L0 target y(x) over its int32 x range
    (3) L1 target y(x) over its int32 x range

Run (from python/gelu/):  python plot_targets.py
Output: targets.png
"""
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

QP_PATH = "../quant_params.pt"


def gelu(u):
    # exact (erf) GELU, matches PyTorch default
    return 0.5 * u * (1.0 + torch.erf(u / math.sqrt(2.0)))


def main():
    qp = torch.load(QP_PATH)

    layers = []
    for L in (0, 1):
        s_x = qp["weight"][f"L{L}.W_1"]["bias_scale"]        # int x scale
        s_y = qp["act"][f"L{L}.ffn_act"]["scale"]            # int y scale
        absmax = qp["act"][f"L{L}.ffn_mid"]["observed_absmax"]  # real GELU-input range
        x_max = int(math.ceil(absmax / s_x))                 # int32 range of x
        layers.append(dict(L=L, s_x=s_x, s_y=s_y, absmax=absmax, x_max=x_max))
        print(f"L{L}: S_x={s_x:.4e} S_y={s_y:.4e} "
              f"real|u|max={absmax:.2f} -> int x in [-{x_max}, {x_max}]")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) plain GELU over the union of real ranges
    u_max = max(l["absmax"] for l in layers) * 1.05
    u = torch.linspace(-u_max, u_max, 1000)
    ax[0].plot(u.numpy(), gelu(u).numpy(), color="k")
    ax[0].set_title("plain GELU(u)")
    ax[0].set_xlabel("u (real pre-activation)")
    ax[0].set_ylabel("GELU(u)")
    ax[0].grid(alpha=0.3)

    # (2),(3) integer-domain target y(x) per layer
    for k, l in enumerate(layers, start=1):
        x = torch.arange(-l["x_max"], l["x_max"] + 1, dtype=torch.float64)
        y = gelu(x * l["s_x"]) / l["s_y"]                    # target (pre-round/clamp)
        ax[k].plot(x.numpy(), y.numpy(), color="C0")
        ax[k].axhline(127, color="r", ls="--", lw=0.8)
        ax[k].axhline(-127, color="r", ls="--", lw=0.8)
        ax[k].set_title(f"L{l['L']} target  y(x)=gelu(S_x·x)/S_y")
        ax[k].set_xlabel("x (int32 accumulator)")
        ax[k].set_ylabel("y (int8, pre-clamp)")
        ax[k].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("targets.png", dpi=120)
    print("wrote targets.png")


if __name__ == "__main__":
    main()