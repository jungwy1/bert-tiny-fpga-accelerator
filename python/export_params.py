"""
Calibrate the float reference model and export all integer parameters.

Scheme (symmetric everywhere -- no zero-points, matching int_ops.py and
I-BERT / SwiftTron / FQ-BERT):
  activations : INT8  symmetric per-tensor   S = max|x| / 127   (A8)
  matmul wgt  : INT4  symmetric per-tensor   S = max|W| / 7      (W4, per quant_sweep.md)
  LayerNorm   : gamma INT8, beta INT32 (both symmetric per-tensor)
  embeddings  : INT8  (lookup tables, handled in PS)
  bias        : INT32 symmetric, in the accumulator domain  S = S_in * S_w

Special case: `L{i}.probs` (softmax output) has a mathematically known range
[0, 1], so its scale is FIXED at 1/127 (signed 8-bit, 0..127) rather than calibrated.

Run (from python/):  python export_params.py
Output: quant_params.pt
"""

import math
import torch

from model_float import GoldenBertTiny
from data import (load_state_dict, load_tokenizer, load_sst2,
                  run_calibration, CALIB_SAMPLES)

OUT_PATH = "quant_params.pt"
A_QMAX = 127                      # INT8 symmetric (activations, embeddings)
WGT_QMAX = 7                      # INT4 symmetric (matmul weights -- A8W4)
B_QMAX = 2 ** 31 - 1              # INT32 symmetric (bias, LayerNorm beta)
G_QMAX = 127                      # INT8 symmetric (LayerNorm gamma)
RSQRT_FRAC = 22                   # x_hat fixed-point scale in integer LN (must match train_rsqrt.py)

# Activations whose range is known a priori, so their scale is FIXED rather than
# calibrated -- and the requantization fuses into the op's final division.
FIXED_ACT_SCALES = {
    "probs":    1.0 / 127,      # softmax output in [0, 1]  (signed 8-bit, 0..127)
    "pool_out": 1.0 / 127,      # tanh output in [-1, 1]    (signed 8-bit)
}


# ---------------------------------------------------------------- observer
class Observer:
    """Per-tensor symmetric min/max observer: S = max|x| / qmax."""
    def __init__(self):
        self.absmax = 0.0

    def observe(self, x):
        self.absmax = max(self.absmax, x.detach().abs().max().item())

    def scale(self, qmax=A_QMAX):
        return max(self.absmax / qmax, 1e-12)     # guard against an all-zero tensor


# ------------------------------------------------------------- quantizers
def quant_sym(t, qmax):
    """Symmetric per-tensor quantization. Returns (q [int64], scale [float]).

    Computed in float64: for INT32 targets, float32 cannot represent 2**31-1
    exactly, so the clamp would silently miss and the cast would wrap to -2**31.
    """
    t = t.double()
    s = max(t.abs().max().item() / qmax, 1e-12)
    return torch.round(t / s).clamp(-qmax, qmax).to(torch.int64), s


# ------------------------------------------------------------- op tables
def matmul_ops():
    """(export name, state_dict prefix, input activation, output activation)."""
    ops = []
    for L in (0, 1):
        p = f"bert.encoder.layer.{L}."
        prev = "emb_out" if L == 0 else f"L{L-1}.ln2_out"
        ops += [
            (f"L{L}.W_q", p + "attention.self.query",   prev,            f"L{L}.q"),
            (f"L{L}.W_k", p + "attention.self.key",     prev,            f"L{L}.k"),
            (f"L{L}.W_v", p + "attention.self.value",   prev,            f"L{L}.v"),
            (f"L{L}.W_o", p + "attention.output.dense", f"L{L}.ctx",     f"L{L}.attn_out"),
            (f"L{L}.W_1", p + "intermediate.dense",     f"L{L}.ln1_out", f"L{L}.ffn_mid"),
            (f"L{L}.W_2", p + "output.dense",           f"L{L}.ffn_act", f"L{L}.ffn_out"),
        ]
    ops += [
        ("W_pool", "bert.pooler.dense", "pool_in",  "pool_mid"),   # output is pre-tanh
        ("W_cls",  "classifier",        "pool_out", "logits"),
    ]
    return ops


def layernorm_params():
    """(export name, state_dict prefix, input activation, output activation).
    The input scale is what i_layernorm needs, so record it like the matmuls do."""
    lns = [("emb_ln", "bert.embeddings.LayerNorm", "emb_sum", "emb_out")]
    for L in (0, 1):
        p = f"bert.encoder.layer.{L}."
        lns += [(f"L{L}.ln1", p + "attention.output.LayerNorm", f"L{L}.res1", f"L{L}.ln1_out"),
                (f"L{L}.ln2", p + "output.LayerNorm",           f"L{L}.res2", f"L{L}.ln2_out")]
    return lns


EMBED_TABLES = [("emb_word", "bert.embeddings.word_embeddings.weight"),
                ("emb_pos",  "bert.embeddings.position_embeddings.weight"),
                ("emb_type", "bert.embeddings.token_type_embeddings.weight")]


# ------------------------------------------------------------------ main
def main():
    sd = load_state_dict()
    tok = load_tokenizer()
    train, _ = load_sst2()

    # 1) calibrate every tapped activation on the float model.
    obs = {}
    def tap(name, x):
        obs.setdefault(name, Observer()).observe(x)
        return x

    model = GoldenBertTiny(sd, tap=tap)
    run_calibration(model, tok, train, n=CALIB_SAMPLES)

    act = {}
    for name, o in obs.items():
        key = name.split(".")[-1]
        if key in FIXED_ACT_SCALES:                       # softmax output: known range
            act[name] = {"scale": FIXED_ACT_SCALES[key], "observed_absmax": o.absmax,
                         "source": "fixed"}
        else:
            act[name] = {"scale": o.scale(), "observed_absmax": o.absmax,
                         "source": "calibrated"}

    # 1b) 2nd pass: NQ-S^2 range on the INT8 residual (how HW forms the LN input).
    #     LayerNorm: x_hat = (Nx-S)/sqrt(NQ-S^2). The residual sum x+attn is FORMED at the skip
    #     branch's scale S_res ( (S_a/S_res)*q_a + q_res ), not res1's own calibrated scale -- so
    #     N,Q accumulate on the sum quantized at S_res (int32, may exceed int8 since S_res<S_res1).
    RES_FORM_SCALE = {"L0.res1": "emb_out",    "L1.res1": "L0.ln2_out",     # skip = layer input
                      "L0.res2": "L0.ln1_out", "L1.res2": "L1.ln1_out"}     # skip = ln1_out
    nqs2 = {}                                     # LN input name -> [den_min, den_max]  (integer domain)
    def tap_nqs2(name, x):
        if name in RES_FORM_SCALE:
            xi = torch.round(x / act[RES_FORM_SCALE[name]]["scale"])          # sum @ skip scale (int32)
            N = xi.shape[-1]                      # 128 features
            s_row = xi.sum(-1); q_row = (xi * xi).sum(-1)
            den = (N * q_row - s_row ** 2).flatten()          # per-row NQ-S^2 (integer) >= 0
            den = den[den > 0]                                # drop exact-constant rows (guarded in HW)
            if den.numel():
                r = nqs2.setdefault(name, [float("inf"), 0.0])
                r[0] = min(r[0], den.min().item())
                r[1] = max(r[1], den.max().item())
        return x

    run_calibration(GoldenBertTiny(sd, tap=tap_nqs2), tok, train, n=CALIB_SAMPLES)

    # 2) matmul weights + INT32 bias (bias lives in the accumulator domain S_in*S_w)
    weights = {}
    for name, prefix, a_in, a_out in matmul_ops():
        W = sd[prefix + ".weight"]
        b = sd[prefix + ".bias"]
        q_w, s_w = quant_sym(W, WGT_QMAX)                 # INT4 weight (A8W4)
        s_a = act[a_in]["scale"]
        s_acc = s_a * s_w                                 # accumulator scale
        q_b = torch.round(b.double() / s_acc).clamp(-B_QMAX, B_QMAX).to(torch.int64)
        weights[name] = {"w_int4": q_w.to(torch.int8), "w_scale": s_w,   # int8 dtype, values [-7,7]
                         "bias_int32": q_b.to(torch.int32), "bias_scale": s_acc,
                         "in": a_in, "out": a_out}

    # 3) LayerNorm gamma (INT8) / beta (INT32) + rsqrt LUT domain (NQ-S^2, integer x)
    for name, prefix, a_in, a_out in layernorm_params():
        q_g, s_g = quant_sym(sd[prefix + ".weight"], G_QMAX)
        q_b, s_b = quant_sym(sd[prefix + ".bias"], B_QMAX)
        entry = {"gamma_int8": q_g.to(torch.int8), "gamma_scale": s_g,
                 "beta_int32": q_b.to(torch.int32), "beta_scale": s_b,
                 "in": a_in, "out": a_out}
        if a_in in nqs2:                          # encoder LN (res1/res2); emb_ln is host-side
            entry["nqs2_int_min"] = int(nqs2[a_in][0])    # den already on the int8 residual
            entry["nqs2_int_max"] = int(nqs2[a_in][1])
            # beta in the gamma*x_hat accumulator domain (scale S_g * 2^-FRAC) for the NN-LUT LN
            # datapath: out = (gamma_int8 * x_hat_fixed + beta_nnlut) * M,  M = S_g*2^-FRAC / S_out.
            s_b_nn = s_g / (1 << RSQRT_FRAC)
            q_b_nn = torch.round(sd[prefix + ".bias"].double() / s_b_nn).clamp(-B_QMAX, B_QMAX).to(torch.int64)
            entry["beta_nnlut"] = q_b_nn.to(torch.int32)
            entry["beta_scale_nnlut"] = s_b_nn
        weights[name] = entry

    # 4) embedding tables (INT8 lookup -- handled in PS)
    for name, key in EMBED_TABLES:
        q_t, s_t = quant_sym(sd[key], A_QMAX)
        weights[name] = {"table_int8": q_t.to(torch.int8), "scale": s_t, "out": "emb_out"}

    torch.save({"act": act, "weight": weights}, OUT_PATH)

    # ------------------------------------------------------------- report
    print(f"calibration: {CALIB_SAMPLES} sentences\n")
    print(f"{'activation':16s} {'scale':>12s} {'absmax':>10s}  {'bits used':>10s}  src")
    for n, d in act.items():
        used = math.ceil(math.log2(d["observed_absmax"] / d["scale"] + 1)) if d["observed_absmax"] else 0
        print(f"{n:16s} {d['scale']:12.3e} {d['observed_absmax']:10.3f}  {used:>10d}  {d['source']}")
    print(f"\nexported {len(act)} activations + {len(weights)} weights -> {OUT_PATH}")

    print(f"\n{'layernorm':10s} {'NQ-S^2 min (int)':>18s} {'NQ-S^2 max (int)':>18s}")
    for name, _, a_in, _ in layernorm_params():
        e = weights[name]
        if "nqs2_int_min" in e:
            print(f"{name:10s} {e['nqs2_int_min']:>18d} {e['nqs2_int_max']:>18d}")


if __name__ == "__main__":
    main()
