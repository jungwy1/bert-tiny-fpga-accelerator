"""
Calibrate the float reference model and export all integer parameters.

Scheme (symmetric everywhere -- no zero-points, matching int_ops.py and
I-BERT / SwiftTron / FQ-BERT):
  activations : INT8  symmetric per-tensor   S = max|x| / 127
  weights     : INT8  symmetric per-tensor   S = max|W| / 127
  LayerNorm   : gamma INT8, beta INT32 (both symmetric per-tensor)
  bias        : INT32 symmetric, in the accumulator domain  S = S_in * S_w

Special case: `L{i}.probs` (softmax output) has a mathematically known range
[0, 1], so its scale is FIXED at 1/255 (unsigned 8-bit) rather than calibrated.

Run (from python/):  python export_params.py
Output: quant_params.pt
"""

import math
import torch

from model_float import GoldenBertTiny
from data import (load_state_dict, load_tokenizer, load_sst2,
                  run_calibration, CALIB_SAMPLES)

OUT_PATH = "quant_params.pt"
W_QMAX = 127                      # INT8 symmetric
B_QMAX = 2 ** 31 - 1              # INT32 symmetric (bias, LayerNorm beta)
G_QMAX = 127                      # INT8 symmetric (LayerNorm gamma)

# Activations whose range is known a priori, so their scale is FIXED rather than
# calibrated -- and the requantization fuses into the op's final division.
FIXED_ACT_SCALES = {
    "probs":    1.0 / 255,      # softmax output in [0, 1]  (unsigned 8-bit)
    "pool_out": 1.0 / 127,      # tanh output in [-1, 1]    (signed 8-bit)
}


# ---------------------------------------------------------------- observer
class Observer:
    """Per-tensor symmetric min/max observer: S = max|x| / qmax."""
    def __init__(self):
        self.absmax = 0.0

    def observe(self, x):
        self.absmax = max(self.absmax, x.detach().abs().max().item())

    def scale(self, qmax=W_QMAX):
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

    # 1) calibrate every tapped activation on the float model
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

    # 2) matmul weights + INT32 bias (bias lives in the accumulator domain S_in*S_w)
    ops = {}
    for name, prefix, a_in, a_out in matmul_ops():
        W = sd[prefix + ".weight"]
        b = sd[prefix + ".bias"]
        q_w, s_w = quant_sym(W, W_QMAX)
        s_a = act[a_in]["scale"]
        s_acc = s_a * s_w                                 # accumulator scale
        q_b = torch.round(b.double() / s_acc).clamp(-B_QMAX, B_QMAX).to(torch.int64)
        ops[name] = {"w_int8": q_w.to(torch.int8), "w_scale": s_w,
                     "bias_int32": q_b.to(torch.int32), "bias_scale": s_acc,
                     "in": a_in, "out": a_out}

    # 3) LayerNorm gamma (INT8) / beta (INT32)
    for name, prefix, a_in, a_out in layernorm_params():
        q_g, s_g = quant_sym(sd[prefix + ".weight"], G_QMAX)
        q_b, s_b = quant_sym(sd[prefix + ".bias"], B_QMAX)
        ops[name] = {"gamma_int8": q_g.to(torch.int8), "gamma_scale": s_g,
                     "beta_int32": q_b.to(torch.int32), "beta_scale": s_b,
                     "in": a_in, "out": a_out}

    # 4) embedding tables (INT8 lookup)
    for name, key in EMBED_TABLES:
        q_t, s_t = quant_sym(sd[key], W_QMAX)
        ops[name] = {"table_int8": q_t.to(torch.int8), "scale": s_t, "out": "emb_out"}

    torch.save({"act": act, "op": ops}, OUT_PATH)

    # ------------------------------------------------------------- report
    print(f"calibration: {CALIB_SAMPLES} sentences\n")
    print(f"{'activation':16s} {'scale':>12s} {'absmax':>10s}  {'bits used':>10s}  src")
    for n, d in act.items():
        used = math.ceil(math.log2(d["observed_absmax"] / d["scale"] + 1)) if d["observed_absmax"] else 0
        print(f"{n:16s} {d['scale']:12.3e} {d['observed_absmax']:10.3f}  {used:>10d}  {d['source']}")
    print(f"\nexported {len(act)} activations + {len(ops)} ops -> {OUT_PATH}")


if __name__ == "__main__":
    main()
