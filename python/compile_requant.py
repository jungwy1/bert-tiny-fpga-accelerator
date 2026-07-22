"""
Turn calibrated scales into the integer constants the hardware actually runs on.

quant_params.pt holds *observations* (one scale per activation). Hardware never
uses a float scale -- it needs, at each point, a dyadic multiplier (M_int, shift)
so a rescale is one integer multiply plus a right shift (SwiftTron Eq. 2), plus
the precomputed constants each nonlinear unit is wired with.

Four kinds of scaling point:
  requant  : accumulator -> next INT8 domain.   M = S_acc / S_target
  align    : two addends at different scales must match before adding (residual,
             embedding sum).                    M = S_operand / S_common
  fused    : the op's own final division already lands on the target scale, so
             there is no separate multiplier (softmax -> 1/255, tanh -> 1/127).
  none     : logits (argmax is scale-invariant).

Nonlinear output scales are DERIVED from int_ops' formulas, never calibrated:
  GELU      S_out = S_in / (2 * ERF_QMAX)     (erf is requantized to a fixed scale,
                                               so this is linear in S, not S^3)
  softmax   S_out = 1/255                     (fixed)
  tanh      S_out = 1/127                     (fixed)
  LayerNorm S_out = S_gamma / K               (input scale cancels in the norm)

Run (from python/):  python compile_requant.py
Output: requant_params.pt
"""

import math
import torch

from int_ops import (to_dyadic, ERF_A, ERF_B, ERF_C, EXP_A, EXP_B, EXP_C,
                     LN2, ERF_QMAX, EXP_QMAX)

IN_PATH = "quant_params.pt"
OUT_PATH = "requant_params.pt"
K_LAYERNORM = 2 ** 10          # i_layernorm's resolution factor (must match int_model)
D_HEAD = 64                    # attention head dim; 1/sqrt(d) folds into the softmax scale

# pool_in is row 0 of L1.ln2_out -- the same buffer, so it inherits that scale.
# Its separately observed range is informational only.
ALIAS = {"pool_in": "L1.ln2_out"}


def poly_consts(S, a, b, c):
    """The compile-time constants i_poly is wired with, for input scale S."""
    return {"q_b": math.floor(b / S), "q_c": math.floor(c / (a * S ** 2)),
            "S_L": a * S ** 2}


def main():
    p = torch.load(IN_PATH, weights_only=True)
    act, op = p["act"], p["op"]
    S = lambda n: act[ALIAS.get(n, n)]["scale"]

    requant, align, fused, nonlin = {}, {}, {}, {}

    def add_requant(name, s_from, s_to, kind, note=""):
        # Size M_int so acc*M_int still fits int64; acc width comes from the range
        # we actually observed at this point.
        absmax = act[ALIAS.get(name, name)]["observed_absmax"]
        acc_bits = max(1, math.ceil(math.log2(absmax / abs(s_from) + 1)))
        m_bits = max(2, min(31, 62 - acc_bits))
        M = abs(s_from) / s_to
        m_int, shift = to_dyadic(M, bits=m_bits)
        requant[name] = {"M": M, "M_int": m_int, "shift": shift,
                         "sign": -1 if s_from < 0 else 1,
                         "kind": kind, "acc_bits": acc_bits,
                         "prod_bits": acc_bits + m_int.bit_length(), "note": note}

    def add_align(name, s_operand, s_common, operand, note=""):
        m_int, shift = to_dyadic(s_operand / s_common)
        align[name] = {"M": s_operand / s_common, "M_int": m_int, "shift": shift,
                       "operand": operand, "common_scale": s_common, "note": note}

    def add_erf_unit(name, S_in):
        """i_erf constants for GELU: polynomial at S_in/sqrt(2), then a fixed-scale
        requant of the erf value (range [-1,1]) to 1/ERF_QMAX."""
        Sp = S_in / math.sqrt(2)
        c = poly_consts(Sp, ERF_A, ERF_B, ERF_C)
        m_int, shift = to_dyadic(abs(c["S_L"]) * ERF_QMAX)
        S_out = S_in * (1.0 / ERF_QMAX) / 2
        nonlin[name] = {"type": "gelu", "S_in": S_in, "S_out": S_out,
                        "clip_qmax": math.floor(-ERF_B / Sp),
                        "q_b": c["q_b"], "q_c": c["q_c"],
                        "erf_M_int": m_int, "erf_shift": shift,
                        "erf_sign": -1 if c["S_L"] < 0 else 1,
                        "q_1": ERF_QMAX}
        return S_out

    def ln_unit(o):
        """LayerNorm constants. The input scale cancels in the normalization, so the
        only derived value is S_out = S_gamma/K. beta is pre-converted into that
        output domain here (offline), so the runtime just adds an INT32."""
        S_out = o["gamma_scale"] / K_LAYERNORM
        beta_out = torch.round(o["beta_int32"].double() * (o["beta_scale"] / S_out))
        return {"type": "layernorm", "K": K_LAYERNORM, "S_out": S_out,
                "beta_out_int32": beta_out.to(torch.int64)}

    def add_exp_unit(name, S_in, kind, S_out):
        """i_exp constants, shared by softmax and tanh."""
        c = poly_consts(S_in, EXP_A, EXP_B, EXP_C)
        m_int, shift = to_dyadic(abs(c["S_L"]) * EXP_QMAX)
        nonlin[name] = {"type": kind, "S_in": S_in, "S_out": S_out,
                        "q_ln2": math.floor(LN2 / S_in),
                        "q_b": c["q_b"], "q_c": c["q_c"],
                        "exp_M_int": m_int, "exp_shift": shift,
                        "exp_qmax": EXP_QMAX}

    # ---- embedding: 3 tables at different scales -> align to the finest, then add
    tbl = {n: op[n]["scale"] for n in ("emb_word", "emb_pos", "emb_type")}
    s_common = min(tbl.values())
    for n, s in tbl.items():
        add_align(f"emb_sum.{n}", s, s_common, n, "embedding table -> common domain")

    # ---- embedding LayerNorm -> emb_out
    o = op["emb_ln"]
    nonlin["emb_ln"] = ln_unit(o)
    add_requant("emb_out", o["gamma_scale"] / K_LAYERNORM, S("emb_out"), "layernorm")

    for L in (0, 1):
        # ---- Q / K / V : matmul accumulator -> INT8
        for t in ("q", "k", "v"):
            w = op[f"L{L}.W_{t}"]
            add_requant(f"L{L}.{t}", S(w["in"]) * w["w_scale"], S(f"L{L}.{t}"), "matmul")

        # ---- Q.K^T -> softmax : the 1/sqrt(d) folds into the scale (no shift, no
        #      precision loss). softmax consumes the accumulator directly.
        s_scores = S(f"L{L}.q") * S(f"L{L}.k") / math.sqrt(D_HEAD)
        add_exp_unit(f"L{L}.softmax", s_scores, "softmax", S(f"L{L}.probs"))
        fused[f"L{L}.probs"] = {"scale": S(f"L{L}.probs"),
                                "note": "softmax normalization already scales to 1/255"}

        # ---- P.V : activation x activation accumulator -> INT8
        add_requant(f"L{L}.ctx", S(f"L{L}.probs") * S(f"L{L}.v"), S(f"L{L}.ctx"),
                    "attn_matmul", "operands are both activations, no weight")

        # ---- W_o accumulator + x : align x, add in the accumulator domain
        w = op[f"L{L}.W_o"]
        x_src = "emb_out" if L == 0 else f"L{L-1}.ln2_out"
        add_align(f"L{L}.res1", S(x_src), S(w["in"]) * w["w_scale"], x_src,
                  "residual: lift x to accumulator")

        # ---- LayerNorm1 -> ln1_out
        o = op[f"L{L}.ln1"]
        nonlin[f"L{L}.ln1"] = ln_unit(o)
        add_requant(f"L{L}.ln1_out", o["gamma_scale"] / K_LAYERNORM,
                    S(f"L{L}.ln1_out"), "layernorm")

        # ---- FFN1 accumulator -> GELU -> ffn_act
        w = op[f"L{L}.W_1"]
        s_gelu = add_erf_unit(f"L{L}.gelu", S(w["in"]) * w["w_scale"])
        add_requant(f"L{L}.ffn_act", s_gelu, S(f"L{L}.ffn_act"), "gelu")

        # ---- W_2 accumulator + x : align, add
        w = op[f"L{L}.W_2"]
        add_align(f"L{L}.res2", S(f"L{L}.ln1_out"), S(w["in"]) * w["w_scale"],
                  f"L{L}.ln1_out", "residual: lift x to accumulator")

        # ---- LayerNorm2 -> ln2_out
        o = op[f"L{L}.ln2"]
        nonlin[f"L{L}.ln2"] = ln_unit(o)
        add_requant(f"L{L}.ln2_out", o["gamma_scale"] / K_LAYERNORM,
                    S(f"L{L}.ln2_out"), "layernorm")

    # ---- head: W_pool accumulator -> tanh (fused) -> W_cls -> argmax
    w = op["W_pool"]
    add_exp_unit("tanh", S(w["in"]) * w["w_scale"], "tanh", S("pool_out"))
    fused["pool_out"] = {"scale": S("pool_out"),
                         "note": "i_tanh's final division already scales to 1/127"}

    torch.save({"requant": requant, "align": align, "fused": fused,
                "nonlin": nonlin, "K_layernorm": K_LAYERNORM}, OUT_PATH)

    # ------------------------------------------------------------- report
    print(f"{'point':16s} {'kind':11s} {'M':>12s} {'M_int':>12s} {'shift':>6s} "
          f"{'acc':>4s} {'prod':>5s}")
    print("-" * 74)
    worst = 0
    for n, d in requant.items():
        worst = max(worst, d["prod_bits"])
        print(f"{n:16s} {d['kind']:11s} {d['M']:12.4e} {d['M_int']:>12d} "
              f"{d['shift']:>6d} {d['acc_bits']:>4d} {d['prod_bits']:>5d}")
    print(f"{'':52s}{'max':>6s} {worst:>5d}   {'OK' if worst <= 63 else 'OVERFLOW'}")

    print(f"\n{'align point':16s} {'M':>12s} {'M_int':>12s} {'shift':>6s}   operand")
    print("-" * 74)
    for n, d in align.items():
        print(f"{n:16s} {d['M']:12.4e} {d['M_int']:>12d} {d['shift']:>6d}   {d['operand']}")

    print(f"\n{'nonlinear unit':16s} {'type':10s} {'S_in':>11s} {'S_out':>11s}   constants")
    print("-" * 90)
    for n, d in nonlin.items():
        if d["type"] == "layernorm":
            print(f"{n:16s} {d['type']:10s} {'(cancels)':>11s} {d['S_out']:11.3e}   K={d['K']}")
        elif d["type"] == "gelu":
            print(f"{n:16s} {d['type']:10s} {d['S_in']:11.3e} {d['S_out']:11.3e}   "
                  f"clip={d['clip_qmax']}, q_b={d['q_b']}, q_c={d['q_c']}, q_1={d['q_1']}")
        else:
            print(f"{n:16s} {d['type']:10s} {d['S_in']:11.3e} {d['S_out']:11.3e}   "
                  f"q_ln2={d['q_ln2']}, q_b={d['q_b']}, q_c={d['q_c']}")

    print(f"\nfused: {', '.join(fused)}   (no multiplier)")
    print(f"requant {len(requant)} + align {len(align)} + nonlin {len(nonlin)} "
          f"+ fused {len(fused)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
