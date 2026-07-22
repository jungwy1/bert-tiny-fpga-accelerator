"""
Hardware-exact integer primitives: no float anywhere, not even in constants.

int_ops.py is the *algorithm reference* -- it derives its constants from a float
scale S at call time, and rescales the fixed-scale intermediates with a float
multiply. That is readable and easy to check against a float reference, but it is
not what silicon does.

This module is the *hardware*. Every function takes a dict of constants already
compiled by compile_requant.py and does nothing but:

    add, subtract, multiply, compare, shift, and floor-divide

so a run here is bit-identical to the RTL. The two modules should agree; the
self-test at the bottom checks exactly that.

Constant sources (all from requant_params.pt["nonlin"][name]):

    gelu       clip_qmax, q_b, q_c, erf_M_int, erf_shift, erf_sign, q_1
    softmax    q_ln2, q_b, q_c, exp_M_int, exp_shift, S_out
    tanh       same as softmax
    layernorm  K, beta_out_int32   (+ gamma_int8 from quant_params.pt)

Run (from python/):  python fully_int_ops.py
"""

import torch

from int_ops import i_sqrt          # already pure integer -- reused as is

INT32_MAX = 2 ** 31 - 1


# ------------------------------------------------------------------ core
def dyadic(q, M_int, shift):
    """q * M_int / 2**shift with round-half-up -- one integer multiply and one
    rounding right shift. The only rescale primitive the hardware has."""
    prod = q.to(torch.int64) * int(M_int)
    if shift > 0:
        prod = (prod + (1 << (shift - 1))) >> shift
    return prod


def requant(acc, M_int, shift, sign=1, qmin=-128, qmax=127):
    """Accumulator -> narrow domain. Saturates; never wraps."""
    return dyadic(acc * sign, M_int, shift).clamp(qmin, qmax)


def i_poly(q, q_b, q_c):
    """(q + q_b)^2 + q_c -- I-BERT Alg.1 with the constants already resolved.
    One add, one square, one add."""
    q = q.to(torch.int64)
    return (q + q_b) ** 2 + q_c


# ------------------------------------------------------------------ GELU
def i_erf(q, c):
    """erf(q*S/sqrt(2)) as an integer at the fixed scale 1/q_1.

    The polynomial is only valid on [q_b, 0], so the magnitude is clipped first
    and the sign reapplied afterwards (erf is odd).
    """
    q_sgn = torch.sign(q)
    q_abs = q.abs().clamp(max=c["clip_qmax"])
    q_L = i_poly(q_abs, c["q_b"], c["q_c"])
    q_out = dyadic(q_L, c["erf_M_int"], c["erf_shift"]) * c["erf_sign"]
    return q_sgn * q_out


def i_gelu(q, c):
    """GELU(x) = x/2 * (1 + erf(x/sqrt(2))).

    Consumes the matmul accumulator directly -- no input requant. Because erf
    carries a *fixed* scale, q_1 is the constant 8191 rather than a per-layer
    value, and the output stays inside INT32.
    """
    return q.to(torch.int64) * (i_erf(q, c) + c["q_1"])


# ------------------------------------------------------------------ exp
def i_exp(q, c):
    """exp(q*S) for q <= 0, at the fixed scale 1/exp_qmax.

    Splits x = -z*ln2 + p so the polynomial only ever sees p in (-ln2, 0]; the
    2^-z factor is an exact right shift. I-BERT Alg.3.
    """
    q_ln2 = c["q_ln2"]
    z = torch.div(-q.to(torch.int64), q_ln2, rounding_mode="floor")
    q_p = q.to(torch.int64) + z * q_ln2
    q_L = i_poly(q_p, c["q_b"], c["q_c"])
    q_f = dyadic(q_L, c["exp_M_int"], c["exp_shift"])
    return torch.bitwise_right_shift(q_f, z)


def i_softmax(q, c, dim=-1):
    """Softmax along `dim`, output in [0, out_qmax] at scale 1/out_qmax.

    The final division normalizes and requantizes in one step, so there is no
    separate requant multiplier after this unit.
    """
    out_qmax = round(1.0 / c["S_out"])
    q_max = q.max(dim, keepdim=True).values
    # q_tilde(int32) = q(int32) - q_max(int32); saturate when overflow occurs
    q_tilde = (q.to(torch.int64) - q_max.to(torch.int64)).clamp(-INT32_MAX, INT32_MAX)
    q_exp = i_exp(q_tilde, c)
    q_sum = q_exp.sum(dim, keepdim=True)
    return torch.div(q_exp * out_qmax, q_sum, rounding_mode="floor")


def i_tanh(q, c):
    """tanh(x) = (1 - e^-2|x|) / (1 + e^-2|x|), sign reapplied (tanh is odd).

    Like softmax, the final division lands directly on the output scale.
    """
    out_qmax = round(1.0 / c["S_out"])
    q_sgn = torch.sign(q)
    q_u = i_exp(-2 * q.abs(), c)
    q_one = c["exp_qmax"]                       # the integer standing for 1.0
    num = (q_one - q_u).clamp(min=0)
    den = q_one + q_u
    return q_sgn * torch.div(num * out_qmax + den // 2, den, rounding_mode="floor")


# ------------------------------------------------------------ LayerNorm
def i_layernorm(q_x, gamma_int8, c):
    """LayerNorm with integer mean/variance and an exact integer sqrt.

    The input scale cancels in (x - mu)/sigma, so no input scale is needed. K sets
    the resolution of the normalized value; beta arrives already converted into the
    output domain (S_gamma/K) by the compiler, so it is a plain INT32 add.
    """
    C = q_x.shape[-1]
    q = q_x.to(torch.int64)
    q_mean = torch.div(q.sum(-1, keepdim=True), C, rounding_mode="floor")
    dev = q - q_mean
    var = torch.div((dev * dev).sum(-1, keepdim=True), C, rounding_mode="floor")
    sigma = i_sqrt(var).clamp(min=1)
    q_norm = torch.div(dev * c["K"], sigma, rounding_mode="floor")
    return q_norm * gamma_int8.to(torch.int64) + c["beta_out_int32"]


# ------------------------------------------------------------------ test
def _main():
    """Cross-check against int_ops on the real compiled constants.

    int_ops rescales with a float multiply, this module with a dyadic one, so a
    difference of a few LSBs is expected and fine; a large one means the compiled
    constants and the reference formula have drifted apart.
    """
    import int_ops as ref

    r = torch.load("requant_params.pt", weights_only=True)
    q_p = torch.load("quant_params.pt", weights_only=True)
    nl = r["nonlin"]
    g = torch.Generator().manual_seed(0)

    def report(name, a, b, scale):
        d = (a - b).abs().max().item()
        print(f"  {name:22s} max |diff| = {d:>10d} LSB   ({d * scale:.3e} in real units)")

    print("fully_int_ops  vs  int_ops   (same constants, dyadic vs float rescale)")

    c = nl["L0.gelu"]
    x = torch.randint(-(2 ** 20), 2 ** 20, (256,), generator=g)
    a = i_gelu(x, c)
    b, _ = ref.i_gelu(x, c["S_in"])
    report("i_gelu", a, b, c["S_out"])

    c = nl["L0.softmax"]
    x = torch.randint(-(2 ** 14), 2 ** 14, (16, 16), generator=g)
    a = i_softmax(x, c)
    b, _ = ref.i_softmax(x, c["S_in"])
    report("i_softmax", a, b.to(torch.int64), c["S_out"])

    c = nl["tanh"]
    x = torch.randint(-(2 ** 16), 2 ** 16, (256,), generator=g)
    a = i_tanh(x, c)
    b, _ = ref.i_tanh(x, c["S_in"])
    report("i_tanh", a, b, c["S_out"])

    c, o = nl["L0.ln1"], q_p["op"]["L0.ln1"]
    x = torch.randint(-(2 ** 20), 2 ** 20, (8, 128), generator=g)
    a = i_layernorm(x, o["gamma_int8"], c)
    b, _ = ref.i_layernorm(x, 1.0, o["gamma_int8"], o["gamma_scale"],
                           o["beta_int32"], o["beta_scale"], K=c["K"])
    report("i_layernorm", a, b, c["S_out"])


if __name__ == "__main__":
    _main()
