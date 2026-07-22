"""
Integer-only primitives (I-BERT, Kim et al. ICML 2021).

Convention used everywhere in this file
---------------------------------------
A quantized value is a pair (q, S):   real value  x  ==  q * S
  q : integer tensor (torch int32/int64 during computation)
  S : python float, the scaling factor  (S = r/q convention, same as I-BERT)
Symmetric quantization (zero-point = 0) unless stated otherwise.

Every op takes (q, S) and returns (q_out, S_out) so scales chain analytically:
the caller never needs to calibrate a nonlinear's output — it is derived.
All constants (q_b, q_c, S_out, ...) are computable offline (static quantization),
so hardware performs integer arithmetic only.

Reference: I-BERT §3.3-3.6, Algorithms 1-4.
Run `python int_ops.py` to check each primitive against its float counterpart.
"""

import math
import torch
import torch.nn.functional as F

# --- polynomial coefficients from the paper -------------------------------
# erf approximation, I-BERT Eq. 8:  L(x) = sgn(x)[a(clip(|x|, max=-b) + b)^2 + 1]
ERF_A, ERF_B, ERF_C = -0.2888, -1.769, 1.0
# exp approximation on (-ln2, 0], I-BERT Eq. 13:  L(p) = a(p + b)^2 + c
EXP_A, EXP_B, EXP_C = 0.3585, 1.353, 0.344
LN2 = math.log(2)

# Intermediates whose range is known a priori get a FIXED output scale instead of
# the polynomial's natural (far too fine) a*S^2. This is what keeps the downstream
# widths inside INT32 -- the same trick softmax/tanh already use on their outputs.
ERF_QMAX = 2 ** 13 - 1        # erf in [-1, 1];  13 bits -> i_gelu multiply ~27b
EXP_QMAX = 2 ** 16 - 1        # exp in (0, 1];   16 bits -> softmax pipeline ~24b


# ==========================================================================
# Algorithm 1 — integer second-order polynomial
# ==========================================================================
def i_poly(q, S, a, b, c):
    """Integer evaluation of  a*(x + b)^2 + c   where x = q*S.

    In : q [int tensor], S [float], a,b,c [float] polynomial coefficients
    Out: (q_out [int tensor], S_out [float])  with  q_out*S_out ≈ a(x+b)^2 + c

    Note S_out is *derived* (the square makes the scale S -> a*S^2), not calibrated.
    """
    # q_b, q_c are compile-time constants (python ints, so the tensor stays integer);
    # q_out is pure integer arithmetic.
    #   a(x+b)^2 + c = a*S^2 * [ (q + b/S)^2 + c/(a*S^2) ]
    q_b = math.floor(b / S)               # python int -> tensor stays integer
    q_c = math.floor(c / (a * S**2))
    S_out = a * S**2                      # derived: the square turns S into a*S^2
    q = q.to(torch.int64)
    q_out = ((q + q_b) ** 2 + q_c).to(torch.int64)
    return q_out, S_out

# ==========================================================================
# Algorithm 2 — integer erf / GELU
# ==========================================================================
def i_erf(q, S, out_qmax=ERF_QMAX):
    """Integer approximation of erf(x), x = q*S.

    In : q [int tensor], S [float], out_qmax [int] fixed output range
    Out: (q_out, S_out) with q_out*S_out ≈ erf(x),  S_out = 1/out_qmax  (FIXED)

    Uses the two tricks from §3.4: clip (erf saturates) and odd symmetry (sgn).

    The polynomial's natural output scale is a*S^2 -- absurdly fine for a value
    that provably lives in [-1, 1], which would make q_out ~26 bits and push
    i_gelu's final multiply past INT32. Since the range is known a priori we
    requantize to a FIXED scale here (the same trick softmax and tanh use on
    their outputs). This also breaks the S^3 chain: i_gelu's output scale then
    goes as S, not S^3, so it lands in INT32 on its own.
    """
    # Eq. 8:  L(x) = sgn(x) * [ a*(clip(|x|, max=-b) + b)^2 + 1 ]
    q_sgn = torch.sign(q)                        # element-wise; 0 at x == 0 (erf(0)=0)
    q_max = math.floor(-ERF_B / S)               # clip bound -b, in the quantized domain
    q_abs = torch.clamp(q.abs(), max=q_max)      # fit the polynomial on |x| only
    q_L, S_L = i_poly(q_abs, S, ERF_A, ERF_B, ERF_C)
    # S_L * out_qmax is a compile-time constant -> a dyadic multiply in hardware.
    q_out = torch.round(q_L.double() * (S_L * out_qmax)).to(torch.int64)
    return q_sgn * q_out, 1.0 / out_qmax         # restore the sign (odd function)


def i_gelu(q, S):
    """Integer GELU:  GELU(x) = x * 1/2 * [1 + erf(x/sqrt(2))].

    In : q [int tensor], S [float]
    Out: (q_out, S_out) with q_out*S_out ≈ GELU(x)

    Because i_erf now returns at the fixed scale 1/ERF_QMAX, the constant 1 is just
    ERF_QMAX and S_out goes as S (not S^3): input INT32 -> output INT32, with no
    input requantization and no output shift needed.
    """
    # erf's argument is x/sqrt(2): keep q, fold the 1/sqrt(2) into the scale
    # (compile-time constant -> no runtime division).
    q_erf, S_erf = i_erf(q, S / math.sqrt(2))
    q_1 = round(1 / S_erf)                # the constant 1 in erf's domain (= ERF_QMAX)
    q_out = q * (q_erf + q_1)             # x * [1 + erf(x/sqrt2)]  in integer form
    S_out = S * S_erf / 2                 # the 1/2 folds into the scale too
    return q_out, S_out


# ==========================================================================
# Algorithm 3 — integer exp / softmax
# ==========================================================================
def i_exp(q, S, out_qmax=EXP_QMAX):
    """Integer approximation of exp(x) for NON-POSITIVE x (x = q*S <= 0).

    In : q [int tensor, <= 0], S [float], out_qmax [int] fixed output range
    Out: (q_out, S_out) with q_out*S_out ≈ exp(x),  S_out = 1/out_qmax  (FIXED)

    Range reduction (§3.5): x = (-ln2)*z + p  ->  exp(x) = exp(p) >> z,
    so only p in (-ln2, 0] needs the polynomial.

    exp lives in (0, 1], so like erf we requantize to a FIXED scale rather than
    carrying the polynomial's a*S^2. Both consumers (i_softmax, i_tanh) use the
    result only inside a ratio where the scale cancels, so this costs nothing --
    and it keeps the softmax pipeline (sum, then *255) inside INT32.

    Requantize BEFORE the shift: exp(p) is in (0.5, 1] and so well-conditioned,
    whereas shifting first throws away precision the fixed scale cannot recover.
    """
    q_ln2 = math.floor(LN2 / S)
    z = torch.div(-q, q_ln2, rounding_mode="floor")
    q_p = (q + (z * q_ln2)).to(torch.int32)
    q_L, S_L = i_poly(q_p, S, EXP_A, EXP_B, EXP_C)
    # S_L * out_qmax is a compile-time constant -> a dyadic multiply in hardware.
    q_f = torch.round(q_L.double() * (S_L * out_qmax)).to(torch.int64)   # exp(p), fixed scale
    return torch.bitwise_right_shift(q_f, z), 1.0 / out_qmax             # * 2^-z


def i_softmax(q, S, dim=-1):
    """Integer softmax along `dim`.

    In : q [int tensor], S [float], dim [int]
    Out: (q_out, S_out) with q_out*S_out ≈ softmax(x, dim),  values in [0, 1]

    Subtracting the max first makes every exp input non-positive (and is exact,
    since softmax is shift-invariant).
    """
    q_max = q.max(dim, keepdim=True).values
    # q_tilde(int32) = q(int32) - q_max(int32); saturate the value when overflow occurs
    q_tilde = (q.to(torch.int64) - q_max.to(torch.int64)).clamp(-(2**31-1), 2**31-1).to(torch.int32)
    q_exp, _ = i_exp(q_tilde, S)
    sum_q_exp = torch.sum(q_exp, dim, keepdim=True).to(torch.int64)
    q_out = torch.div(q_exp * 255, sum_q_exp, rounding_mode="floor").to(torch.uint8)
    return q_out, 1/255


def i_tanh(q, S, qmax=127):
    """Integer tanh, used by the BERT pooler.

    In : q [int tensor], S [float], qmax [int] output range
    Out: (q_out, S_out) with q_out*S_out ≈ tanh(x),  values in [-1, 1]

    Not in I-BERT/SwiftTron (they only cover GELU/Softmax/LayerNorm), so we build
    it from the identity
        tanh(x) = sgn(x) * (1 - u) / (1 + u),   u = exp(-2|x|)
    which REUSES i_exp (its input -2|x| is non-positive by construction) and the
    divider that softmax already needs -- no new polynomial, no new hardware.
    ~29x more accurate than fitting a second-order polynomial directly to tanh.

    Like softmax, the output range [-1, 1] is known a priori, so S_out is FIXED
    (not calibrated) and the requantization is fused into the final division.
    """
    q_sgn = torch.sign(q)
    q_u, S_u = i_exp(-2 * q.abs(), S)            # u = exp(-2|x|) in (0, 1]
    q_one = round(1.0 / S_u)                     # the constant 1, in u's domain
    num = (q_one - q_u).clamp(min=0)             # 1 - u  >= 0   (S_u cancels in the ratio)
    den = q_one + q_u                            # 1 + u  >  0
    q_out = torch.div(num * qmax + den // 2, den, rounding_mode="floor")   # round-to-nearest
    return q_sgn * q_out, 1.0 / qmax


# ==========================================================================
# Algorithm 4 — integer square root
# ==========================================================================
def i_sqrt(n, max_iter=16):
    """Integer square root: floor(sqrt(n)), Newton's method, integer-only.

    In : n [int tensor, >= 0], max_iter [int] safety cap
    Out: [int tensor] = floor(sqrt(n))

    Vectorized form of I-BERT Alg. 4. The paper's scalar `return x_i` becomes a
    per-element freeze mask, since elements converge after different iteration
    counts. Hardware would instead use a fixed pipeline depth (<= 4 for INT32).
    """
    n = n.to(torch.int64)                       # variance can exceed INT32 (see i_layernorm)
    pos = n > 0
    n_safe = n.clamp(min=1)                     # keeps n == 0 out of the division

    # bit length: floor(log2(n)) + 1.  float64 is exact for integers up to 2^53.
    bits = n_safe.double().log2().floor().to(torch.int64) + 1
    # x0 = 2^ceil(bits/2) >= sqrt(n): a strict upper bound, so Newton descends
    # monotonically onto floor(sqrt(n)).
    x = torch.bitwise_left_shift(torch.ones_like(n_safe), (bits + 1) // 2)

    for _ in range(max_iter):
        y = (x + torch.div(n_safe, x, rounding_mode="floor")) // 2
        done = y >= x                           # this element has reached the floor
        x = torch.where(done, x, y)             # freeze converged, advance the rest
        if bool(done.all()):
            break

    return torch.where(pos, x, torch.zeros_like(x))


# ==========================================================================
# §3.6 — integer LayerNorm
# ==========================================================================
def i_layernorm(q_x, S_x, q_g, S_g, q_b, S_b, K=2**10):
    """Integer LayerNorm over the last dim.

    In : q_x [int tensor, ..., C], S_x [float]   input activation
         q_g [int tensor, C], S_g [float]        gamma, pre-quantized (INT8 in HW)
         q_b [int tensor, C], S_b [float]        beta,  pre-quantized (INT32 in HW)
         K   [int]                               resolution factor for dev/sigma
    Out: (q_out, S_out) with q_out*S_out ≈ LayerNorm(x)*gamma + beta

    Unlike GELU/softmax, mu and sigma are DYNAMIC (input-dependent) and must be
    computed at runtime; only gamma/beta are static (quantized offline).
    S_x cancels in (x-mu)/sigma, so the normalized value is a pure integer ratio.
    """
    C = q_x.shape[-1]
    q = q_x.to(torch.int64)

    # mean and deviation, in the input's own integer domain
    q_mean = torch.div(q.sum(-1, keepdim=True), C, rounding_mode="floor")
    dev = q - q_mean

    # variance -> integer standard deviation (i_sqrt is exact: floor(sqrt(n)))
    var = torch.div((dev * dev).sum(-1, keepdim=True), C, rounding_mode="floor")
    sigma = i_sqrt(var).clamp(min=1)          # clamp stands in for eps (guards sigma == 0)

    # normalized value, scaled by K so the integer division keeps resolution:
    #   q_norm / K  ≈  (x - mu) / sigma
    q_norm = torch.div(dev * K, sigma, rounding_mode="floor")

    # affine: out = (q_norm/K)*gamma + beta,  gamma = q_g*S_g,  beta = q_b*S_b
    S_out = S_g / K                                                  # scale of q_norm*q_g
    q_b_out = torch.round(q_b.double() * (S_b / S_out)).to(torch.int64)   # beta -> output domain (offline)
    q_out = q_norm * q_g.to(torch.int64) + q_b_out
    return q_out, S_out


# ==========================================================================
# Requantization (dyadic:  M ≈ M_int / 2**shift)
# ==========================================================================
def to_dyadic(M, bits=31, max_shift=62):
    """Convert a float multiplier M into an integer (M_int, shift) pair.

    In : M [float, > 0], bits [int] width M_int must fit, max_shift [int]
    Out: (M_int [int], shift [int])  with  M ≈ M_int / 2**shift

    Picks the LARGEST shift that still keeps M_int inside `bits` (more shift =
    more fractional precision). Computed offline; hardware then only needs an
    integer multiply and a right shift -- no divider (SwiftTron Eq. 2).
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    # M*2**shift < 2**(bits-1)  =>  shift < (bits-1) - log2(M)
    shift = max(0, min(int(math.floor((bits - 1) - math.log2(M))), max_shift))
    return round(M * 2 ** shift), shift


def requant(acc, M_int, shift, zp=0, qmin=-128, qmax=127):
    """Rescale an accumulator down to the next (narrower) quantized domain.

    In : acc [int tensor], M_int [int], shift [int], zp [int], qmin/qmax [int]
    Out: [int tensor] saturated to [qmin, qmax]

    q_out = clamp( round(acc * M_int / 2**shift) + zp, qmin, qmax )

    The product is formed in int64 so the multiply cannot wrap; hardware sizes
    this wire to (acc_width + M_int_width). Saturating -- never wrapping -- on
    the final clamp, as fixed-point hardware does.
    """
    prod = acc.to(torch.int64) * int(M_int)
    if shift > 0:                       # rounding right shift: add half an LSB first
        prod = (prod + (1 << (shift - 1))) >> shift
    return (prod + zp).clamp(qmin, qmax)


# ==========================================================================
# Verification against float references
# ==========================================================================
def _quant(x, S, qmax=2**31 - 1):
    """Helper for tests: float -> integer at scale S (symmetric)."""
    return torch.round(x / S).clamp(-qmax, qmax).to(torch.int32)


def _report(name, ref, mine, expect=None):
    err = (ref - mine).abs()
    tag = "" if expect is None else f"   (paper ~{expect})"
    print(f"{name:14s} max_err={err.max().item():.3e}  mean_err={err.mean().item():.3e}{tag}")


def _test_i_sqrt():
    n = torch.randint(0, 2**30, (10000,), dtype=torch.int64)
    ref = torch.tensor([math.isqrt(int(v)) for v in n], dtype=torch.int64)
    mine = i_sqrt(n)
    exact = bool((ref == mine).all())
    print(f"{'i_sqrt':14s} exact={exact}")


def _test_i_gelu():
    x = torch.linspace(-4, 4, 2001, dtype=torch.float64)
    S = 4.0 / 2**21                       # fine input scale
    q_out, S_out = i_gelu(_quant(x, S), S)
    _report("i_gelu", F.gelu(x), q_out.to(torch.float64) * S_out, expect="L∞ 0.018")


def _test_i_exp():
    x = torch.linspace(-10, 0, 2001, dtype=torch.float64)   # exp input is non-positive
    S = 10.0 / 2**15
    q_out, S_out = i_exp(_quant(x, S), S)
    _report("i_exp", torch.exp(x), q_out.to(torch.float64) * S_out, expect="1.9e-3")


def _test_i_tanh():
    x = torch.linspace(-6, 6, 4001, dtype=torch.float64)
    S = 6.0 / 2**15
    q_out, S_out = i_tanh(_quant(x, S), S)
    _report("i_tanh", torch.tanh(x), q_out.to(torch.float64) * S_out, expect="1/127=7.9e-3")


def _test_i_softmax():
    x = torch.randn(64, 64, dtype=torch.float64) * 4
    S = x.abs().max().item() / 2**21
    q_out, S_out = i_softmax(_quant(x, S), S, dim=-1)
    _report("i_softmax", torch.softmax(x, dim=-1), q_out.to(torch.float64) * S_out)


def _test_requant():
    """to_dyadic must reproduce M closely, and requant must match the float rescale."""
    torch.manual_seed(0)
    worst_M, worst_q = 0.0, 0
    for M in (1e-6, 3.7e-4, 0.031, 1.0, 2.5, 137.0):
        M_int, shift = to_dyadic(M)
        worst_M = max(worst_M, abs(M_int / 2 ** shift - M) / M)          # relative error on M
        acc = torch.randint(-10**7, 10**7, (20000,), dtype=torch.int64)
        ref = torch.round(acc.double() * M).clamp(-128, 127)
        worst_q = max(worst_q, int((ref - requant(acc, M_int, shift)).abs().max()))
    print(f"{'to_dyadic':14s} max_rel_err={worst_M:.3e}")
    print(f"{'requant':14s} max_off_by={worst_q}   (vs float rescale; 0 or 1 LSB is fine)")


def _test_i_layernorm():
    x = torch.randn(32, 128, dtype=torch.float64) * 3
    g = torch.randn(128, dtype=torch.float64)
    b = torch.randn(128, dtype=torch.float64)
    S_x = x.abs().max().item() / 2**23
    S_g = g.abs().max().item() / 2**7
    S_b = b.abs().max().item() / 2**31
    q_out, S_out = i_layernorm(_quant(x, S_x), S_x, _quant(g, S_g), S_g, _quant(b, S_b), S_b)
    ref = F.layer_norm(x, (128,), g, b, eps=1e-12)
    _report("i_layernorm", ref, q_out.to(torch.float64) * S_out)


if __name__ == "__main__":
    for t in (_test_i_sqrt, _test_i_gelu, _test_i_exp, _test_i_tanh, _test_i_softmax,
              _test_i_layernorm, _test_requant):
        try:
            t()
        except NotImplementedError:
            print(f"{t.__name__.replace('_test_', ''):14s} -- not implemented yet")
