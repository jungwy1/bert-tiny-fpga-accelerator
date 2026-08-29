"""
Decompose every requant scale into a dyadic (mult:S18, shift:U6) pair for the VFU.

Each linear op's requant is a single float M = S_in*S_w / S_out (bias already lives in
the accumulator).  The VFU applies it as  RNE( (acc * mult) >> shift ),  so M is stored as
    M ~= mult / 2^shift,   mult in signed 18-bit,   shift in [0,63]
picking the largest shift that keeps mult inside 18-bit (max precision), same rule as the
NN-LUT per-segment shamt.  LN affine scales carry the extra *2^8 exactly as model_int does.

Run (from python/):  python export_req_scale.py
Output: req_scales.json  + printed table.
"""
import json
import torch

QP_PATH = "quant_params.pt"
M_BITS, SH_BITS = 18, 6
M_MAX  = (1 << (M_BITS - 1)) - 1        # 131071  (signed 18-bit)
SH_MAX = (1 << SH_BITS) - 1            # 63


def to_mult_shift(scale):
    """float M -> (mult:S18, shift:U6) with M ~= mult/2^shift, largest shift keeping mult in 18b."""
    if scale == 0:
        return 0, 0
    for sh in range(SH_MAX, -1, -1):
        m = round(scale * (1 << sh))
        if abs(m) <= M_MAX:
            return m, sh
    return (M_MAX if scale > 0 else -M_MAX), 0     # scale too large for 18b even @shift=0


def req_scales(qp, L):
    """The float requant multipliers used by model_int (bias folded in the accumulator)."""
    w, a = qp["weight"], qp["act"]
    ln1_in_src = "emb_out" if L == 0 else "L0.ln2_out"    # res1 skip = layer input
    return {
        "q":       w[f"L{L}.W_q"]["bias_scale"] / a[f"L{L}.q"]["scale"],
        "k":       w[f"L{L}.W_k"]["bias_scale"] / a[f"L{L}.k"]["scale"],
        "v":       w[f"L{L}.W_v"]["bias_scale"] / a[f"L{L}.v"]["scale"],
        "ln1_in":  w[f"L{L}.W_o"]["bias_scale"] / a[ln1_in_src]["scale"],                    # attn_out -> res1 skip
        "ln1_out": (w[f"L{L}.ln1"]["beta_scale_nnlut"] / a[f"L{L}.ln1_out"]["scale"]) * (2 ** 8),  # LN1 affine
        "ln2_in":  w[f"L{L}.W_2"]["bias_scale"] / a[f"L{L}.ln1_out"]["scale"],               # ffn_out -> res2 skip
        "ln2_out": (w[f"L{L}.ln2"]["beta_scale_nnlut"] / a[f"L{L}.ln2_out"]["scale"]) * (2 ** 8),  # LN2 affine
    }


def main():
    qp = torch.load(QP_PATH)
    report = {"widths": {"mult": M_BITS, "shift": SH_BITS}, "layers": {}}

    print(f"{'name':14s} {'scale':>12s} {'mult(S18)':>10s} {'sh':>3s} {'approx':>12s} {'rel_err':>9s}")
    for L in (0, 1):
        layer = {}
        for name, scale in req_scales(qp, L).items():
            m, sh = to_mult_shift(scale)
            approx = m / (1 << sh) if sh or m else 0.0
            rel = abs(approx - scale) / scale if scale else 0.0
            assert abs(m) <= M_MAX, f"L{L}.{name}: mult overflows 18-bit ({m})"
            assert 0 <= sh <= SH_MAX, f"L{L}.{name}: shift out of 6-bit range ({sh})"
            print(f"L{L}.{name:11s} {scale:12.4e} {m:10d} {sh:3d} {approx:12.4e} {rel*100:8.4f}%")
            layer[name] = {"scale": scale, "mult": m, "shift": sh, "rel_err": rel}
        report["layers"][f"L{L}"] = layer

    with open("req_scales.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote req_scales.json")


if __name__ == "__main__":
    main()
