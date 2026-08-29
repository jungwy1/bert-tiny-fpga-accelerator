"""
Compile each encoder LayerNorm's affine (gamma*x_hat + beta) into VFU coefficients.

The rsqrt page gives x_hat at scale s_rho (T = (Nz-S)*R,  x_hat = T*s_rho).  The affine
requant to the LN output int8 (scale S_out) folds gamma, beta, s_rho, S_out into one
per-feature multiplier + intercept and a single shift (VFU OP_LN_AFFINE):

    M_gamma[i] = round( gamma[i] * s_rho / S_out * 2^F_final )      # signed 18-bit
    C_beta[i]  = round( beta[i]  / S_out          * 2^F_final )      # signed 48-bit
    out_int8   = NARROW_S8( RNE( (T*M_gamma[i] + C_beta[i]) >> F_final ) )

F_final = the largest shift (<=63) keeping every M_gamma in 18b and every C_beta in 48b
(same rule as the NN-LUT per-page shift / req_scale mult).  Runs AFTER train_rsqrt.py
(needs its per-LN s_rho).  Depends on quant_params.pt (S_out) + state_dict (gamma/beta).

Run (from python/):  python export_ln_affine.py
Output: adds M_gamma/C_beta/F_final tensors into each LN entry of quant_params.pt (re-saved),
so the model uses them directly: qp["weight"]["L0.ln1"]["M_gamma"] etc.
"""
import json
import os
import torch

from data import load_state_dict

QP_PATH    = "quant_params.pt"
RSQRT_PATH = "NN-LUT/layernorm/lut_rsqrt.json"
OUT_DIR    = "mem"
LN_DEPTH   = 512                    # 4 LN x 128 feature (mem_ln_param bank)
W_M, W_C, F_MAX = 18, 48, 63
M_MAX = (1 << (W_M - 1)) - 1        # 131071  (signed 18-bit)
C_MAX = (1 << (W_C - 1)) - 1        # signed 48-bit

# encoder LN -> (state_dict LayerNorm prefix, output activation)
LN_MAP = {
    "L0.ln1": ("bert.encoder.layer.0.attention.output.LayerNorm", "L0.ln1_out"),
    "L0.ln2": ("bert.encoder.layer.0.output.LayerNorm",           "L0.ln2_out"),
    "L1.ln1": ("bert.encoder.layer.1.attention.output.LayerNorm", "L1.ln1_out"),
    "L1.ln2": ("bert.encoder.layer.1.output.LayerNorm",           "L1.ln2_out"),
}


def pick_F(a_max, b_max):
    """Largest F in [0,F_MAX] with round(a_max*2^F) in 18b and round(b_max*2^F) in 48b."""
    for F in range(F_MAX, -1, -1):
        if round(a_max * (1 << F)) <= M_MAX and round(b_max * (1 << F)) <= C_MAX:
            return F
    return 0


def main():
    qp = torch.load(QP_PATH)
    sd = load_state_dict()
    with open(RSQRT_PATH) as f:
        rsqrt = json.load(f)["layers"]

    lnmem = [0] * LN_DEPTH             # ln_param bank: word = {F_final[71:66], M_gamma[65:48], C_beta[47:0]}

    print(f"{'LN':8s} {'s_rho':>11s} {'S_out':>11s} {'F_final':>7s} "
          f"{'|M_gamma|max':>12s} {'|C_beta|max':>12s} {'rel_err':>9s}")
    for ln_id, (ln, (prefix, out_act)) in enumerate(LN_MAP.items()):
        gamma = sd[prefix + ".weight"].double()
        beta  = sd[prefix + ".bias"].double()
        s_rho = rsqrt[ln]["s_rho"]
        s_out = qp["act"][out_act]["scale"]

        a = gamma * (s_rho / s_out)                       # M_gamma real coeff (per-feature)
        b = beta / s_out                                  # C_beta  real coeff (per-feature)
        F = pick_F(float(a.abs().max()), float(b.abs().max()))
        M_gamma = torch.round(a * (1 << F)).to(torch.int64)
        C_beta  = torch.round(b * (1 << F)).to(torch.int64)
        assert int(M_gamma.abs().max()) <= M_MAX, f"{ln}: M_gamma overflows 18-bit"
        assert int(C_beta.abs().max())  <= C_MAX, f"{ln}: C_beta overflows 48-bit"

        # sanity: reconstructed affine scale vs true (gamma*s_rho/S_out)
        recon = M_gamma.double() / (1 << F)
        rel = float(((recon - a).abs() / a.abs().clamp_min(1e-12)).max())

        print(f"{ln:8s} {s_rho:11.4e} {s_out:11.4e} {F:>7d} "
              f"{int(M_gamma.abs().max()):>12d} {int(C_beta.abs().max()):>12d} {rel*100:8.4f}%")
        # add directly into the LN entry (tensors, ready to use like the rest of qp)
        qp["weight"][ln]["M_gamma"] = M_gamma.to(torch.int32)   # int18 fits int32
        qp["weight"][ln]["C_beta"]  = C_beta                    # int48 -> int64
        qp["weight"][ln]["F_final"] = int(F)
        qp["weight"][ln]["s_rho"]   = float(s_rho)

        # pack per-feature into the ln_param bank word (addr = ln_id*128 + feature)
        mg = M_gamma.tolist(); cb = C_beta.tolist()
        for i in range(len(mg)):
            m = mg[i] & 0x3FFFF                                 # M_gamma 18b two's complement
            c = cb[i] & 0xFFFFFFFFFFFF                          # C_beta  48b two's complement
            lnmem[ln_id * 128 + i] = ((F & 0x3F) << 66) | (m << 48) | c

    torch.save(qp, QP_PATH)
    print(f"\nadded M_gamma/C_beta/F_final/s_rho into {QP_PATH}")

    # write the ln_param bank ($readmemh, 72-bit = 18 hex / word)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "ln_param.txt")
    with open(path, "w") as f:
        for word in lnmem:
            f.write(f"{word:018x}\n")
    print(f"wrote {len([w for w in lnmem if w])} / {LN_DEPTH} words -> {path}")

    # round-trip: unpack L0.ln1 feature 0
    w0 = lnmem[0]
    c0 = w0 & ((1 << 48) - 1);  m0 = (w0 >> 48) & 0x3FFFF;  f0 = (w0 >> 66) & 0x3F
    c0 = c0 - (1 << 48) if c0 >> 47 else c0                 # sign-extend
    m0 = m0 - (1 << 18) if m0 >> 17 else m0
    e = qp["weight"]["L0.ln1"]
    ok = (m0 == int(e["M_gamma"][0])) and (c0 == int(e["C_beta"][0])) and (f0 == e["F_final"])
    print(f"round-trip (L0.ln1 f=0): F={f0} M={m0} C={c0}  {'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
