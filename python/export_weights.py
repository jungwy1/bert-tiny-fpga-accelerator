"""
Export INT4 matmul weights + INT32 bias to $readmemh .txt for the on-chip banks.

Weight (memory.md §2/§6/§9, pe.sv pack2) -> 3x URAM (4096 x 128), one .txt per bank:
  stored W^T k-major, addr = baseW + k*NT + tc.  word=128b=16 byte; byte c={w_b[7:4],w_a[3:0]}
  (INT4 [-7,7]).  w_a->feature tc*32+2c (acc0),  w_b->feature tc*32+2c+1 (acc1).
  gap-20 pack is a runtime DSP pre-adder job; here dense nibbles only.

Bias (memory.md §4, addr = baseB + tc*16 + c) -> 1x BRAM (1536 x 64), bias.txt:
  raw INT32 pairs.  word=64b={bias_b[63:32],bias_a[31:0]}.  bias_a->feature 2c, bias_b->2c+1.
  per layer Wq/Wk/Wv/Wo/W1/W2 at {0,64,128,192,256,512}; layer offset += 576.
  (scores/context have no bias; pool/cls are host-side.)

Run (from python/):  python export_weights.py
"""

import os
import torch

QP_PATH    = "quant_params.pt"
OUT_DIR    = "mem"
DEPTH      = 4096              # words per URAM weight bank
COLS       = 16               # columns per word (16 byte weight / 16 pair bias)
BIAS_DEPTH = 1536             # words in the bias BRAM bank (512 x 3-stack)
BIAS_OFF   = {"W_q": 0, "W_k": 64, "W_v": 128, "W_o": 192, "W_1": 256, "W_2": 512}
BIAS_LAYER = 576              # per-layer bias word span

def pack_matrix(mem, q_w, base):
    """Place one INT4 weight matrix q_w[out=N, in=K] into mem at `base`.

    Returns the word count used (K*NT). q_w values are in [-7,7]."""
    N, K = q_w.shape
    NT = N // 32
    w = q_w.tolist()                                   # [N][K], plain ints
    for k in range(K):
        for tc in range(NT):
            word = 0
            for c in range(COLS):
                f_a = tc * 32 + 2 * c                  # low  nibble feature
                f_b = f_a + 1                          # high nibble feature
                wa = w[f_a][k] & 0xF                   # 4-bit two's complement
                wb = w[f_b][k] & 0xF
                word |= ((wb << 4) | wa) << (c * 8)    # byte c = {w_b, w_a}
            mem[base + k * NT + tc] = word
    return K * NT

def pack_bias(mem, bias, base):
    """Place per-channel INT32 bias[N] into the bias bank at `base`.

    Returns the word count used (NT*16).  word c = {bias_b, bias_a} (column pair)."""
    N = bias.shape[0]
    assert N % 32 == 0, f"N={N} must be a multiple of 32 (pack2 col-tile)"
    NT = N // 32
    b = bias.tolist()
    for tc in range(NT):
        for c in range(COLS):
            f_a = tc * 32 + 2 * c                      # low  INT32 feature
            f_b = f_a + 1                              # high INT32 feature
            ba = b[f_a] & 0xFFFFFFFF                   # 32-bit two's complement
            bb = b[f_b] & 0xFFFFFFFF
            mem[base + tc * 16 + c] = (bb << 32) | ba  # {bias_b, bias_a}
    return NT * COLS

def write_bank(path, mem, hexw):
    """Write a bank as $readmemh hex (one `hexw`-char line per word)."""
    with open(path, "w") as f:
        for word in mem:
            f.write(f"{word:0{hexw}x}\n")

def main():
    qp = torch.load(QP_PATH, weights_only=False)
    W = qp["weight"]
    os.makedirs(OUT_DIR, exist_ok=True)

    # bank -> list of (matrix name, base word)
    banks = {
        "weight_qkvo": [],
        "weight_ffn_l0": [],
        "weight_ffn_l1": [],
    }
    for L in (0, 1):
        off = L * 2048
        banks["weight_qkvo"] += [(f"L{L}.W_q", off + 0),   (f"L{L}.W_k", off + 512),
                                 (f"L{L}.W_v", off + 1024), (f"L{L}.W_o", off + 1536)]
        banks[f"weight_ffn_l{L}"] += [(f"L{L}.W_1", 0), (f"L{L}.W_2", 2048)]

    for bank, entries in banks.items():
        mem = [0] * DEPTH
        used = 0
        rows = []
        for name, base in entries:
            q_w = W[name]["w_int4"].to(torch.int64)    # [N,K], values [-7,7]
            n = pack_matrix(mem, q_w, base)
            used += n
            rows.append((name, base, q_w.shape[0], q_w.shape[1], n))
        path = os.path.join(OUT_DIR, bank + ".txt")
        write_bank(path, mem, 32)

        print(f"{path}   ({used}/{DEPTH} words used)")
        for name, base, N, K, n in rows:
            print(f"   {name:8s}  base={base:5d}  N={N:4d} K={K:4d}  NT={N//32:2d}  words={n}")

    # ---- bias bank (single BRAM, both layers) ----
    bmem = [0] * BIAS_DEPTH
    used = 0
    rows = []
    for L in (0, 1):
        for m, off in BIAS_OFF.items():
            name = f"L{L}.{m}"
            base = L * BIAS_LAYER + off
            b = W[name]["bias_int32"].to(torch.int64)  # [N]
            n = pack_bias(bmem, b, base)
            used += n
            rows.append((name, base, b.shape[0], n))
    path = os.path.join(OUT_DIR, "bias.txt")
    write_bank(path, bmem, 16)
    print(f"\n{path}   ({used}/{BIAS_DEPTH} words used)")
    for name, base, N, n in rows:
        print(f"   {name:8s}  base={base:5d}  N={N:4d}  words={n}")

    # round-trip sanity: unpack one byte of the first QKVO word, compare to source
    q0 = W["L0.W_q"]["w_int4"].to(torch.int64)
    mem0 = [0] * DEPTH
    pack_matrix(mem0, q0, 0)
    word0 = mem0[0]                                    # k=0, tc=0
    b0 = word0 & 0xFF                                  # column 0
    wa = b0 & 0xF; wb = (b0 >> 4) & 0xF
    wa = wa - 16 if wa >= 8 else wa                    # sign-extend INT4
    wb = wb - 16 if wb >= 8 else wb
    ok = (wa == int(q0[0, 0])) and (wb == int(q0[1, 0]))
    print(f"\nround-trip (L0.W_q k=0,col=0): w_a={wa}={int(q0[0,0])} w_b={wb}={int(q0[1,0])}  {'OK' if ok else 'MISMATCH'}")

    bq = W["L0.W_q"]["bias_int32"].to(torch.int64)     # bias round-trip (col 0)
    ba = bmem[0] & 0xFFFFFFFF; bb = (bmem[0] >> 32) & 0xFFFFFFFF
    ba = ba - (1 << 32) if ba >> 31 else ba
    bb = bb - (1 << 32) if bb >> 31 else bb
    okb = (ba == int(bq[0])) and (bb == int(bq[1]))
    print(f"round-trip (L0.W_q bias col=0): a={ba}={int(bq[0])} b={bb}={int(bq[1])}  {'OK' if okb else 'MISMATCH'}")


if __name__ == "__main__":
    main()
