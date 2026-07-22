"""
Integer-only BERT-Tiny: the golden model the FPGA must reproduce.

Assembles fully_int_ops primitives using the constants compiled into
quant_params.pt (weights, bias, gamma/beta) and requant_params.pt (dyadic
multipliers, nonlinear constants). See docs/int_datapath.md for the step-by-step
spec.

No float arithmetic anywhere -- not on the data path and not in the constants.
Activations move as INT8, accumulate in INT32, and every rescale is an integer
multiply plus a rounding right shift. A run here is bit-identical to the RTL.

Run (from python/):  python int_model.py
"""

import torch

from fully_int_ops import i_gelu, i_softmax, i_tanh, i_layernorm, requant, dyadic
from data import load_tokenizer, load_sst2, accuracy, MAX_LEN

QUANT = "quant_params.pt"
REQUANT = "requant_params.pt"
H, D_HEAD = 2, 64          # heads, head dim


class IntBertTiny:
    def __init__(self, quant_path=QUANT, requant_path=REQUANT):
        q = torch.load(quant_path, weights_only=True)
        r = torch.load(requant_path, weights_only=True)
        self.op = q["op"]
        self.rq, self.al, self.nl = r["requant"], r["align"], r["nonlin"]

    # ------------------------------------------------------------- helpers
    def _matmul(self, x_q, name):
        """INT8 x INT8 -> INT32 accumulator, with the INT32 bias already carrying
        the zero-point correction (symmetric scheme, so there is none here)."""
        o = self.op[name]
        return x_q.to(torch.int64) @ o["w_int8"].to(torch.int64).T \
            + o["bias_int32"].to(torch.int64)

    def _requant(self, acc, name, qmax=127):
        """Accumulator -> next INT8 domain (saturating, never wrapping)."""
        d = self.rq[name]
        return requant(acc, d["M_int"], d["shift"], d["sign"], -qmax, qmax)

    def _align(self, x, name):
        """Lift an addend into another scale domain before adding. No clamp: this
        always widens (M > 1), so there is nothing to saturate."""
        d = self.al[name]
        return dyadic(x, d["M_int"], d["shift"])

    def _layernorm(self, q_in, name, out_act):
        ln = i_layernorm(q_in, self.op[name]["gamma_int8"], self.nl[name])
        return self._requant(ln, out_act)

    # -------------------------------------------------------------- blocks
    def embedding(self, input_ids):
        """Three INT8 tables at different scales -> align -> sum (INT32) -> LayerNorm."""
        S = input_ids.shape[-1]
        w = self.op["emb_word"]["table_int8"][input_ids]
        p = self.op["emb_pos"]["table_int8"][:S]
        t = self.op["emb_type"]["table_int8"][0]        # single sentence: row 0 only
        q = (self._align(w, "emb_sum.emb_word")
             + self._align(p, "emb_sum.emb_pos")
             + self._align(t, "emb_sum.emb_type"))
        return self._layernorm(q, "emb_ln", "emb_out")

    def encoder_layer(self, x, L):
        S = x.shape[0]
        p = f"L{L}."

        # --- Q/K/V projection -> INT8
        q = self._requant(self._matmul(x, p + "W_q"), p + "q")
        k = self._requant(self._matmul(x, p + "W_k"), p + "k")
        v = self._requant(self._matmul(x, p + "W_v"), p + "v")
        qh = q.view(S, H, D_HEAD).transpose(0, 1)
        kh = k.view(S, H, D_HEAD).transpose(0, 1)
        vh = v.view(S, H, D_HEAD).transpose(0, 1)

        # --- scores -> softmax.  1/sqrt(d) is folded into S_in, so no shift here.
        scores = qh.to(torch.int64) @ kh.to(torch.int64).transpose(-1, -2)
        probs = i_softmax(scores, self.nl[p + "softmax"], dim=-1)

        # --- P.V -> INT8   (both operands are activations, no weight)
        ctx = (probs.to(torch.int64) @ vh.to(torch.int64)).transpose(0, 1) \
            .reshape(S, H * D_HEAD)
        ctx = self._requant(ctx, p + "ctx")

        # --- attention output + residual (added in the accumulator domain)
        res1 = self._matmul(ctx, p + "W_o") + self._align(x, p + "res1")
        x1 = self._layernorm(res1, p + "ln1", p + "ln1_out")

        # --- FFN1 -> GELU (consumes the accumulator directly) -> INT8
        g = i_gelu(self._matmul(x1, p + "W_1"), self.nl[p + "gelu"])
        h = self._requant(g, p + "ffn_act")

        # --- FFN2 + residual
        res2 = self._matmul(h, p + "W_2") + self._align(x1, p + "res2")
        return self._layernorm(res2, p + "ln2", p + "ln2_out")

    # ------------------------------------------------------------- forward
    def __call__(self, input_ids):
        x = self.embedding(input_ids)
        for L in (0, 1):
            x = self.encoder_layer(x, L)
        cls = x[0]                                    # row 0 of ln2_out: same scale
        pooled = i_tanh(self._matmul(cls, "W_pool"), self.nl["tanh"])
        return self._matmul(pooled, "W_cls")          # INT32 logits -> argmax


# ----------------------------------------------------------------- main
def main():
    tok = load_tokenizer()
    _, val = load_sst2()
    model = IntBertTiny()

    ids = tok("the world best", return_tensors="pt", truncation=True,
              max_length=MAX_LEN)["input_ids"][0]
    print(f"logits (INT32)   : {model(ids).tolist()}")

    acc = accuracy(model, tok, val)
    print(f"INT8 accuracy    : {acc:.4f}")
    print(f"  FP32 reference : 0.8142")
    print(f"  fake-quant     : 0.8177")


if __name__ == "__main__":
    main()
