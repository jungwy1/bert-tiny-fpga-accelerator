"""
Fake-quant accuracy sweep on BERT-Tiny / SST-2.

Two scopes:
  Stage 1 (linear only): quantize the LINEAR (act x weight) GEMMs -- Q/K/V/O, FFN1/2,
      pooler, classifier. Attention matmuls (Q@K^T, P@V), softmax/gelu/layernorm, and
      embeddings stay FP32. This is the common conservative W8A8 scope.
  Stage 2 (+ attention): additionally quantize the attention matmul (Q@K^T, P@V)
      operands -- Q, K, V, and probs (P) -- so both are int x int. scores stays wide
      (softmax needs the range) and softmax/gelu/LN/emb remain FP32. Full-integer scope.

Mechanism:
  * model_float.linear() is monkey-patched -> every act x weight GEMM is fake-quant.
  * a tap hook fake-quants {q, k, v, probs} when _CFG["attn"] is on -> the Q@K^T / P@V.

Configs: {A8W8, A8W4, A4W4} x weight {per-tensor, per-channel}.  act = per-tensor.
Reference FP32 baseline = 0.8142 (docs/finetune.md).

Run:  python quant_sweep.py
"""
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset
import model_float as mf

BASELINE_FP32 = 0.8142
MAX_LEN = 64
ATTN_QPTS = {"q", "k", "v", "probs"}   # Q@K^T / P@V operands (scores stays wide)


# ------------------------------------------------------------ fake quantization
def fq(x, bits, ch_dim=None):
    """Symmetric fake-quant. ch_dim=None -> per-tensor; else one scale per index
    along ch_dim (reduce over the other dims). Returns dequantized FP tensor."""
    qmax = 2 ** (bits - 1) - 1
    if ch_dim is None:
        s = x.abs().max().clamp_min(1e-8) / qmax
    else:
        dims = [d for d in range(x.dim()) if d != ch_dim]
        s = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / qmax
    return (x / s).round().clamp(-qmax, qmax) * s


# current sweep config (set per run); read by the patched linear + tap
_CFG = {"a": 8, "w": 8, "pc": False, "attn": False}


def quant_linear(x, W, b):
    """Drop-in for model_float.linear: fake-quant act (per-tensor) and weight
    (per-tensor or per-channel over output dim 0) before the matmul. W is [out, in]."""
    xq = fq(x, _CFG["a"], ch_dim=None)
    Wq = fq(W, _CFG["w"], ch_dim=0 if _CFG["pc"] else None)
    return xq @ Wq.T + b


def quant_tap(name, x):
    """Stage-2 hook: fake-quant the Q@K^T / P@V operands (q/k/v/probs) at act bits.
    Everything else passes through unchanged."""
    if _CFG["attn"] and name.split(".")[-1] in ATTN_QPTS:
        return fq(x, _CFG["a"], ch_dim=None)
    return x


mf.linear = quant_linear   # patch: all act x weight GEMMs now fake-quant


# ---------------------------------------------------------------------- eval
def evaluate(model, val, tok):
    correct = 0
    with torch.no_grad():
        for ex in val:
            ids = tok(ex["sentence"], return_tensors="pt",
                      truncation=True, max_length=MAX_LEN)["input_ids"]
            pred = model(ids[0]).argmax().item()
            correct += int(pred == ex["label"])
    return correct / len(val)


def sweep_table(model, val, tok, attn):
    _CFG["attn"] = attn
    print(f"{'config':8} | {'W per-tensor':>13} | {'W per-channel':>14}")
    print("-" * 42)
    for a, w in [(8, 8), (8, 4), (4, 4)]:
        accs = {}
        for pc in (False, True):
            _CFG.update(a=a, w=w, pc=pc)
            accs[pc] = evaluate(model, val, tok)
        print(f"A{a}W{w:<5} | {accs[False]:>13.4f} | {accs[True]:>14.4f}")


if __name__ == "__main__":
    hf  = BertForSequenceClassification.from_pretrained("./bert-tiny-sst2").eval()
    sd  = hf.state_dict()
    tok = BertTokenizerFast.from_pretrained("./bert-tiny-sst2")
    val = load_dataset("stanfordnlp/sst2")["validation"]
    model = mf.GoldenBertTiny(sd, tap=quant_tap)

    print(f"FP32 baseline: {BASELINE_FP32:.4f}\n")
    print("=== Stage 1: linear GEMMs only (attention Q@K^T,P@V = FP32) ===")
    sweep_table(model, val, tok, attn=False)
    print("\n=== Stage 2: + attention Q@K^T,P@V (Q,K,V,probs quant; scores wide) ===")
    sweep_table(model, val, tok, attn=True)
