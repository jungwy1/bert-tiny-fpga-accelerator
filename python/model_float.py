import torch
import torch.nn as nn

from transformers import BertForSequenceClassification, BertTokenizerFast

def _no_tap(name, x):
    """Default tap: identity — the model behaves exactly as before.
    Pass a real `tap(name, tensor)` (see export_params.py) to observe the
    activation at every quantization point without changing the math."""
    return x


def layer_norm(x, weight, bias, eps=1e-12):
    """Per-token LayerNorm over the last dim (hidden=128).
    Kept as explicit math (not nn.LayerNorm) so it can be hardened to
    fixed-point in the integer golden model later."""
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)   # unbiased=False = /N (BERT 방식)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return x_hat * weight + bias

class Embeddings(nn.Module):
    def __init__(self, sd, eps=1e-12, tap=None):
        super().__init__()
        self.tap = tap or _no_tap
        # fine-tuned weight load
        self.word = sd["bert.embeddings.word_embeddings.weight"]        # [30522,128]
        self.pos  = sd["bert.embeddings.position_embeddings.weight"]    # [512,128]
        self.type = sd["bert.embeddings.token_type_embeddings.weight"]  # [2,128]
        self.ln_w = sd["bert.embeddings.LayerNorm.weight"]
        self.ln_b = sd["bert.embeddings.LayerNorm.bias"]
        self.eps  = eps

    def forward(self, input_ids, token_type_ids=None):
        S = input_ids.shape[-1]
        pos_ids = torch.arange(S)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # three INT8 tables with different scales -> the sum needs scale alignment,
        # and it is the LayerNorm input, so it is its own quantization point.
        x = self.tap("emb_sum",
                     self.word[input_ids] + self.pos[pos_ids] + self.type[token_type_ids])
        x = layer_norm(x, self.ln_w, self.ln_b, self.eps)
        return self.tap("emb_out", x)

def linear(x, W, b):
    # W: [out, in]
    return x @ W.T + b

class EncoderLayer(nn.Module):
    def __init__(self, sd, L, eps=1e-12, tap=None):
        super().__init__()
        self.L = L
        self.tap = tap or _no_tap
        p = f"bert.encoder.layer.{L}."
        # attention
        self.Wq = sd[p+"attention.self.query.weight"]; self.bq = sd[p+"attention.self.query.bias"]
        self.Wk = sd[p+"attention.self.key.weight"];   self.bk = sd[p+"attention.self.key.bias"]
        self.Wv = sd[p+"attention.self.value.weight"]; self.bv = sd[p+"attention.self.value.bias"]
        self.Wo = sd[p+"attention.output.dense.weight"]; self.bo = sd[p+"attention.output.dense.bias"]
        self.ln1_w = sd[p+"attention.output.LayerNorm.weight"]; self.ln1_b = sd[p+"attention.output.LayerNorm.bias"]
        # FFN
        self.W1 = sd[p+"intermediate.dense.weight"]; self.b1 = sd[p+"intermediate.dense.bias"]
        self.W2 = sd[p+"output.dense.weight"];       self.b2 = sd[p+"output.dense.bias"]
        self.ln2_w = sd[p+"output.LayerNorm.weight"]; self.ln2_b = sd[p+"output.LayerNorm.bias"]
        self.eps = eps

    def self_attention(self, x):
        S = x.shape[0]          # [S, 128]
        H, d = 2, 64            # heads, head_dim
        t, L = self.tap, self.L
        # Q/K/V projection  [S,128]
        Q = t(f"L{L}.q", linear(x, self.Wq, self.bq))
        K = t(f"L{L}.k", linear(x, self.Wk, self.bk))
        V = t(f"L{L}.v", linear(x, self.Wv, self.bv))
        # head [H, S, d]
        Q = Q.view(S, H, d).transpose(0,1)
        K = K.view(S, H, d).transpose(0,1)
        V = V.view(S, H, d).transpose(0,1)
        # attetion socres = QK_T * scaling  [H, S, S]
        scaling = 1.0 / (d ** 0.5)
        scores = t(f"L{L}.scores", (Q @ K.transpose(-1,-2)) * scaling) # K.t = [S, d, H]
        # softmax
        P = t(f"L{L}.probs", torch.softmax(scores, dim=-1))
        # context = PV [H, S, d] & concat (-> [S,H,d] -> [S,128])
        ctx = t(f"L{L}.ctx", (P @ V).transpose(0,1).reshape(S, H * d))
        # output projection
        return t(f"L{L}.attn_out", linear(ctx, self.Wo, self.bo))

    def forward(self, x):
        t, L = self.tap, self.L
        a = self.self_attention(x)
        r = t(f"L{L}.res1", x + a)                                  # residual sum
        x = t(f"L{L}.ln1_out", layer_norm(r, self.ln1_w, self.ln1_b, self.eps))
        h = t(f"L{L}.ffn_mid", linear(x, self.W1, self.b1))         # FFN 1 (pre-GELU)
        h = t(f"L{L}.ffn_act", torch.nn.functional.gelu(h))         # GELU
        f = t(f"L{L}.ffn_out", linear(h, self.W2, self.b2))         # FFN 2
        r = t(f"L{L}.res2", x + f)                                  # residual sum
        return t(f"L{L}.ln2_out", layer_norm(r, self.ln2_w, self.ln2_b, self.eps))

class Pooler(nn.Module):
    def __init__(self, sd, tap=None):
        super().__init__()
        self.tap = tap or _no_tap
        self.Wp = sd["bert.pooler.dense.weight"]; self.bp = sd["bert.pooler.dense.bias"]
    def forward(self, x):                     # x: [S,128]
        cls = self.tap("pool_in", x[0])                                  # [CLS] = 0번 토큰 [128]
        h = self.tap("pool_mid", linear(cls, self.Wp, self.bp))          # pre-tanh (cf. ffn_mid)
        return self.tap("pool_out", torch.tanh(h))                       # [128]

class GoldenBertTiny(nn.Module):
    def __init__(self, sd, tap=None):
        super().__init__()
        self.tap = tap or _no_tap
        self.emb = Embeddings(sd, tap=self.tap)
        self.layers = [EncoderLayer(sd, 0, tap=self.tap), EncoderLayer(sd, 1, tap=self.tap)]
        self.pooler = Pooler(sd, tap=self.tap)
        self.Wc = sd["classifier.weight"]; self.bc = sd["classifier.bias"]
    def forward(self, input_ids):
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        pooled = self.pooler(x)
        return self.tap("logits", linear(pooled, self.Wc, self.bc))   # logits [2]
 

if __name__ == "__main__":
    hf  = BertForSequenceClassification.from_pretrained("./bert-tiny-sst2").eval()
    sd  = hf.state_dict()
    tok = BertTokenizerFast.from_pretrained("./bert-tiny-sst2")

    ids = tok("Feel so high", return_tensors="pt")["input_ids"]   # [1, S]

    print("===========================Embedding==========================")
    emb = Embeddings(sd)
    with torch.no_grad():
        my_emb   = emb(ids[0])                    # 우리: [S,128]
        hf_emb = hf.bert.embeddings(ids)[0]     # HF:   [S,128]
    print(f"my_emb: {my_emb.shape}")
    print(f"hf_emb: {hf_emb.shape}")
    print("emb diff:", (my_emb - hf_emb).abs().max().item())   # 1e-5 이하면 OK

    print("===========================Encoder==========================")
    layer0 = EncoderLayer(sd, 0)
    with torch.no_grad():
        x = my_emb                                  # embedding 출력
        my_enc = layer0(x)
        hf_enc = hf.bert.encoder.layer[0](x.unsqueeze(0))[0]
    print(f"my_enc: {my_enc.shape}")
    print(f"hf_enc: {hf_enc.shape}")
    print("layer0 diff:", (my_enc - hf_enc).abs().max().item())   # 1e-5 이하면 OK

    print("===========================Logits==========================")
    model = GoldenBertTiny(sd)
    with torch.no_grad():
        my_logits = model(ids[0])                      # [2]
        hf_logits = hf(ids).logits[0]             # [2]
    print(f"my_logits: {my_logits}\n{my_logits.shape}")
    print(f"hf_logits: {hf_logits.shape}")
    print("logits diff:", (my_logits - hf_logits).abs().max().item())   # 1e-4 이하면 OK

    # print("===========================Validation==========================")
    # from datasets import load_dataset

    # val = load_dataset("stanfordnlp/sst2")["validation"]
    # correct = 0
    # with torch.no_grad():
    #     for ex in val:
    #         ids = tok(ex["sentence"], return_tensors="pt", truncation=True, max_length=64)["input_ids"]
    #         pred = model(ids[0]).argmax().item()
    #         correct += (pred == ex["label"])
    # print("golden acc:", correct / len(val))      # 0.8142 나오면 Stage A 완료!


    
