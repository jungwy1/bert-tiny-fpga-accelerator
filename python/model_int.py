import torch, json
import torch.nn as nn

from transformers import BertForSequenceClassification, BertTokenizerFast

QP_PATH   = "quant_params.pt"

def requant(x: torch.Tensor, req_scale: float) -> torch.Tensor:
    return torch.round(x.to(torch.float64) * req_scale).clamp(-127,127).to(torch.int8)

def dequant(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x.to(torch.float64) * scale).to(torch.float32)

def quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.round(x / scale).clamp(-127, 127).to(torch.int8)

def float_layer_norm(x, weight, bias, eps=1e-12):
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)   # unbiased=False -> divide by N (BERT convention)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return x_hat * weight + bias

class Embeddings(nn.Module):
    def __init__(self, sd, eps=1e-12, tap=None):
        super().__init__()
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
        x = self.word[input_ids] + self.pos[pos_ids] + self.type[token_type_ids]
        return float_layer_norm(x, self.ln_w, self.ln_b, self.eps)

class EncoderLayer(nn.Module):
    def __init__(self, sd, qp, L, eps=1e-12, tap=None):
        super().__init__()
        self.L = L
        p = f"bert.encoder.layer.{L}."
        # attention
        self.Wq = qp["weight"][f"L{L}.W_q"]["w_int4"]; self.bq = qp["weight"][f"L{L}.W_q"]["bias_int32"]
        self.Wk = qp["weight"][f"L{L}.W_k"]["w_int4"]; self.bk = qp["weight"][f"L{L}.W_k"]["bias_int32"]
        self.Wv = qp["weight"][f"L{L}.W_v"]["w_int4"]; self.bv = qp["weight"][f"L{L}.W_v"]["bias_int32"]
        self.Wo = qp["weight"][f"L{L}.W_o"]["w_int4"]; self.bo = qp["weight"][f"L{L}.W_o"]["bias_int32"]
        self.ln1_w = sd[p+"attention.output.LayerNorm.weight"]; self.ln1_b = sd[p+"attention.output.LayerNorm.bias"]
        # FFN
        self.W1 = qp["weight"][f"L{L}.W_1"]["w_int4"]; self.b1 = qp["weight"][f"L{L}.W_1"]["bias_int32"]
        self.W2 = qp["weight"][f"L{L}.W_2"]["w_int4"]; self.b2 = qp["weight"][f"L{L}.W_2"]["bias_int32"]
        self.ln2_w = sd[p+"output.LayerNorm.weight"]; self.ln2_b = sd[p+"output.LayerNorm.bias"]
        self.eps = eps
        # requant scale
        self.req_scale_q = qp["weight"][f"L{L}.W_q"]["bias_scale"] / qp["act"][f"L{L}.q"]["scale"]
        self.req_scale_k = qp["weight"][f"L{L}.W_k"]["bias_scale"] / qp["act"][f"L{L}.k"]["scale"]
        self.req_scale_v = qp["weight"][f"L{L}.W_v"]["bias_scale"] / qp["act"][f"L{L}.v"]["scale"]

    def self_attention(self, x):
        S = x.shape[0]          # [S, 128]
        H, d = 2, 64            # heads, head_dim
        L = self.L
        # Q/K/V projection  [S,128]
        Q = x.to(torch.int32) @ self.Wq.to(torch.int32) + self.bq
        K = x.to(torch.int32) @ self.Wk.to(torch.int32) + self.bk
        V = x.to(torch.int32) @ self.Wc.to(torch.int32) + self.bv
        Q = requant(Q, self.req_scale_q)
        K = requant(K, self.req_scale_k)
        V = requant(V, self.req_scale_v)

        # per head [H, S, d]
        Q = Q.view(S, H, d).transpose(0,1)
        K = K.view(S, H, d).transpose(0,1)
        V = V.view(S, H, d).transpose(0,1)
        # attention scores = QK^T * scaling  [H, S, S]


        scaling = 1.0 / (d ** 0.5)
        scores = t(f"L{L}.scores", (Q @ K.transpose(-1,-2)) * scaling)   # K^T: [H, d, S]
        # softmax
        P = t(f"L{L}.probs", torch.softmax(scores, dim=-1))
        # context = PV [H, S, d], then concat  ([H,S,d] -> [S,H,d] -> [S,128])
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

def main():
    qp = torch.load(QP_PATH)
    hf  = BertForSequenceClassification.from_pretrained("./bert-tiny-sst2").eval()
    sd  = hf.state_dict()
    tok = BertTokenizerFast.from_pretrained("./bert-tiny-sst2")

    ids = tok("Feel so high", return_tensors="pt")["input_ids"]   # [1, S]

