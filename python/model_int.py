import torch, json
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForSequenceClassification, BertTokenizerFast

QP_PATH   = "quant_params.pt"
GELU_LUT  = "NN-LUT/gelu/lut_gelu.json"
EXP_LUT   = "NN-LUT/softmax/lut_exp.json"
RECIP_LUT = "NN-LUT/softmax/lut_recip.json"
RSQRT_LUT = "NN-LUT/layernorm/lut_rsqrt.json"
FRAC = 23

_LUT_PATH = {"gelu": GELU_LUT, "exp": EXP_LUT, "recip": RECIP_LUT, "rsqrt": RSQRT_LUT}

def load_pages() -> dict:
    pages = {}
    for func, path in _LUT_PATH.items():
        with open(path) as f:
            j = json.load(f)
        pages[func] = {}
        for k, e in j["layers"].items():
            page = {"boundary": torch.tensor(e["d"], dtype=torch.int64),
                    "M":        torch.tensor(e["s"], dtype=torch.int64),
                    "C":        torch.tensor(e["t"], dtype=torch.int64)}
            page["shift"] = e["shift"] if "shift" in e else torch.tensor(e["sh"], dtype=torch.int64)
            pages[func][k] = page
    return pages

_req = None
def req_ms(L, name):
    global _req
    if _req is None:
        with open("req_scales.json") as f:
            _req = json.load(f)["layers"]
    e = _req[f"L{L}"][name]
    return e["mult"], e["shift"]

def requant(x: torch.Tensor, req_scale: float) -> torch.Tensor:
    return torch.round(x.to(torch.float64) * req_scale).clamp(-127,127).to(torch.int8)

def dequant(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x.to(torch.float64) * scale).to(torch.float32)

def quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.round(x / scale).clamp(-127, 127).to(torch.int8)

def check_overflow(x, bit: int, name: str) -> bool:
    mx = int(x.abs().max()) if torch.is_tensor(x) else abs(x)   # tensor or python scalar
    if mx > (2 ** (bit - 1) - 1):
        print(f"{name} is over the range")
        return True
    else:
        return False

def float_layer_norm(x, weight, bias, eps=1e-12):
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)   # unbiased=False -> divide by N (BERT convention)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return x_hat * weight + bias

class VFU:
    A_W, B_W, C_W, P_W = 27, 18, 48, 48

    def _dsp(self, A, B, C=0):       
        check_overflow(A, self.A_W, "A"); check_overflow(B, self.B_W, "B")
        P = A * B + C
        check_overflow(P, self.P_W, "P")
        return P

    @staticmethod
    def _rne(P, sh):                          
        q = P >> sh # quotient           
        rem = P - (q << sh) # remainder    
        half = 1 << (sh - 1) 
        inc = (rem > half) | ((rem == half) & (q & 1))  
        return q + inc
    
    @staticmethod
    def _narrow_s8(x): return x.clamp(-127, 127).to(torch.int8)

    @staticmethod
    def _clamp_exp(x): return x.clamp(0,127).to(torch.int8)

    def _seg(self, page, x):                          # S0: segment index
        return torch.searchsorted(page["boundary"], x, right=True)

    # ------------ method
    def requant(self, x, M, F):                  
        return self._narrow_s8(self._rne(self._dsp(x, M, 0), F))

    def gelu(self, x, page):
        i = self._seg(page, x)
        return self._narrow_s8(self._rne(self._dsp(x, page["M"][i], page["C"][i]), page["shift"]))

    def pre_softmax(self, scores, page):
        rowmax = scores.amax(dim=-1, keepdim=True)
        x = scores - rowmax
        i = self._seg(page, x)
        out = self._clamp_exp(self._rne(self._dsp(x, page["M"][i], page["C"][i]), page["shift"]))
        rowsum = out.to(torch.int32).sum(dim=-1, keepdim=True)
        return out, rowsum

    def post_softmax(self, x, rowsum, page):
        i = self._seg(page, rowsum)
        recip = self._rne(self._dsp(rowsum, page["M"][i], page["C"][i]), page["shift"])  # M_fixed=1/rowsum
        return self._narrow_s8(self._rne(self._dsp(x, recip, 0), FRAC))

    def layer_norm(self, x, M, F, res, gamma, beta, F_final, page):
        main = self._narrow_s8(self._rne(self._dsp(x, M, 0), F))
        z = main.to(torch.int64) + res.to(torch.int64)  # residual sum @ skip scale
        # 2) moments per row (last dim):  S = sum z (S16),  Q = sum z^2 (U23)
        S = z.sum(dim=-1, keepdim=True)
        Q = (z * z).sum(dim=-1, keepdim=True)
        # 3) D = 128Q - S^2  (op_ln_d: A=S, B=-S, C=Q<<7),  D27 = RNE(D/16)
        D27 = self._rne(self._dsp(S, -S, Q << 7), 4)             # D_PRE_SHIFT=4
        # 4) rho = rsqrt page (U8 [0,255]);  D=0 -> D27=0 -> seg0 -> rho, but num=0 so it cancels
        i = self._seg(page, D27)
        rho = self._rne(self._dsp(D27, page["M"][i], page["C"][i]), page["shift"]).clamp(0, 255)
        # 5) num = 128z - S (S17),  T = num*rho  (= x_hat at scale s_rho, no intermediate shift)
        num = (z << 7) - S
        T = self._dsp(num, rho, 0)                               # T:S25
        # 6) affine:  out = NARROW_S8(RNE((T*M_gamma + C_beta) >> F_final))
        return self._narrow_s8(self._rne(self._dsp(T, gamma, beta), F_final))


class Embeddings(nn.Module):
    def __init__(self, sd, eps=1e-12):
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
    def __init__(self, sd, qp, L, pages, vfu, eps=1e-12):
        super().__init__()
        self.L = L
        self.vfu = vfu
        # NN-LUT pages
        self.page_gelu   = pages["gelu"][f"L{L}"]
        self.page_exp    = pages["exp"][f"L{L}"]
        self.page_recip  = pages["recip"][f"L{L}"]
        self.page_rsqrt1 = pages["rsqrt"][f"L{L}.ln1"]
        self.page_rsqrt2 = pages["rsqrt"][f"L{L}.ln2"]
        p = f"bert.encoder.layer.{L}."
        self.rq_q = req_ms(L, "q")          # (mult, shift) tuple
        self.rq_k = req_ms(L, "k")
        self.rq_v = req_ms(L, "v")
        self.rq_ln1_in  = req_ms(L, "ln1_in")
        self.rq_ln1_out = req_ms(L, "ln1_out")
        self.rq_ln2_in  = req_ms(L, "ln2_in")
        self.rq_ln2_out = req_ms(L, "ln2_out")
        # attention
        self.Wq = qp["weight"][f"L{L}.W_q"]["w_int4"].T; self.bq = qp["weight"][f"L{L}.W_q"]["bias_int32"]
        self.Wk = qp["weight"][f"L{L}.W_k"]["w_int4"].T; self.bk = qp["weight"][f"L{L}.W_k"]["bias_int32"]
        self.Wv = qp["weight"][f"L{L}.W_v"]["w_int4"].T; self.bv = qp["weight"][f"L{L}.W_v"]["bias_int32"]
        self.Wo = qp["weight"][f"L{L}.W_o"]["w_int4"].T; self.bo = qp["weight"][f"L{L}.W_o"]["bias_int32"]
        self.ln1_w = qp["weight"][f"L{L}.ln1"]["M_gamma"]; self.ln1_b = qp["weight"][f"L{L}.ln1"]["C_beta"]
        self.ln1_frac = qp["weight"][f"L{L}.ln1"]["F_final"]
        # FFN
        self.W1 = qp["weight"][f"L{L}.W_1"]["w_int4"].T; self.b1 = qp["weight"][f"L{L}.W_1"]["bias_int32"]
        self.W2 = qp["weight"][f"L{L}.W_2"]["w_int4"].T; self.b2 = qp["weight"][f"L{L}.W_2"]["bias_int32"]
        self.ln2_w = qp["weight"][f"L{L}.ln2"]["M_gamma"]; self.ln2_b = qp["weight"][f"L{L}.ln2"]["C_beta"]
        self.ln2_frac = qp["weight"][f"L{L}.ln2"]["F_final"]
        self.eps = eps

    def forward(self, x):
        S = x.shape[0]          # [S, 128]
        H, d = 2, 64            # heads, head_dim

        residual = x
        # Q/K/V projection  [S,128]
        Q = x.to(torch.int32) @ self.Wq.to(torch.int32) + self.bq
        K = x.to(torch.int32) @ self.Wk.to(torch.int32) + self.bk
        V = x.to(torch.int32) @ self.Wv.to(torch.int32) + self.bv

        Q = self.vfu.requant(Q, *self.rq_q)
        K = self.vfu.requant(K, *self.rq_k)
        V = self.vfu.requant(V, *self.rq_v)

        # per head [H, S, d]
        Q = Q.view(S, H, d).transpose(0,1)
        K = K.view(S, H, d).transpose(0,1)
        V = V.view(S, H, d).transpose(0,1)
        # attention scores = QK^T * scaling  [H, S, S]
        scores = Q.to(torch.int32) @ K.transpose(1,2).to(torch.int32)

        pre_softmax, rowsum = self.vfu.pre_softmax(scores, self.page_exp)

        # unnormalized context = P'(pre-softmax) @ V
        ctx_unnorm = pre_softmax.to(torch.int32) @ V.to(torch.int32)

        ctx = self.vfu.post_softmax(ctx_unnorm, rowsum, self.page_recip)
        ctx = ctx.transpose(0,1).reshape(S, 128)

        # attention dense layer
        attn_out = ctx.to(torch.int32) @ self.Wo.to(torch.int32) + self.bo

        ln1_out = self.vfu.layer_norm(
            attn_out, 
            *self.rq_ln1_in, residual, 
            self.ln1_w, 
            self.ln1_b, 
            self.ln1_frac, 
            self.page_rsqrt1
        )

        residual = ln1_out
        # FFN1
        ffn1_out = ln1_out.to(torch.int32) @ self.W1.to(torch.int32) + self.b1

        gelu_out = self.vfu.gelu(ffn1_out, self.page_gelu)

        # FFN2
        ffn2_out = gelu_out.to(torch.int32) @ self.W2.to(torch.int32) + self.b2

        ln2_out = self.vfu.layer_norm(
            ffn2_out, 
            *self.rq_ln2_in, 
            residual, 
            self.ln2_w, 
            self.ln2_b, 
            self.ln2_frac, 
            self.page_rsqrt2
        )

        return ln2_out

class Pooler(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.Wp = sd["bert.pooler.dense.weight"].T; self.bp = sd["bert.pooler.dense.bias"]
    def forward(self, x): # x: [S,128]
        cls = x[0]  # [CLS] = token 0, [128]
        out = cls @ self.Wp + self.bp
        return torch.tanh(out)

class IntBertTiny(nn.Module):
    def __init__(self, sd, qp):
        super().__init__()
        self.pages = load_pages()
        self.vfu = VFU()
        self.emb = Embeddings(sd)
        self.layers = [EncoderLayer(sd, qp, L, self.pages, self.vfu) for L in (0, 1)]
        self.pooler = Pooler(sd)
        self.Wc = sd["classifier.weight"].T; self.bc = sd["classifier.bias"]
        self.quant_scale_emb_out = qp["act"]["emb_out"]["scale"]
        self.deq_scale_pool_in = qp["act"]["L1.ln2_out"]["scale"]
    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = quant(x, self.quant_scale_emb_out)
        for layer in self.layers:
            x = layer(x)
        x = dequant(x, self.deq_scale_pool_in)
        pooled = self.pooler(x)
        logits = pooled @ self.Wc + self.bc
        return logits

def main():
    qp = torch.load(QP_PATH)
    hf  = BertForSequenceClassification.from_pretrained("./bert-tiny-sst2").eval()
    sd  = hf.state_dict()
    tok = BertTokenizerFast.from_pretrained("./bert-tiny-sst2")

    model = IntBertTiny(sd, qp)

    # ---- sanity: one sentence, INT8 vs FP32(HF) ----
    ids = tok("The world best movie", return_tensors="pt")["input_ids"]   # [1,S]
    with torch.no_grad():
        int_logits = model(ids[0])                                # [2]
        hf_logits  = hf(ids).logits[0]                            # [2]
    print("int logits:", int_logits.tolist())
    print("hf  logits:", hf_logits.tolist())
    print("argmax match:", int_logits.argmax().item() == hf_logits.argmax().item())

    # ---- SST-2 validation accuracy ----
    from datasets import load_dataset
    val = load_dataset("stanfordnlp/sst2")["validation"]
    correct = 0
    with torch.no_grad():
        for ex in val:
            ids = tok(ex["sentence"], return_tensors="pt", truncation=True, max_length=64)["input_ids"]
            pred = model(ids[0]).argmax().item()
            correct += (pred == ex["label"])
    print(f"int8 val acc: {correct/len(val):.4f}  (float baseline 0.8142)")


if __name__ == "__main__":
    main()
