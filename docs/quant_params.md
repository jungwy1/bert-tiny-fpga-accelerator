# Quant Params: `quant_params.pt` 구조와 사용법

`python/export_params.py` 가 float reference model 을 calibration 해서 뽑은 정수 파라미터 번들.
**A8W4** (activation INT8 / matmul weight INT4), 전부 **symmetric per-tensor** (zero-point 없음).

```python
import torch
qp = torch.load("quant_params.pt")
qp["act"]     # dict: activation 이름 -> scale 정보
qp["weight"]  # dict: op 이름 -> weight/bias/gamma/beta/table (정수) + scale
```

---

## 1. 최상위 구조

| 키 | 내용 |
|----|------|
| `qp["act"]` | 34개 activation 의 **입력/출력 scale** (`S_x = max\|x\|/qmax`) |
| `qp["weight"]` | 22개 op 의 정수 weight/bias/gamma/beta/embed + 각자의 scale |

전부 대칭 양자화라 정수↔실수 변환은 `x ≈ q * S`, `q = round(x / S)` 만 있으면 됨.

---

## 2. `qp["act"]` — activation scale

각 항목:

```python
qp["act"]["L0.q"] = {
    "scale":          0.04249,   # S_x = max|x| / 127
    "observed_absmax": 5.396,    # calibration 에서 관측한 max|x|
    "source":         "calibrated",   # or "fixed"
}
```

- `source == "calibrated"`: 512문장 calibration 의 running absmax 로 결정.
- `source == "fixed"`: 수학적으로 range 가 알려진 것 → scale 고정.
  - `probs` (softmax 출력, [0,1]) 와 `pool_out` (tanh, [-1,1]) → **1/127**.

### 관측된 scale (calibration 512문장)

| activation | scale | absmax | src |
|---|---|---|---|
| emb_sum | 8.220e-03 | 1.044 | calibrated |
| emb_out | 6.920e-02 | 8.789 | calibrated |
| L0.q | 4.249e-02 | 5.396 | calibrated |
| L0.k | 4.028e-02 | 5.116 | calibrated |
| L0.v | 6.286e-02 | 7.983 | calibrated |
| L0.scores | 7.412e-02 | 9.413 | calibrated |
| L0.probs | 7.874e-03 | 0.999 | **fixed** (1/127) |
| L0.ctx | 5.524e-02 | 7.016 | calibrated |
| L0.attn_out | 6.917e-02 | 8.785 | calibrated |
| L0.res1 | 1.035e-01 | 13.148 | calibrated |
| L0.ln1_out | 1.111e-01 | 14.115 | calibrated |
| L0.ffn_mid | 8.715e-02 | 11.068 | calibrated |
| L0.ffn_act | 8.564e-02 | 10.877 | calibrated |
| L0.ffn_out | 1.448e-01 | 18.394 | calibrated |
| L0.res2 | 2.408e-01 | 30.580 | calibrated |
| L0.ln2_out | 5.260e-02 | 6.680 | calibrated |
| L1.q | 5.484e-02 | 6.965 | calibrated |
| L1.k | 4.114e-02 | 5.224 | calibrated |
| L1.v | 5.114e-02 | 6.495 | calibrated |
| L1.scores | 8.349e-02 | 10.603 | calibrated |
| L1.probs | 7.874e-03 | 1.000 | **fixed** (1/127) |
| L1.ctx | 4.115e-02 | 5.226 | calibrated |
| L1.attn_out | 1.339e-01 | 17.010 | calibrated |
| L1.res1 | 1.457e-01 | 18.505 | calibrated |
| L1.ln1_out | 3.876e-02 | 4.922 | calibrated |
| L1.ffn_mid | 7.971e-02 | 10.123 | calibrated |
| L1.ffn_act | 5.129e-02 | 6.514 | calibrated |
| L1.ffn_out | 2.911e-02 | 3.697 | calibrated |
| L1.res2 | 5.598e-02 | 7.109 | calibrated |
| L1.ln2_out | 4.539e-02 | 5.765 | calibrated |
| pool_in | 3.881e-02 | 4.929 | calibrated |
| pool_mid | 9.257e-02 | 11.757 | calibrated |
| pool_out | 7.874e-03 | 1.000 | **fixed** (1/127) |
| logits | 2.264e-02 | 2.876 | calibrated |

> `bits used = ceil(log2(absmax/scale + 1))` = 전부 7 → INT8 (부호 1비트) 를 꽉 채워 씀.

---

## 3. `qp["weight"]` — 정수 weight/bias/param

3가지 종류. 각 항목에 `in`/`out` 이 있어 어느 activation scale 과 짝인지 알려줌.

### (a) matmul weight (Q/K/V/O, FFN1/2, pooler, classifier) — 8×2+2 = 14개

```python
qp["weight"]["L0.W_q"] = {
    "w_int4":     Tensor[out,in] int8,   # 값 범위 [-7,7] (INT4 를 int8 컨테이너에 저장)
    "w_scale":    S_w = max|W| / 7,
    "bias_int32": Tensor[out] int32,     # accumulator domain 에 이미 양자화됨
    "bias_scale": S_acc = S_in * S_w,    # bias_int32 의 scale (= acc scale)
    "in":  "emb_out",   # 입력 activation 이름 -> qp["act"]["emb_out"]["scale"]
    "out": "L0.q",      # 출력 activation 이름 -> requant target scale
}
```

이름: `L{0,1}.W_{q,k,v,o,1,2}`, `W_pool`, `W_cls`.

### (b) LayerNorm (gamma INT8 / beta INT32) — 5개

```python
qp["weight"]["L0.ln1"] = {
    "gamma_int8": Tensor int8,  "gamma_scale": S_g,
    "beta_int32": Tensor int32, "beta_scale":  S_b,
    "in": "L0.res1", "out": "L0.ln1_out",
}
```

이름: `emb_ln`, `L{0,1}.ln{1,2}`.

### (c) embedding table (INT8, PS 담당) — 3개

```python
qp["weight"]["emb_word"] = {
    "table_int8": Tensor int8, "scale": S_t, "out": "emb_out",
}
```

이름: `emb_word`, `emb_pos`, `emb_type`.

---

## 4. 사용법 — 정수 datapath 재구성

### 4.1 대칭 양자화 기본식

```python
q = round(x / S)          # float -> int
x = q * S                 # int   -> float (dequant)
```

### 4.2 Linear GEMM (act × weight) + bias + requant

핵심: 모든 곱은 **정수**로, scale 은 나중에 한 번에 곱함 (requant).

```python
def int_linear(x_int8, w):                 # w = qp["weight"]["L0.W_q"]
    S_in  = qp["act"][w["in"]]["scale"]    # 입력 activation scale
    S_out = qp["act"][w["out"]]["scale"]   # 출력 activation scale
    S_w   = w["w_scale"]

    # 1) 정수 MAC:  acc = sum(x_int8 * w_int4)   -> INT32
    acc = x_int8.int() @ w["w_int4"].int().T   # [.,in]@[in,out]

    # 2) bias: 이미 acc domain (S_acc = S_in*S_w) 이라 정수 덧셈만
    acc = acc + w["bias_int32"]

    # 3) requant: acc(정수, scale S_in*S_w) -> 출력 int8(scale S_out)
    #    x_out = acc * S_in*S_w  ;  q_out = round(x_out / S_out)
    M = (S_in * S_w) / S_out                # dyadic 상수 하나 (HW: mult+shift)
    return (acc * M).round().clamp(-127, 127).to(torch.int8)
```

- **M = S_in·S_w / S_out** 이 op 당 스칼라 1개. HW 에서는 `(acc * M0) >> shift` dyadic 로 구현.
- bias 는 export 시 이미 `S_acc = S_in·S_w` 로 나눠 저장 → 런타임 정수 덧셈만.
- `w_int4` 는 int8 컨테이너지만 값이 [-7,7] → INT4 packing (`W1<<20 + W2`) 시 그대로 사용.

### 4.3 attention matmul (Q@Kᵀ, P@V)

weight 가 없는 act×act. 두 입력 다 정수, 출력 scale 로 requant.

```python
# scores = Q @ K^T :  S_score_acc = S_q * S_k
acc = q_int8.int() @ k_int8.int().transpose(-1,-2)
scores_int = (acc * (S_q*S_k / S_scores)).round()...   # -> L{i}.scores scale

# ctx = P @ V :  probs scale 고정 1/127
acc = p_int8.int() @ v_int8.int()
ctx_int = (acc * ((1/127)*S_v / S_ctx)).round()...     # -> L{i}.ctx scale
```

### 4.4 LayerNorm

```python
ln = qp["weight"]["L0.ln1"]
# gamma * normalized + beta.  gamma_int8(S_g), beta_int32(S_b) 를 정수 datapath 에 fold.
# 자세한 정수 LN 수식은 I-BERT / SwiftTron 참고 (mean/var 는 wide 정수).
```

### 4.5 비선형 (softmax / gelu / tanh) — NN-LUT

- 입력은 **INT32 accumulator**, 출력은 다음 op 의 입력 scale 로 **fused requant**.
- `probs`/`pool_out` 은 range 고정(1/127)이라 LUT 의 output scale 에 그대로 반영.
- 학습: calibration 으로 얻은 입력 range 로 16-entry piecewise-linear fit → `rtl/nn_lut.v`.

---

## 5. scale 흐름 요약 (한 op = 3 scale)

```
  입력 act (S_in) ──┐
                    ├─ 정수 MAC ─> acc (S_in·S_w) ─+bias(int32)─> requant(×M) ─> 출력 act (S_out)
  weight  (S_w)  ──┘                                                    M = S_in·S_w / S_out
```

- **S_in, S_w**: 곱해져서 acc domain 결정 (bias 도 여기).
- **S_out**: requant target. 다음 op 의 `in` scale 과 동일해야 체인 성립.
- 각 op requant 상수 **M 1개** → HW 는 op 당 `(mult, shift)` 한 쌍만 저장.