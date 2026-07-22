# INT8 데이터패스 명세 (int_model / 하드웨어 공통)

> `model_float.py`(검증된 아키텍처) + `int_ops.py`(정수 프리미티브) + 컴파일된 상수로
> 정수 전용 추론을 조립하기 위한 명세. 각 단계가 **어떤 파일의 어떤 키**를 쓰는지 명시한다.

---

## 0. 입력 파일 3종

| 파일 | 키 | 내용 |
|------|-----|------|
| `quant_params.pt` | `op[…]` | `w_int8`, `w_scale`, `bias_int32`, `gamma_int8`, `beta_int32`, `table_int8` |
| | `act[…]` | activation scale (관찰) — **대부분 compile 단계에서 이미 소비됨** |
| `requant_params.pt` | `requant[…]` | `M_int`, `shift`, `sign` — 누산기 → 다음 INT8 |
| | `align[…]` | `M_int`, `shift` — 덧셈 전 scale 정렬 |
| | `nonlin[…]` | 비선형 유닛의 하드와이어 상수 |
| | `fused[…]` | 곱수 없는 지점 (연산에 융합) |

**런타임에 float scale은 쓰지 않는다.** 전부 정수 상수로 컴파일되어 있다.

---

## 1. 폭 규약

```
레이어 간 데이터  : INT8
누산기 / 비선형 I/O : INT32
곱셈 중간값        : 필요한 만큼 (HW는 임의 폭, 예: 제곱기 18×18→36)
```

---

## 2. Embedding

| 단계 | 연산 | 파라미터 |
|------|------|----------|
| 1 | 테이블 lookup ×3 | `op["emb_word"/"emb_pos"/"emb_type"].table_int8` |
| 2 | **scale 정렬** (3개 테이블 scale이 다름) | `align["emb_sum.emb_word"/"emb_pos"/"emb_type"]` |
| 3 | 합산 → `emb_sum` (INT32) | — |
| 4 | LayerNorm | `nonlin["emb_ln"].K`, `op["emb_ln"].gamma_int8/gamma_scale/beta_int32/beta_scale` |
| 5 | requant → `emb_out` (INT8) | `requant["emb_out"]` |

```python
q = word[ids]*A_w + pos[0:S]*A_p + type[0]*A_t      # A_* = align dyadic
q, S_ln = i_layernorm(q, S_common, g, S_g, b, S_b, K)
q = requant(q, **requant["emb_out"])                 # -> INT8
```

> `emb_type`은 단일 문장이라 0번 행만 쓴다. `emb_pos`도 앞 S행만. HW에서 잘라낼 수 있음.

---

## 3. Encoder Layer `L` (0, 1)

입력: `x` = INT8 (`emb_out` 또는 `L{L-1}.ln2_out`)

### 3.1 QKV projection

| 연산 | 파라미터 |
|------|----------|
| `acc = x·W + bias` (INT8×INT8→INT32) | `op["L{L}.W_q"/"W_k"/"W_v"].w_int8`, `.bias_int32` |
| requant → INT8 | `requant["L{L}.q"/"k"/"v"]` |

### 3.2 Attention score → softmax

| 연산 | 파라미터 |
|------|----------|
| `scores = q·kᵀ` (INT8×INT8→INT32) | — (weight 없음) |
| **`1/√d`는 scale에 흡수** — 시프트 불필요 | `nonlin["L{L}.softmax"].S_in = s_q·s_k/√64` |
| `i_softmax` | `nonlin["L{L}.softmax"]`: `q_ln2`, `q_b`, `q_c`, `exp_M_int/shift`, `exp_qmax` |
| 출력 `probs` = INT8 @ **1/255** | `fused["L{L}.probs"]` — **requant 없음 (융합)** |

### 3.3 Context (P·V)

| 연산 | 파라미터 |
|------|----------|
| `acc = probs·v` (INT8×INT8→INT32) | — (양쪽 다 activation) |
| requant → `ctx` (INT8) | `requant["L{L}.ctx"]` |

### 3.4 Attention output + residual

| 연산 | 파라미터 |
|------|----------|
| `acc = ctx·W_o + bias` → INT32 | `op["L{L}.W_o"]` |
| **`x`를 누산기 도메인으로 정렬** | `align["L{L}.res1"]` |
| `res1 = acc + x_aligned` (INT32) | — |

> `attn_out`은 INT8이 되지 않는다. 누산기 그대로 residual로 간다.

### 3.5 LayerNorm 1

| 연산 | 파라미터 |
|------|----------|
| `i_layernorm(res1, …)` | `nonlin["L{L}.ln1"].K`, `op["L{L}.ln1"].gamma_int8/beta_int32` |
| requant → `ln1_out` (INT8) | `requant["L{L}.ln1_out"]` |

### 3.6 FFN1 + GELU

| 연산 | 파라미터 |
|------|----------|
| `acc = ln1_out·W_1 + bias` → INT32 | `op["L{L}.W_1"]` |
| `i_gelu` — **누산기를 직접 받음** (requant 없음) | `nonlin["L{L}.gelu"]`: `clip_qmax`, `q_b`, `q_c`, `erf_M_int/shift`, `q_1` |
| requant → `ffn_act` (INT8) | `requant["L{L}.ffn_act"]` |

### 3.7 FFN2 + residual

| 연산 | 파라미터 |
|------|----------|
| `acc = ffn_act·W_2 + bias` → INT32 | `op["L{L}.W_2"]` |
| `x`(=`ln1_out`) 정렬 | `align["L{L}.res2"]` |
| `res2 = acc + x_aligned` | — |

### 3.8 LayerNorm 2

| 연산 | 파라미터 |
|------|----------|
| `i_layernorm(res2, …)` | `nonlin["L{L}.ln2"]`, `op["L{L}.ln2"]` |
| requant → `ln2_out` (INT8) | `requant["L{L}.ln2_out"]` |

---

## 4. Head

| 단계 | 연산 | 파라미터 |
|------|------|----------|
| 1 | `pool_in = L1.ln2_out[0]` — **같은 버퍼의 0번 행** | requant **없음** (scale 상속) |
| 2 | `acc = pool_in·W_pool + bias` → INT32 | `op["W_pool"]` |
| 3 | `i_tanh` — 누산기 직접 | `nonlin["tanh"]`: `q_ln2`, `q_b`, `q_c`, `exp_M_int/shift` |
| 4 | 출력 `pool_out` = INT8 @ **1/127** | `fused["pool_out"]` — **requant 없음 (융합)** |
| 5 | `logits = pool_out·W_cls + bias` → INT32 | `op["W_cls"]` |
| 6 | `argmax(logits)` | **requant 없음** — 대소 비교는 scale 무관 |

---

## 5. requant / align 규칙

```python
# requant : 누산기 -> 좁은 도메인
q_out = clamp( (acc * M_int + (1 << (shift-1))) >> shift , qmin, qmax) * sign

# align   : 덧셈 전 한쪽을 다른 쪽 도메인으로 (clamp 없음, 넓히는 방향)
x_aligned = (x * M_int + (1 << (shift-1))) >> shift
```

- 둘 다 **정수 곱 + 반올림 우시프트**. 나눗셈기 불필요 (SwiftTron Eq. 2).
- requant는 **포화(saturate)**, 절대 wraparound 금지.
- `sign`은 GELU처럼 유도 scale이 음수인 경우에만 -1.

---

## 6. 왜 어떤 지점엔 requant가 없는가

| 지점 | 이유 |
|------|------|
| `scores`, `ffn_mid`, `pool_mid` | 비선형이 **누산기를 직접** 받음 (I-BERT/SwiftTron 방식) |
| `attn_out`, `ffn_out` | residual 덧셈으로 감 (align만) |
| `res1`, `res2` | LayerNorm이 INT32를 받음 |
| `probs`, `pool_out` | 출력 범위가 수학적으로 고정 → **연산에 융합** |
| `pool_in` | `ln2_out`의 슬라이스, scale 상속 |
| `logits` | argmax는 scale 무관 |

→ `act` 34개 중 **requant 목표로 실제 쓰이는 건 15개**. 나머지는 비트폭 사이징 근거.

---

## 7. 비선형 유닛이 INT32에 들어가는 이유

범위가 **수학적으로 알려진 중간값은 고정 scale로 requant**한다. 다항식의 자연 scale
(`a·S²`)은 입력이 정밀할수록 폭발하지만, 값의 범위는 그와 무관하게 고정이기 때문.

| 중간값 | 범위 | 고정 scale |
|--------|------|-----------|
| `erf` (i_gelu 내부) | [-1, 1] | `1/ERF_QMAX` (13b) |
| `exp` (i_exp 출력) | (0, 1] | `1/EXP_QMAX` (16b) |
| `probs` | [0, 1] | 1/255 |
| `pool_out` | [-1, 1] | 1/127 |

효과 (실측):

| | 전 | 후 |
|---|---|---|
| GELU 출력 | 40~43b (INT64) | **29~30b (INT32)** |
| GELU 곱셈기 | 40~43b | **~28b** |
| GELU `q_1` | 레이어별 2400만~8600만 | **8191 고정** |
| softmax `q_exp×255` | 43~63b (오버플로) | **24b** |

→ 입력 requant도 출력 shift도 불필요. **INT32 in → INT32 out**이 자연히 성립.

---

## 8. 검증 계획

```
1. int_model 정확도  vs  fake-quant(0.8177)  vs  FP32(0.8142)
2. 각 지점 정수값의 실제 비트폭 로깅 → HW 사이징 확정
3. 레이어별 중간 텐서를 float 모델과 대조 (어디서 오차가 커지는지)
```
