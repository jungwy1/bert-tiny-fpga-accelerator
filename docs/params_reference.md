# 파라미터 이름 레퍼런스

> `python dump_params.py`로 재생성. 실제 `.pt` 파일에서 읽어 만들므로 항상 최신.

> 원본: `quant_params.pt`, `requant_params.pt`


---

## `quant_params.pt`

calibration 산출물. activation scale과 양자화된 weight/bias/gamma/beta/table.


### `act` — activation scale (34개)

| 이름 | scale | 관찰 absmax | 출처 |
|------|-------|------------|------|
| `emb_sum` | 8.1587e-03 | 1.036 | calibrated |
| `emb_out` | 6.9286e-02 | 8.799 | calibrated |
| `L0.q` | 4.1658e-02 | 5.291 | calibrated |
| `L0.k` | 3.9702e-02 | 5.042 | calibrated |
| `L0.v` | 6.2142e-02 | 7.892 | calibrated |
| `L0.scores` | 6.6450e-02 | 8.439 | calibrated |
| `L0.probs` | 3.9216e-03 | 0.997 | fixed |
| `L0.ctx` | 4.6328e-02 | 5.884 | calibrated |
| `L0.attn_out` | 6.3776e-02 | 8.100 | calibrated |
| `L0.res1` | 1.0206e-01 | 12.962 | calibrated |
| `L0.ln1_out` | 1.0833e-01 | 13.758 | calibrated |
| `L0.ffn_mid` | 8.9748e-02 | 11.398 | calibrated |
| `L0.ffn_act` | 7.0486e-02 | 8.952 | calibrated |
| `L0.ffn_out` | 1.0387e-01 | 13.191 | calibrated |
| `L0.res2` | 1.9985e-01 | 25.381 | calibrated |
| `L0.ln2_out` | 5.2507e-02 | 6.668 | calibrated |
| `L1.q` | 5.6903e-02 | 7.227 | calibrated |
| `L1.k` | 4.3839e-02 | 5.568 | calibrated |
| `L1.v` | 5.1272e-02 | 6.512 | calibrated |
| `L1.scores` | 8.4833e-02 | 10.774 | calibrated |
| `L1.probs` | 3.9216e-03 | 0.999 | fixed |
| `L1.ctx` | 4.3747e-02 | 5.556 | calibrated |
| `L1.attn_out` | 1.1260e-01 | 14.300 | calibrated |
| `L1.res1` | 1.1192e-01 | 14.214 | calibrated |
| `L1.ln1_out` | 3.8513e-02 | 4.891 | calibrated |
| `L1.ffn_mid` | 7.7117e-02 | 9.794 | calibrated |
| `L1.ffn_act` | 4.1602e-02 | 5.283 | calibrated |
| `L1.ffn_out` | 3.0343e-02 | 3.854 | calibrated |
| `L1.res2` | 4.7015e-02 | 5.971 | calibrated |
| `L1.ln2_out` | 4.0390e-02 | 5.130 | calibrated |
| `pool_in` | 3.0247e-02 | 3.841 | calibrated |
| `pool_mid` | 9.0287e-02 | 11.466 | calibrated |
| `pool_out` | 7.8740e-03 | 1.000 | fixed |
| `logits` | 2.0237e-02 | 2.570 | calibrated |

### `op` — 파라미터를 가진 연산 (22개)


**matmul** (14개)

| 이름 | `w_int8` | `w_scale` | `bias_int32` | `bias_scale` | `in` | `out` |
|------|------|------|------|------|------|------|
| `L0.W_q` | `(128, 128)` int8 | 2.9058e-03 | `(128,)` int32 | 2.0133e-04 | `emb_out` | `L0.q` |
| `L0.W_k` | `(128, 128)` int8 | 2.4489e-03 | `(128,)` int32 | 1.6967e-04 | `emb_out` | `L0.k` |
| `L0.W_v` | `(128, 128)` int8 | 4.0984e-03 | `(128,)` int32 | 2.8396e-04 | `emb_out` | `L0.v` |
| `L0.W_o` | `(128, 128)` int8 | 4.7971e-03 | `(128,)` int32 | 2.2224e-04 | `L0.ctx` | `L0.attn_out` |
| `L0.W_1` | `(512, 128)` int8 | 5.0000e-03 | `(512,)` int32 | 5.4166e-04 | `L0.ln1_out` | `L0.ffn_mid` |
| `L0.W_2` | `(128, 512)` int8 | 1.0027e-02 | `(128,)` int32 | 7.0677e-04 | `L0.ffn_act` | `L0.ffn_out` |
| `L1.W_q` | `(128, 128)` int8 | 3.7078e-03 | `(128,)` int32 | 1.9469e-04 | `L0.ln2_out` | `L1.q` |
| `L1.W_k` | `(128, 128)` int8 | 2.7670e-03 | `(128,)` int32 | 1.4529e-04 | `L0.ln2_out` | `L1.k` |
| `L1.W_v` | `(128, 128)` int8 | 3.1396e-03 | `(128,)` int32 | 1.6485e-04 | `L0.ln2_out` | `L1.v` |
| `L1.W_o` | `(128, 128)` int8 | 9.2865e-03 | `(128,)` int32 | 4.0625e-04 | `L1.ctx` | `L1.attn_out` |
| `L1.W_1` | `(512, 128)` int8 | 7.3651e-03 | `(512,)` int32 | 2.8365e-04 | `L1.ln1_out` | `L1.ffn_mid` |
| `L1.W_2` | `(128, 512)` int8 | 4.9453e-03 | `(128,)` int32 | 2.0573e-04 | `L1.ffn_act` | `L1.ffn_out` |
| `W_pool` | `(128, 128)` int8 | 6.0976e-03 | `(128,)` int32 | 1.8444e-04 | `pool_in` | `pool_mid` |
| `W_cls` | `(2, 128)` int8 | 5.3137e-04 | `(2,)` int32 | 4.1840e-06 | `pool_out` | `logits` |

**LayerNorm** (5개)

| 이름 | `gamma_int8` | `gamma_scale` | `beta_int32` | `beta_scale` | `in` | `out` |
|------|------|------|------|------|------|------|
| `emb_ln` | `(128,)` int8 | 1.2045e-02 | `(128,)` int32 | 6.9926e-10 | `emb_sum` | `emb_out` |
| `L0.ln1` | `(128,)` int8 | 1.4321e-02 | `(128,)` int32 | 9.5390e-10 | `L0.res1` | `L0.ln1_out` |
| `L0.ln2` | `(128,)` int8 | 1.1475e-02 | `(128,)` int32 | 2.8386e-10 | `L0.res2` | `L0.ln2_out` |
| `L1.ln1` | `(128,)` int8 | 1.3106e-02 | `(128,)` int32 | 5.4181e-10 | `L1.res1` | `L1.ln1_out` |
| `L1.ln2` | `(128,)` int8 | 1.0702e-02 | `(128,)` int32 | 2.1082e-10 | `L1.res2` | `L1.ln2_out` |

**embedding** (3개)

| 이름 | `table_int8` | `scale` | `out` |
|------|------|------|------|
| `emb_word` | `(30522, 128)` int8 | 4.7531e-03 | `emb_out` |
| `emb_pos` | `(512, 128)` int8 | 4.1626e-03 | `emb_out` |
| `emb_type` | `(2, 128)` int8 | 1.4313e-03 | `emb_out` |

---

## `requant_params.pt`

compile 산출물. 하드웨어가 쓰는 정수 상수만 담는다 (런타임 float 없음).


### `requant` — 누산기 → 다음 INT8 (15개)

| 이름 | kind | M | M_int | shift | sign | acc비트 | 곱비트 |
|------|------|---|-------|-------|------|--------|-------|
| `emb_out` | layernorm | 1.6977e-04 | 746664300 | 42 | 1 | 20 | 50 |
| `L0.q` | matmul | 4.8330e-03 | 664239646 | 37 | 1 | 15 | 45 |
| `L0.k` | matmul | 4.2737e-03 | 587379517 | 37 | 1 | 15 | 45 |
| `L0.v` | matmul | 4.5696e-03 | 628037242 | 37 | 1 | 15 | 45 |
| `L0.ctx` | attn_matmul | 5.2601e-03 | 722945827 | 37 | 1 | 15 | 45 |
| `L0.ln1_out` | layernorm | 1.2910e-04 | 567804227 | 42 | 1 | 20 | 50 |
| `L0.ffn_act` | gelu | 4.6908e-07 | 1056284131 | 51 | 1 | 29 | 59 |
| `L0.ln2_out` | layernorm | 2.1342e-04 | 938637353 | 42 | 1 | 20 | 50 |
| `L1.q` | matmul | 3.4214e-03 | 940467204 | 38 | 1 | 16 | 46 |
| `L1.k` | matmul | 3.3142e-03 | 910990045 | 38 | 1 | 16 | 46 |
| `L1.v` | matmul | 3.2152e-03 | 883784905 | 38 | 1 | 16 | 46 |
| `L1.ctx` | attn_matmul | 4.5961e-03 | 631689129 | 37 | 1 | 15 | 45 |
| `L1.ln1_out` | layernorm | 3.3233e-04 | 730802705 | 41 | 1 | 19 | 49 |
| `L1.ffn_act` | gelu | 4.1620e-07 | 937208123 | 51 | 1 | 29 | 59 |
| `L1.ln2_out` | layernorm | 2.5875e-04 | 568998617 | 41 | 1 | 19 | 49 |

### `align` — 덧셈 전 scale 정렬 (7개)

| 이름 | M | M_int | shift | 정렬 대상 |
|------|---|-------|-------|----------|
| `emb_sum.emb_word` | 3.3209e+00 | 891452091 | 28 | `emb_word` |
| `emb_sum.emb_pos` | 2.9084e+00 | 780706093 | 28 | `emb_pos` |
| `emb_sum.emb_type` | 1.0000e+00 | 1073741824 | 30 | `emb_type` |
| `L0.res1` | 3.1176e+02 | 653807545 | 21 | `emb_out` |
| `L0.res2` | 1.5327e+02 | 642879429 | 22 | `L0.ln1_out` |
| `L1.res1` | 1.2925e+02 | 542099556 | 22 | `L0.ln2_out` |
| `L1.res2` | 1.8720e+02 | 785162311 | 22 | `L1.ln1_out` |

### `nonlin` — 비선형 유닛 하드와이어 상수 (10개)


**gelu**

| 이름 | `S_in` | `S_out` | `clip_qmax` | `q_b` | `q_c` | `erf_M_int` | `erf_shift` | `erf_sign` | `q_1` |
|------|------|------|------|------|------|------|------|------|------|
| `L0.gelu` | 5.4166e-04 | 3.3064e-08 | 4618 | -4619 | -23603991 | 763099751 | 41 | -1 | 8191 |
| `L1.gelu` | 2.8365e-04 | 1.7315e-08 | 8819 | -8820 | -86071746 | 837078388 | 43 | -1 | 8191 |

**softmax**

| 이름 | `S_in` | `S_out` | `q_ln2` | `q_b` | `q_c` | `exp_M_int` | `exp_shift` | `exp_qmax` |
|------|------|------|------|------|------|------|------|------|
| `L0.softmax` | 2.0673e-04 | 3.9216e-03 | 3352 | 6544 | 22451591 | 552019527 | 39 | 65535 |
| `L1.softmax` | 3.1182e-04 | 3.9216e-03 | 2222 | 4339 | 9868981 | 627912639 | 38 | 65535 |

**tanh**

| 이름 | `S_in` | `S_out` | `q_ln2` | `q_b` | `q_c` | `exp_M_int` | `exp_shift` | `exp_qmax` |
|------|------|------|------|------|------|------|------|------|
| `tanh` | 2.4629e-04 | 7.8740e-03 | 2814 | 5493 | 15819366 | 783452172 | 39 | 65535 |

**layernorm**

| 이름 | `K` | `S_out` | `beta_out_int32` |
|------|------|------|------|
| `emb_ln` | 1024 | 1.1763e-05 | `(128,)` int64 |
| `L0.ln1` | 1024 | 1.3986e-05 | `(128,)` int64 |
| `L0.ln2` | 1024 | 1.1206e-05 | `(128,)` int64 |
| `L1.ln1` | 1024 | 1.2799e-05 | `(128,)` int64 |
| `L1.ln2` | 1024 | 1.0451e-05 | `(128,)` int64 |

### `fused` — 연산에 융합, 곱수 없음 (3개)

| 이름 | scale | 비고 |
|------|-------|------|
| `L0.probs` | 3.9216e-03 | softmax normalization already scales to 1/255 |
| `L1.probs` | 3.9216e-03 | softmax normalization already scales to 1/255 |
| `pool_out` | 7.8740e-03 | i_tanh's final division already scales to 1/127 |

### 기타

- `K_layernorm` = `1024`

