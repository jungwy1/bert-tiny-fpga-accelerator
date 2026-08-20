# 온칩 메모리 구성

> BERT-Tiny **A8W4** accelerator · KV260 / **xck26-sfvc784-2LV-c**
> INT8 activation · INT4 weight(pack2) · per-tensor. 전량 on-chip 상주.

---

## 0. 전제

- **모델**: seq=64, hidden=128, heads=2, head_dim=64, intermediate=512, layer=2.
- **전량 on-chip**: inference 중 DRAM 트래픽 0. DDR는 **init 때 weight 로드 1회**만.
- **datapath = 전용 포트**: operand마다 독립 뱅크 → crossbar/arbiter 없음. bank 선택은 **decode+mux**(경합 없음).
- **저장 규약**: 모든 operand **k-major(feature-major)** — drain이 공짜로 생산. token 단위(LN 등)는 병렬 accumulator로 처리.
- **HW transpose = V 하나만** (P·V가 NT-native array에서 NN이라). 나머지 transpose는 없음/논리적(공짜)/host.

### 리소스 (xck26)
| 자원 | 블록 | 블록당 | 총량 |
|------|------|--------|------|
| BRAM (RAMB36E2) | 144 | 36 Kbit | ≈ 648 KB |
| URAM (URAM288E2) | 64 | 288 Kbit | ≈ 2.25 MB |

- URAM: 4096×72 **고정**. 128-bit = 2블록 병렬(72 중 64 사용). 4096 이내면 cascade 없음.
- BRAM: geometry 가변(…512×72 SDP). 좁은 폭/flexible에 유리.

---

## 1. 메모리 맵 (요약)

| 뱅크 | 메모리 | geometry | 용도 |
|------|--------|----------|------|
| Weight QKVO_T (L0+L1) | URAM ×2 | 4096 × 128 (full) | Wq/Wk/Wv/Wo × 2 layer |
| Weight L0 FFN_T | URAM ×2 | 4096 × 128 (full) | W1/W2 |
| Weight L1 FFN_T | URAM ×2 | 4096 × 128 (full) | W1/W2 |
| residual(+X) | BRAM ×2 | 512 × 128 | layer 입력 = residual skip 공용 |
| scratch b0 | BRAM ×2 | 512 × 128 | Q → PV → H |
| scratch b1 | BRAM ×2 | 512 × 128 | K_T → H |
| scratch b2 | BRAM ×2 | 512 × 128 | V → H |
| scratch b3 | BRAM ×2 | 512 × 128 | P → H |
| bias | BRAM ×3 (직렬) | 1536 × 64 (2 INT32/word) | GEMM per-channel bias |
| ln_param | BRAM ×1 | 512 × 72 (40 used) | LN γ(INT8)+β(INT32), 4 LN × 128 |
| V transpose | reg-array (FF) | 16×16 INT8 ×2 | corner-turn (ping-pong) |
| INT32 결과 | — | 버퍼 없음 | VFU 직통 |

합계: **URAM 6 / BRAM 14** (+ transpose reg-array). LN 2-pass 버퍼 등 VFU 내부는 별도.

---

## 2. Weight (URAM)

INT4 weight를 **`Wᵀ` row-major**(hidden=k-major)로 저장. word = 128-bit = **32 INT4** = 16 column × 2 feature(pack2).

- **뱅크 = 2 URAM 병렬 = 4096 × 128**, **네이티브 4096 depth → cascade latency 없음.**
- gap-20 pack은 **DSP pre-adder**가 런타임에 (RAM엔 dense INT4 그대로).

### 뱅크 내 배치 (base = 앞 매트릭스 누적)
| 뱅크 | 매트릭스 | word | base |
|------|----------|------|------|
| QKVO (L0+L1) | Wq / Wk / Wv / Wo | 각 512 | **L·2048** + {0 / 512 / 1024 / 1536} |
| FFN (per-layer) | W1 (128→512) / W2 (512→128) | 각 2048 | 0 / 2048 |

- QKVO: L0(0–2047) + L1(2048–4095) = 4096 **꽉 참** → 2 URAM. base에 **layer offset `L·2048`**.
- FFN: per-layer 뱅크, 2048+2048 = 4096 (꽉 참). 합치면 8192 → cascade 생기니 **layer 분리 유지**. W2는 K=512 (누산 깊이 512).

---

## 3. Activation / Intermediate (BRAM)

INT8. word = 128-bit = **16 token @ 고정 k**(k-major). 전용 뱅크라 **base = 0**.

### residual(+X) 뱅크
- 64 tok × 128 hidden = 8 KB. layer 입력이 곧 residual skip → **한 자리 in-place 순환**(emb→ln1→ln2→…).

### scratch 4뱅크 (phase 재사용)
Q/K_T/V/P 를 4뱅크로 (Q는 scores 후 죽어 PV가 재사용), FFN 땐 4뱅크 통째 = H(32 KB).

| 뱅크 | attention | FFN |
|------|-----------|-----|
| b0 | **Q → PV** | H[0] |
| b1 | **K_T** | H[1] |
| b2 | **V** | H[2] |
| b3 | **P** | H[3] |

**동시 read = 항상 다른 뱅크** (뱅킹 목적):
| op | act | weight | write |
|----|-----|--------|-------|
| scores | Q(b0) | K_T(b1) | P → b3 |
| context | P(b3) | V(b2) | PV → b0 (죽은 Q) |

→ act/weight가 물리적으로 안 겹쳐 각 뱅크 독립 포트로 병렬. **arbiter 불필요.**

---

## 4. bias (BRAM)

GEMM per-channel INT32 bias, accumulator 도메인에서 C포트로 더함. 폭 좁고 작아 BRAM.

- **저장 = raw INT32**, word = 64-bit = **2 INT32 = (bias_b, bias_a) column 쌍**.
- **BRAM ×3 (직렬/depth)** = 512×3 = 1536 word × 64 = **3072 INT32 capacity** (2304 사용).
- 필요량: (Q/K/V/O 128×4 + FFN1 512 + FFN2 128) × 2 layer = **2304**. scores·context는 bias 없음.

### 배치 (layer 내 offset, word 단위 = 채널쌍)
| 매트릭스 | 채널 | word | offset |
|----------|------|------|--------|
| Wq/Wk/Wv/Wo | 128×4 | 64×4 | 0/64/128/192 |
| W1 | 512 | 256 | 256 |
| W2 | 128 | 64 | 512 |
| layer 합 | 1152 | 576 | (layer offset += 576) |

### on-chip gap-20 pack (preload)
pre-adder는 weight가 쓰므로 bias는 **fabric에서 pack**:
```
read word c = {bias_b, bias_a}  →  C = (bias_b<<20) + bias_a  →  bias_col[c]
```
- tile당 16 read(column쌍 1개씩) → 16×48b `bias_col` 레지스터 → init에서 C포트.
- **이전 tile drain 중 preload** → critical path 밖 (depth 3-stack의 ~2 cycle latency 완전 흡수).

---

## 4b. ln_param (BRAM, LN γ/β)

LayerNorm affine `y = γ·(x−μ)/σ + β`. **VFU**가 Res+Norm 때 읽음. encoder-only 스코프라 LN 4개
(layer당 2개: post-attn / post-FFN × 2 layer). embedding LN은 host.

- **γ+β를 한 word에 pack**: word = **γ(INT8) + β(INT32) = 40-bit** (72 중 40 사용).
- word 수 = **4 LN × 128 feature = 512** = BRAM36 SDP 깊이(512) **딱 맞음** → **BRAM ×1**.

### 배치
```
word layout:  [39:32] = γ (INT8)   [31:0] = β (INT32)
addr        =  ln_id·128 + feature       (ln_id 0..3, feature 0..127)
```
| ln_id | LN | addr 범위 |
|-------|----|-----------|
| 0 | L0 post-attn | 0 – 127 |
| 1 | L0 post-FFN | 128 – 255 |
| 2 | L1 post-attn | 256 – 383 |
| 3 | L1 post-FFN | 384 – 511 |

- **1 read = γ,β 동시** (VFU feature당 1접근). command `ln_base = ln_id·128`.
- γ INT8 민감하면 폭 조정(β↓/γ INT16) — 40→여전히 72 안이라 여유.
- (γ/β 실제 폭·스케일은 quant sweep / VFU 구현 기준으로 최종 확정.)

---

## 5. Layout & Transpose 정책

- **모든 operand k-major(feature-major)** 저장 → act/weight 둘 다 128-bit 1-read(full-width). drain이 이 layout을 공짜로 생산.
- **NT-native array** (`C[r][c]=Σ_k A[r][k]·W[c][k]` = A·Wᵀ):
  - **Q·Kᵀ (scores) = NT** → K는 "논리적 ᵀ"이나 **배선(w_col에 꽂기)로 공짜**, 물리 transpose 없음.
  - **P·V (context) = NN** → NT array라 **V만 물리 transpose 필요**.
- **V transpose** = INT8 **reg-array corner-turn**(16×16, 열-write/행-read, ping-pong ~4Kbit FF). VFU 출력 → V뱅크 경로. bias는 C포트 그대로 유지.
- **최종 output** = hidden-major 그대로 host 전달(계약), transpose는 host(SW). (SST-2는 로짓 벡터라 무의미.)
- 즉 **런타임 물리 transpose = V 딱 하나.**

---

## 6. 주소 생성

각 뱅크는 **로컬 주소공간**. 주소 = `base(matmul) + tile·stride + counter`.

- **카운터**: `tr`=token-tile(0..M/16−1), `tc`=feature-tile(0..N/32−1, pack2), `k`=수축(0..K−1, stream), `i`=bias preload(0..15), `d`=drain(0..15). `NT`=col-tile 수, `MT`=token-tile 수(=M/16).
- **저장 규약**: weight/intermediate 모두 **feature-major(k-major)** 로 통일 → weight `k·NT+tc`, act `k·MT+tr`, drain-write `f·MT+token_tile`. **drain-write와 act/weight read가 대칭**이라 addr-gen 균일, drain 한 모드.

### 주소식 (뱅크-로컬)
| operand | 주소 | 저장 layout |
|---------|------|------|
| weight | `baseW + k·NT + tc` | **natural `Wᵀ` row-major** (feature-major) |
| act | `baseA + k·MT + tr` | intermediate **feature-major** (drain-written) |
| bias | `baseB + tc·16 + i` | column 쌍 순 |
| out write | `baseO + o_feat·MT + o_tr` | feature-major, pack2 → drain당 2 word |

### matmul별 base (layer L)
| matmul | act (뱅크:base) | weight (뱅크:base) | out (뱅크:base) | K | N |
|--------|------|------|------|---|---|
| Q = X·Wq | resid:0 | QKVO:L·2048+0 | b0:0 | 128 | 128 |
| K = X·Wk | resid:0 | QKVO:L·2048+512 | b1:0 | 128 | 128 |
| V = X·Wv | resid:0 | QKVO:L·2048+1024 | b2:0 (→transpose) | 128 | 128 |
| scores = Q·Kᵀ | b0:0 | b1:0 | b3:0 (→softmax) | 64\* | 64\* |
| context = P·V | b3:0 | b2:0 | b0:0 (PV) | 64\* | 64\* |
| attn = PV·Wo | b0:0 | QKVO:L·2048+1536 | resid:0 (→+res,LN) | 128 | 128 |
| FFN1 = X·W1 | resid:0 | FFN(L):0 | b0..3:H | 128 | 512 |
| FFN2 = H·W2 | b0..3:H | FFN(L):2048 | resid:0 (→+res,LN) | 512 | 128 |

\* attention은 **head별**(head_dim=64, key=64) → base에 head offset, 2 head 반복.
\* QKVO는 L0/L1 공유 뱅크라 **`L·2048`** layer offset. FFN은 layer별 뱅크(FFN(L)).

- **공유 뱅크(weight/bias)** → base = 앞 텐서 누적. **전용 뱅크(intermediate)** → base = 0.
- compute 중엔 `k`만 증가: weight stride-NT, act stride-MT (둘 다 feature-major). URAM/BRAM은 random-access라 stride 무관 1 read/cycle. tile/matmul 전환 시 offset·base만 스위치.

---

## 7. Operand source mux

operand가 matmul마다 다른 뱅크에서 옴 → **read mux**로 선택 (경합 없어 **arbiter 아님**).

```
act:  {resid, b0, b1, b2, b3} ─128b N:1 mux─▶ act_row     (act_sel)
wgt:  {URAM(QKVO/FFN), b1, b2} ─128b N:1 mux─▶ w_col       (w_sel)
```
| matmul | act_sel | w_sel |
|--------|---------|-------|
| Q/K/V, FFN1 | resid | URAM |
| scores | b0 (Q) | b1 (K_T) |
| context | b3 (P) | b2 (V) |
| attn | b0 (PV) | URAM(QKVO) |
| FFN2 | **b[k/128]** (H 4뱅크) | URAM(FFN) |

- select = **matmul phase**(대부분 static). H(FFN2)만 `k` 상위비트로 서브뱅크 선택.
- 128-bit 5:1 mux(~128 LUT). select 미리 알려져 register 1단이면 Fmax 안전.

---

## 8. 계층 (DRAM 생략)

```
[init 1회]  DDR ──DMA──▶ URAM (weight 적재)
[inference] URAM(weight) / BRAM(act·intermediate·bias) ──▶ PE datapath reg
             └ DRAM tier 없음. 2 level.
```
- 전량 on-chip fit(weight 1.5 Mbit ≪ URAM 18 Mbit) → inference 중 DRAM 트래픽 0.
- memory map + arbiter는 **config/DMA 경로에만** (datapath는 전용 포트).

---

## 9. pack 정책

| | 저장 | gap-20 pack | 어디서 |
|--|------|------------|--------|
| weight | dense INT4 byte `{w_b,w_a}` | `(w_b<<20)+w_a` | **DSP pre-adder** (런타임) |
| bias | raw INT32 쌍 | `(bias_b<<20)+bias_a` | **fabric adder** (preload) |

- offline은 "byte 자리 배치"(어느 두 feature가 한 column 공유)만. 실제 pack은 전부 on-chip.
