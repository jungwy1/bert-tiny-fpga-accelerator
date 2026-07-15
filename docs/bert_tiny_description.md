# BERT-Tiny 모델 분석 (FPGA INT8 가속기용)

> 대상 체크포인트: HuggingFace [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny)
> 목표: BERT-Tiny 인코더를 FPGA에 INT8 가속기로 **구현**
> 타깃 태스크: **SST-2 감성분석 (단일 문장, 2-class 분류)** — 아래 §9 참조
> 작성 관점: 아키텍처 이해 → 연산/메모리 특성 → 가속기 설계 시사점

---

## 1. 개요 & 하이퍼파라미터

BERT-Tiny는 Turc et al., 2019 ("Well-Read Students Learn Better")에서 제안한 가장 작은 BERT 계열 모델이다.
구조 자체는 표준 BERT(인코더 전용 Transformer)와 동일하고 크기만 축소되었다.

| 기호 | 항목 | 값 |
|------|------|-----|
| L | Transformer 인코더 레이어 수 | **2** |
| H | Hidden size (d_model) | **128** |
| A | Attention head 수 | **2** |
| d_head | Head dimension (= H/A) | 64 |
| I | Intermediate(FFN) size (= 4·H) | **512** |
| V | Vocab size (WordPiece, uncased) | 30,522 |
| P | Max position embeddings | 512 |
| T | Token type(segment) 수 | 2 |
| — | 활성화 함수 | GELU |
| — | 정규화 | LayerNorm (post-LN 방식) |

- config 원본: `hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512, vocab_size=30522, hidden_act=gelu`
- 체크포인트 포맷: `pytorch_model.bin` (PyTorch pickle), vocab은 `vocab.txt`

---

## 2. 파라미터 상세 (텐서별 shape & 개수)

MatMul weight는 PyTorch `nn.Linear` 규약상 `[out, in]`으로 저장되지만, 아래 표는 수학적 방향 `[in→out]`으로 표기한다.
(bias 포함)

### 2.1 Embedding 블록 (레이어당 아님, 1회)

| 텐서 | shape | 파라미터 |
|------|-------|----------|
| word_embeddings | [30522, 128] | 3,906,816 |
| position_embeddings | [512, 128] | 65,536 |
| token_type_embeddings | [2, 128] | 256 |
| embeddings.LayerNorm (γ, β) | [128]×2 | 256 |
| **소계** | | **3,972,864** |

### 2.2 인코더 레이어 1개 (×2 반복)

| 서브블록 | 텐서 | shape (in→out) | 파라미터 |
|----------|------|----------------|----------|
| Attention | query (W+b) | 128→128 | 16,512 |
| | key (W+b) | 128→128 | 16,512 |
| | value (W+b) | 128→128 | 16,512 |
| | output.dense (W+b) | 128→128 | 16,512 |
| | attention LayerNorm | [128]×2 | 256 |
| FFN | intermediate.dense (W+b) | 128→512 | 66,048 |
| | output.dense (W+b) | 512→128 | 65,664 |
| | output LayerNorm | [128]×2 | 256 |
| **레이어 소계** | | | **198,272** |

- 2개 레이어 합계: **396,544**

### 2.3 Pooler (분류 태스크에서 [CLS] 처리용, 선택)

| 텐서 | shape | 파라미터 |
|------|-------|----------|
| pooler.dense (W+b) + tanh | 128→128 | 16,512 |

### 2.4 총계

| 구성 | 파라미터 |
|------|----------|
| Embeddings | 3,972,864 |
| Encoder ×2 | 396,544 |
| Pooler | 16,512 |
| **GRAND TOTAL** | **4,385,920 (≈ 4.39M)** |

> ⭐ **핵심 인사이트 1** — 전체의 **89%(3.9M)가 word embedding 테이블**이다.
> 이건 곱셈이 아니라 **인덱스 lookup(메모리 read)**이다.
> 실제 GEMM weight(연산 대상)는 **396K(9%)뿐**. 가속기는 compute보다 **메모리/데이터무브** 설계가 중요해진다.

---

## 3. 데이터 흐름 (추론 1회, 시퀀스 길이 S)

```
input_ids [S]  ─┐
position   [S]  ├─► Embedding lookup & 합산 ─► LayerNorm ─► X0 [S,128]
segment    [S]  ─┘

for layer in {0,1}:
    # ── Multi-Head Self-Attention (2 heads, d_head=64) ──
    Q = X·Wq + bq        [S,128]
    K = X·Wk + bk        [S,128]
    V = X·Wv + bv        [S,128]
    (head별로 128을 64+64로 split)
    scores = Q·Kᵀ / √64  [S,S]  per head
    P      = softmax(scores + mask)   [S,S]
    ctx    = P·V          [S,64] per head → concat → [S,128]
    A_out  = ctx·Wo + bo  [S,128]
    X = LayerNorm(X + A_out)          # residual + post-LN

    # ── Feed-Forward Network ──
    h = GELU(X·W1 + b1)   [S,512]
    F = h·W2 + b2         [S,128]
    X = LayerNorm(X + F)             # residual + post-LN

# 출력: X [S,128]  (task head는 별도)
```

- **연산 종류**: GEMM(대부분), Softmax, LayerNorm, GELU, residual add, scale(1/√64)
- **Attention scaling**: 1/√d_head = 1/8. 상수이므로 `Wq`에 접거나 QKᵀ 뒤 requant scale에 흡수 가능 → 런타임 비용 0
- **정규화 위치**: post-LayerNorm (서브블록 출력 + residual 후 LN) — BERT 원본 방식
- **LayerNorm 개수**: 총 5개 = 임베딩 직후 1 + 인코더 레이어당 2 × 2레이어. 유닛은 1개만 만들어 γ/β 바꿔가며 재사용
- **mask 종류**: BERT는 causal mask **없음**(양방향). `scores + mask`의 mask는 **padding mask**(=[PAD] 무시용)뿐. **batch=1 + 실제 길이 S만 처리**하면 [PAD]가 없어 mask 로직 **생략 가능**
- **head 처리**: QKV/output projection은 head 공유 → 단일 GEMM. score(QKᵀ)·context(P·V)는 **head별 독립**(block-diagonal) → batched GEMM(head 수만큼 반복). concat은 context 계산 직후·output proj 직전에서 발생하며 HW에선 주소 배치로 처리(연산 0)

---

## 4. 연산량 (MAC / FLOPs)

레이어당 MAC = `4·S·H² + 2·S²·H`(attention) + `2·S·H·I`(FFN)

| 시퀀스 길이 S | 총 MACs (2 레이어) | ≈ FLOPs |
|---------------|--------------------|---------|
| 32 | 13.1 M | 26 MFLOPs |
| 64 | 27.3 M | 55 MFLOPs |
| 128 | 58.7 M | 117 MFLOPs |

- Embedding lookup은 곱셈이 0 (테이블 read만).
- S²항(attention score/context)은 S가 커질수록 비중이 커지지만, S≤128에서는 **FFN·QKV projection이 지배적**.
- 절대 연산량이 매우 작음 → 중급 FPGA에서도 실시간 처리 여유. 병목은 연산이 아니라 **비선형 함수 처리와 데이터 스케줄링**이 될 가능성이 높다.

---

## 5. 메모리 풋프린트

### 5.1 Weight 저장

| 정밀도 | Embedding 테이블 | Encoder weight | 합계 |
|--------|------------------|----------------|------|
| FP32 | 15.9 MB | 1.59 MB | ~17.5 MB |
| INT8 | 3.97 MB | 0.40 MB | ~4.4 MB |

- INT8로 가면 encoder weight는 **400KB** → 웬만한 FPGA의 **on-chip BRAM/URAM에 상주 가능**.
- Embedding 테이블(INT8 4MB)은 on-chip에 다 올리기엔 큼 → **외부 DRAM에 두고 필요한 토큰 행만 lookup**하는 구조가 자연스럽다.

### 5.2 Activation (중간값, S=128 기준)

- X, Q, K, V 등 [128,128] activation: FP32 64KB / INT8 16KB 수준 → 작다.
- Attention score P: [S,S] = [128,128] → head당 16KB(FP32). 2 head.
- 전체 activation 워킹셋이 작아 **레이어 단위 온칩 상주**가 가능.

---

## 6. 비선형 연산 — FPGA 구현 노트

INT8 GEMM은 DSP로 쉽지만, 아래 3개가 정확도/설계의 실질적 난관이다.

| 연산 | 위치/횟수 | FPGA 구현 고려사항 |
|------|-----------|--------------------|
| **Softmax** | attention, 레이어당 (S×S) | exp + 나눗셈. 보통 max-subtraction 후 LUT 기반 exp, 누산 후 역수. INT에서 스케일 관리 주의 |
| **LayerNorm** | 레이어당 2회 | 평균/분산(√) 필요. 온라인 mean/var 또는 2-pass. 역제곱근 근사(LUT/Newton) |
| **GELU** | FFN, 레이어당 1회 (S×512) | tanh 근사식 또는 LUT. INT8에서는 piecewise-linear LUT가 일반적 |

- 세 연산 모두 **LUT 기반 근사 + 고정소수점 스케일 관리**가 핵심.
- Softmax/LayerNorm은 정밀도에 민감 → **부분적으로 int32 누산 / fp16 처리** 후 재양자화하는 하이브리드가 흔함.

---

## 7. INT8 양자화 시사점

- **대칭 per-tensor 또는 per-channel** 양자화가 GEMM에 유리. weight는 per-channel, activation은 per-tensor가 흔한 조합.
- GEMM은 `INT8 × INT8 → INT32 누산` 후, `requantize(scale)`로 다시 INT8로.
- **민감 지점**: embedding scale, softmax 입력, LayerNorm 통계. 여기서 정확도 손실이 주로 발생하므로 이 경로만 고정밀 처리하는 전략 검토.
- 후속 작업: PyTorch에서 대표 데이터로 **PTQ(Post-Training Quantization)** → per-tensor scale 추출 → HW 매핑.

---

## 8. 가속기 설계 시사점 (요약)

1. **하나의 재사용 가능한 GEMM/systolic 엔진**으로 QKV·output·FFN을 모두 처리 (weight만 교체). 레이어가 2개뿐이라 시간 분할(time-multiplex)이 자연스럽다.
2. **Encoder weight(INT8 400KB)는 온칩 상주**, embedding 테이블(4MB)은 DRAM lookup.
3. 진짜 병목은 GEMM이 아니라 **Softmax/LayerNorm/GELU 비선형 유닛과 재양자화 파이프라인**. 여기에 LUT/근사 유닛 설계 집중.
4. Activation 워킹셋이 작아 **레이어별 온칩 처리 + 순차 스트리밍**으로 외부 대역폭 요구가 낮다.
5. 타깃 = **SST-2 감성분류(단일 문장, 2-class)** → 가속 스코프가 **임베딩 + 인코더 2레이어 + 초소형 classifier(128→2)**로 깔끔. pretraining 헤드(MLM/NSP)와 무거운 vocab 투영은 **제외**(§9).

---

## 9. 타깃 태스크 & 가속 스코프 (SST-2 감성분류)

### 9.1 태스크 정의

- **SST-2**: 영화 리뷰 문장 → 긍정(1)/부정(0), **단일 문장 2-class 분류** (GLUE 벤치마크).
- 데이터: `datasets.load_dataset("glue", "sst2")` (train ~67k, validation 872).
- 모델은 **영어 uncased(vocab 30522)** → 영어 데이터 전용. 한국어 불가(→ [UNK]).

### 9.2 분류 경로

```
[CLS] 토큰 벡터 [128]  (인코더 최종 출력의 0번 행)
  → pooler: dense(128→128) + tanh   [128]
  → classifier: [128]·[128,2] + b   [2]     ← fine-tuning으로 새로 학습
  → softmax → 긍정/부정
```

- `[CLS]` 벡터 하나만 사용 → **마지막 레이어는 [CLS] 행 출력만 필요**(단, attention 때문에 중간 레이어는 전 토큰 필요).
- classifier(`128→2`)는 pretraining 체크포인트에 **없음** → SST-2로 fine-tuning해서 붙인다.

### 9.3 체크포인트에 딸려온 pretraining 헤드 (우리는 미사용)

state_dict에는 인코더 백본 외에 사전학습용 출력 헤드가 함께 들어있다. **분류 태스크에선 전부 제외**한다.

| 텐서 그룹 | 용도 | 우리 사용 |
|-----------|------|-----------|
| `bert.pooler.dense` (128→128 + tanh) | [CLS] 요약 | ✅ (분류 경로에서 사용) |
| `cls.predictions.*` (transform + decoder 128→30522) | **MLM**(마스크 단어 예측) | ❌ 제외 (무거운 vocab 투영) |
| `cls.seq_relationship` (128→2) | **NSP**(다음 문장 예측) | ❌ 제외 |

- `cls.predictions.decoder.weight` `[30522,128]`는 `word_embeddings.weight`와 shape 동일 → 보통 **weight tying**. MLM을 안 하므로 무관.
- MLM을 했다면 decoder(`[S,128]·[128,30522]`)가 인코더 전체보다 큰 연산 폭탄이 됐을 것 → 분류 선택으로 회피.

### 9.4 태스크 확정에서 파생되는 HW 단순화

| 부품 | 단일 문장 2-class에서 | 근거 |
|------|----------------------|------|
| token_type(segment) embedding | **생략 가능** (전부 0 → word emb에 상수로 미리 합산) | 문장 1개뿐 |
| padding mask | **생략 가능** (batch=1 + 실제 길이 S만 처리) | [PAD] 없음 |
| 최종 출력 헤드 | `128→2` 초소형 | 2-class |
| vocab 투영(MLM decoder) | 없음 | 분류라서 |

### 9.5 fine-tuning & 검증 흐름

```
1. prajjwal1/bert-tiny 인코더 로드 + classifier[128→2] 부착
   (AutoModelForSequenceClassification, num_labels=2)
2. SST-2 train으로 fine-tuning (CPU 수 분~십수 분)
3. validation 정확도 측정 → FP32 baseline 확보 (bert-tiny 기대치 ~80%대 초반)
4. FP32 모델을 INT8 양자화(PTQ) → 정확도 유지율 확인
5. INT8 가중치/scale을 FPGA로 매핑
```

- **3번 FP32 정확도 = INT8 가속기의 정답 기준선.** 양자화 후 이와의 격차로 성공 판정.

---

## 부록 A. 미결정 사항 / 다음 단계

- [x] 타깃 태스크 확정 → **SST-2 감성분류 (단일 문장, 2-class)**
- [ ] SST-2로 fine-tuning → **FP32 정확도 baseline** 확보
- [ ] 실제 weight 로드해서 텐서명·값 범위(min/max) 확인 → 양자화 scale 산정
- [ ] 대표 시퀀스 길이(S) 확정 (SST-2는 짧음 → S≤64 예상; 지연/자원 예산에 직결)
- [ ] PTQ(INT8) 정확도 측정 (FP32 대비 유지율)
- [ ] 오프라인 전처리 정의: fused QKV concat, 1/√d 접기, [out,in]↔[in,out] 레이아웃 정렬
- [ ] 타깃 FPGA 보드/자원(DSP, BRAM/URAM) 확정
```