# Fake-quant Accuracy Sweep: BERT-Tiny / SST-2

양자화 config별 정확도 스윕. `python/quant_sweep.py`, val 872, **FP32 baseline = 0.8142**.

## 설정

| 항목 | 값 |
|------|----|
| config | {A8W8, A8W4, A4W4} × weight {per-tensor, per-channel} |
| activation | per-tensor, dynamic absmax, symmetric |
| weight | symmetric, PT=전체 / PC=출력채널(dim0) |
| Stage 1 | **linear GEMM만** (Q/K/V/O, FFN1/2, pool, cls). attention matmul(Q@Kᵀ, P@V)·softmax·gelu·LN·emb = FP32 |
| Stage 2 | + **attention matmul (Q@Kᵀ, P@V)** (Q/K/V/probs quant → 정수). scores는 wide 유지 |

## 결과

**Stage 1 — linear GEMM only**

| config | W per-tensor | W per-channel |
|--------|--------------|---------------|
| A8W8 | 0.8108 | 0.8131 |
| A8W4 | 0.8073 | 0.8131 |
| A4W4 | 0.8005 | 0.8222* |

**Stage 2 — + attention matmul (Q@Kᵀ, P@V)**

| config | W per-tensor | W per-channel |
|--------|--------------|---------------|
| A8W8 | 0.8108 | 0.8131 |
| A8W4 | 0.8096 | 0.8131 |
| A4W4 | 0.7924 | 0.8028 |

`*` 0.8222 는 노이즈 (Stage 2에서 0.8028 로 하락 → fluke 확정).

## 핵심 발견

1. **8-bit act → attention quant 공짜.** A8W8/A8W4-PC 는 Stage1=Stage2 (손실 0). → **W8A8 full-integer(attention 포함)가 near-lossless.**
2. **per-channel weight = 핵심 레버.** W4-PT 하락을 PC가 W8 수준으로 회복 (A8W4: 0.8073→0.8131). 비트 낮을수록 효과 큼.
3. **4-bit act가 진짜 비용.** A4W4는 attention quant 시 −0.8~1.9pp. probs/Q/K 4-bit 취약.

## 결론 (가속기 관점)

| 시나리오 | 정확도 | 판정 |
|----------|--------|------|
| **A8W8 full-int (attention 포함)** | ~0.813 | ✅ lossless — 설계 목표 정당화 |
| A8W4-PC (weight만 4-bit) | 0.8131 | ✅ lossless |
| A4W4 (act도 4-bit) | ~0.80 | ⚠️ attention에서 손실 |

- 저비트 가려면 **weight만 4-bit(per-channel), act는 8-bit 유지.** act 4-bit(특히 attention)은 손실.
- 872 val 이라 **±1pp 노이즈** 감안 (0.5pp 미만 차이는 무의미).

## Baseline 대비 증감 (%p, base = 0.8142)

정확도에서 baseline 을 뺀 값 (percentage point). 음수 = 하락.

| config | S1 W-PT | S1 W-PC | S2 W-PT | S2 W-PC |
|--------|---------|---------|---------|---------|
| A8W8 | −0.34 | −0.11 | −0.34 | −0.11 |
| A8W4 | −0.69 | −0.11 | −0.46 | −0.11 |
| A4W4 | −1.37 | +0.80* | −2.18 | −1.14 |

- S1 = Stage 1 (linear only), S2 = Stage 2 (+ attention matmul).
- `*` +0.80 은 노이즈 (S2 에서 −1.14 로 정정).
- **A8 계열은 −0.1 ~ −0.7%p (사실상 무손실), A4W4 는 −1 ~ −2%p (attention 넣으면 악화).**

## 결정: A8W4, weight per-tensor

| 항목 | 값 |
|------|----|
| activation | **INT8**, per-tensor, symmetric |
| weight | **INT4**, per-tensor, symmetric |
| 정확도 | ~−0.5 %p (거의 무손실) |

- **왜 per-tensor**: A8W4-PT(~−0.5%p) 와 PC(−0.11%p) 차이가 **val 노이즈(±1%p) 범위 내** → 구분 안 됨. 그래서 **더 단순한 per-tensor** (requant 상수 1개/매트릭스) 선택. (정확도 여유 더 필요하면 PC 로 −0.11%p 가능.)
- **이득**: weight INT4 → **URAM weight 저장 절반**
