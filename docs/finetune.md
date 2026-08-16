# Fine-tune: BERT-Tiny / SST-2

FP32 baseline 확보 — INT8 가속기의 정확도 기준점.

## 설정

| 항목 | 값 |
|------|----|
| 모델 | `prajjwal1/bert-tiny` (L=2, H=128, heads=2) |
| 태스크 | SST-2 (2-class sentiment), train ~67k / val 872 |
| MAX_LEN | 64 (= S 상한) |
| epochs | 5 |
| batch | 64 (train) / 128 (eval) |
| lr | 3e-4, warmup_ratio 0.1, weight_decay 0.01 |
| best model | `load_best_model_at_end` (metric=accuracy) |
| **seed** | **42** (`set_seed` + `seed`/`data_seed`) — 재현 고정 |
| 출력 | `./bert-tiny-sst2/` |

## 결과

**FP32 validation accuracy = 0.8142** (best, epoch 2–3)

| epoch | 1 | 2 | 3 | 4 | 5 |
|-------|---|---|---|---|---|
| val acc | 0.8131 | **0.8142** | **0.8142** | 0.8131 | 0.8096 |

- 학습 시간 ~22.7분 (CPU).
- epoch 4–5는 overfit 경향 (loss↓ acc↓) → best(0.8142) 저장.

## 의미

- 이 **0.8142** 가 이후 fake-quant 스윕 / INT8 int_model 의 **비교 기준**.
- seed 고정으로 재현 가능 → export → compile → int_model 파이프라인이 이 값을 기준으로 일관.
