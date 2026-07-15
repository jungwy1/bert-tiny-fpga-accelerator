# BERT-Tiny SST-2 Fine-tuning 정리

> 목적: `prajjwal1/bert-tiny` 인코더에 분류 head를 붙여 **SST-2 감성분석**으로 fine-tuning →
> **FP32 정확도 baseline** 확보. 이 값이 이후 **INT8 가속기가 도달해야 할 기준선**이 된다.
> 스크립트: [`python/finetune_sst2.py`](../python/finetune_sst2.py)

---

## 0. 결과 요약

| 항목 | 값 |
|------|-----|
| 태스크 | SST-2 (단일 문장, 2-class: 긍정/부정) |
| 모델 | prajjwal1/bert-tiny 인코더 + classifier(128→2) |
| **FP32 validation accuracy** | **0.8142** |
| 학습 환경 | CPU, 학습 시간 ~20–30분 |
| 저장 위치 | `python/bert-tiny-sst2/` |

- 참고 기대치: BERT-Tiny SST-2 공개 보고치 ~83%. 우리 값 81.4%는 정상 범위.
- 하이퍼파라미터 튜닝 전(lr=1e-3, 3 epoch, no warmup) = 79.7% → 튜닝 후 = **81.4%**.

---

## 1. 사용 라이브러리 & 최소 버전

| 라이브러리 | 최소 버전 | 역할 |
|------------|-----------|------|
| `torch` | ≥ 2.0 | 모델·학습 백엔드 |
| `transformers` | ≥ 4.46 | 모델/토크나이저/`Trainer`. (`processing_class` 인자, `eval_strategy` 명칭이 이 버전대부터) |
| `datasets` | ≥ 2.19 | SST-2 로드 (parquet) |
| `accelerate` | ≥ 1.1.0 | `Trainer` 실행에 필수 |
| `numpy` | ≥ 1.24 | accuracy 계산 |

- 참고: 본 작업은 torch 2.12 / transformers 5.13 / datasets 5.0 / accelerate 1.14 / numpy 2.5 에서 검증.
- `sentencepiece`는 불필요(BERT는 WordPiece + `vocab.txt` 사용).

---

## 2. 파이프라인 (스크립트 7단계)

| 단계 | 코드 | 하는 일 |
|------|------|---------|
| 1 | `load_dataset("stanfordnlp/sst2")` | SST-2 로드 (train 67,349 / val 872) |
| 2 | `BertTokenizerFast.from_pretrained` | 문장 → input_ids / attention_mask |
| 3 | `BertForSequenceClassification.from_pretrained(..., num_labels=2)` | 인코더 로드 + classifier(128→2) 부착 |
| 4 | `compute_metrics` | argmax → accuracy |
| 5 | `TrainingArguments` | 학습 설정 (아래 §4) |
| 6 | `trainer.train()` / `.evaluate()` | 학습 + validation 정확도 |
| 7 | `trainer.save_model()` | `bert-tiny-sst2/`에 저장 |

### 데이터 전처리 (2단계)
```python
def preprocess(batch):
    return tok(batch["sentence"], truncation=True, max_length=MAX_LEN)  # MAX_LEN=64
ds_tok = ds.map(preprocess, batched=True)
```
- SST-2 문장은 평균 ~10–15토큰으로 짧음. `Trainer`의 동적 패딩으로 실제 S는 64보다 훨씬 작음 → 학습이 빠른 이유.

---

## 3. 모델 로드 리포트 해석 (중요)

3단계에서 아래 리포트가 뜨는데 **모두 정상**이다. §9.3(모델 분석 문서)의 설계 스코프가 실제로 실행된 증거.

```
cls.predictions.*      | UNEXPECTED   ← MLM head
cls.seq_relationship.* | UNEXPECTED   ← NSP head
classifier.weight/bias | MISSING
```

| 상태 | 의미 | 처리 |
|------|------|------|
| **UNEXPECTED** | 체크포인트엔 있으나 우리 모델(분류)은 안 씀 = pretraining head | **버려짐** |
| **MISSING** | 우리 모델엔 필요하나 체크포인트에 없음 = 새 분류층 | **랜덤 초기화 → 학습으로 채움** |

- 인코더 weight(`bert.encoder.*`)는 리포트에 **안 뜸** = 조용히 정상 로드됨.
- 즉 "인코더 재사용 + pretraining head 폐기 + 분류층 신규 학습"이 의도대로 동작.

---

## 4. 하이퍼파라미터 (튜닝 결과)

```python
TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=3e-4,        # 1e-3은 tiny 모델엔 과함 → 낮춤
    warmup_ratio=0.1,          # 안정적 수렴
    weight_decay=0.01,         # 가벼운 정규화
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
```

| 항목 | 초기값 | 튜닝값 | 효과 |
|------|--------|--------|------|
| learning_rate | 1e-3 | **3e-4** | tiny 모델 안정화 |
| num_train_epochs | 3 | **5** | 수렴 여유 |
| warmup_ratio | — | **0.1** | 초반 발산 방지 |
| weight_decay | — | **0.01** | 과적합 억제 |
| 정확도 | 79.7% | **81.4%** | +1.7%p |

- `load_best_model_at_end=True`: 5 epoch 중 **정확도 최고 시점** 모델을 자동 저장. 마지막 epoch가 최고가 아니어도 best 보관.

---

## 5. FPGA 관점에서 꼭 기억할 점 ⭐

- **양자화 대상 = fine-tuned 모델(`bert-tiny-sst2/`), 원본 `prajjwal1/bert-tiny`가 아님.**
  full fine-tuning이므로 인코더 weight도 SST-2에 맞게 값이 바뀌었다. 원본을 양자화하면 81.4%를 재현할 수 없다.
- 양자화 범위 = **임베딩 + 인코더 2레이어 + pooler + classifier(128→2)**. pretraining head(MLM/NSP)는 애초에 로드되지 않았으므로 대상 아님.
- 파라미터 **shape**는 fine-tuning으로 바뀌지 않음(§2 표 유효). 바뀐 것은 **값**뿐.

---

> 재현: `python python/finetune_sst2.py` (기존 `bert-tiny-sst2/`는 깔끔한 재현을 위해 삭제 후 실행 권장)
