# BERT-Tiny FPGA INT8 Accelerator

BERT-Tiny 모델을 **FPGA에 INT8 가속기로 구현**하는 프로젝트.
모델 구조 분석 → fine-tuning(FP32 baseline) → INT8 양자화 → FPGA 매핑 순서로 진행한다.

- **Target Model**: HuggingFace [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) (L=2, H=128, A=2, ≈4.4M params)
- **Target Task**: SST-2 감성분석 (단일 문장, 2-class classification)
- **Precision**: INT8 quantization
- **Target Board**: AMD Xilinx Kria KV260 (Zynq UltraScale+ MPSoC)
- **Acceleration Scope**: Embedding + Encoder ×2 + Pooler + Classifier(128→2).

---

## 진행상황

| 단계 | 상태 | 결과 / 비고 |
|------|------|-------------|
| 모델 구조 분석 (architecture, param, dataflow) | ✅ 완료 | [`docs/bert_tiny_description.md`](docs/bert_tiny_description.md) |
| 타깃 태스크 확정 | ✅ 완료 | SST-2 (단일 문장, 2-class)  |
| SST-2 fine-tuning → FP32 baseline | ✅ 완료 | **validation accuracy = 0.8142** |
| INT8 양자화 (PTQ) | ⬜ 예정 | FP32 대비 accuracy 유지율 측정 |
| 가속기 설계 (GEMM engine, 비선형 유닛) | ⬜ 예정 | |
| FPGA 매핑 & 검증 | ⬜ 예정 | weight/scale HW 반영 |

---

## 문서

| 문서 | 내용 |
|------|------|
| [`docs/bert_tiny_description.md`](docs/bert_tiny_description.md) | 모델 구조 상세 — hyperparameter, 텐서별 param, dataflow, 연산량/메모리, 비선형 연산, INT8·가속기 설계 시사점 |
| [`docs/finetune.md`](docs/finetune.md) | fine-tuning 과정 — 라이브러리, 파이프라인, 하이퍼파라미터, FP32 baseline |

---

## 현재 결과

- **FP32 baseline (SST-2 validation accuracy): 0.8142**
- 참고 기대치: BERT-Tiny SST-2 공개 보고치 ~83% → 정상 범위.
- 이 값이 이후 **INT8 가속기가 도달해야 할 기준선**이 된다.

---

## 프로젝트 구조

```
BERT_Tiny/
├── README.md                       # 프로젝트 개요 & 진행상황 (이 문서)
├── docs/
│   ├── bert_tiny_description.md     # 모델 구조 분석
│   └── finetune.md                 # fine-tuning 정리
└── python/
    ├── finetune_sst2.py            # SST-2 fine-tuning 스크립트
    └── inspect_model.py            # state_dict 탐색용
```
