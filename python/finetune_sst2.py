"""
Fine-tune BERT-Tiny on SST-2 (sentiment analysis, 2-class).
Goal: obtain the FP32 accuracy baseline -> the reference target for the INT8 accelerator.

Run:     python finetune_sst2.py
Output:  trained model saved to ./bert-tiny-sst2/ + validation accuracy printed
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL = "prajjwal1/bert-tiny"
MAX_LEN = 64          # SST-2 sentences are short. Upper bound on S (ties to FPGA sequence length).
OUT_DIR = "./bert-tiny-sst2"

# 1) Load data ------------------------------------------------------------
# SST-2: train ~67k, validation 872. label 0=negative, 1=positive
# (use the modern parquet repo; the legacy "glue" script path breaks on new datasets/hf_hub)
ds = load_dataset("stanfordnlp/sst2")

# 2) Tokenizer ------------------------------------------------------------
# sentence text -> input_ids (WordPiece over vocab.txt) + attention_mask
# NOTE: bert-tiny's config has no "model_type" and ships only vocab.txt, so the
# Auto* classes can't resolve it. Use explicit Bert* classes instead.
tok = BertTokenizerFast.from_pretrained(MODEL)

def preprocess(batch):
    return tok(batch["sentence"], truncation=True, max_length=MAX_LEN)

ds_tok = ds.map(preprocess, batched=True)

# 3) Model ----------------------------------------------------------------
# Load the pretrained encoder + attach a fresh (random) [128->2] classifier head.
# The "classifier weights not initialized" warning is expected -- that's the head we train.
model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# 4) Accuracy metric ------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# 5) Training configuration -----------------------------------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=3e-4,           # tuned down; 1e-3 was a bit high for this tiny model
    warmup_ratio=0.1,             # 10% warmup for a more stable start
    weight_decay=0.01,            # light regularization
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    processing_class=tok,   # newer transformers renamed `tokenizer` -> `processing_class`
    compute_metrics=compute_metrics,
)

# 6) Train & evaluate -----------------------------------------------------
trainer.train()
metrics = trainer.evaluate()
print("\n=== FP32 baseline ===")
print(f"validation accuracy: {metrics['eval_accuracy']:.4f}")

# 7) Save -----------------------------------------------------------------
trainer.save_model(OUT_DIR)   # config + weights + classifier head
tok.save_pretrained(OUT_DIR)
print(f"\nModel saved to: {OUT_DIR}")
