"""
Shared data utilities: tokenizer, SST-2 dataset, and a model-agnostic eval loop.

Every model variant in this project (float reference / fake-quant / integer golden)
exposes the same interface -- takes input_ids [S] and returns logits [2] -- so these
helpers work unchanged across all of them.

Tokenization runs on the host (CPU) and is NOT part of the accelerator: the model
boundary starts at input_ids.
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset

MODEL_DIR = "./bert-tiny-sst2"     # fine-tuned model (from finetune_sst2.py)
MAX_LEN = 64                       # SST-2 sentences are short; upper bound on S
CALIB_SAMPLES = 512                # default calibration set size


# ----------------------------------------------------------------- loading
def load_tokenizer(model_dir=MODEL_DIR):
    return BertTokenizerFast.from_pretrained(model_dir)


def load_hf(model_dir=MODEL_DIR):
    """Reference HuggingFace model — used only to verify our reimplementation."""
    return BertForSequenceClassification.from_pretrained(model_dir).eval()


def load_state_dict(model_dir=MODEL_DIR):
    """Fine-tuned FP32 weights that every model variant is built from."""
    return load_hf(model_dir).state_dict()


def load_sst2():
    """Returns (train, validation)."""
    ds = load_dataset("stanfordnlp/sst2")
    return ds["train"], ds["validation"]


# ---------------------------------------------------------------- encoding
def encode(tok, sentence, max_len=MAX_LEN):
    """sentence -> input_ids [S].  No batch dim: the models are batch-free (batch=1),
    matching how the FPGA processes one sentence at a time."""
    return tok(sentence, truncation=True, max_length=max_len,
               return_tensors="pt")["input_ids"][0]


# ------------------------------------------------------------ eval / calib
@torch.no_grad()
def accuracy(model, tok, dataset, limit=None):
    """Model-agnostic accuracy. `model` is any callable: input_ids [S] -> logits [2]."""
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    correct = 0
    for ex in dataset:
        logits = model(encode(tok, ex["sentence"], MAX_LEN))
        correct += int(logits.argmax().item() == ex["label"])
    return correct / len(dataset)


@torch.no_grad()
def run_calibration(model, tok, dataset, n=CALIB_SAMPLES):
    """Push n samples through the model to drive observers. Outputs are discarded.
    Uses the TRAIN split (never validation — calibrating on eval data would leak)."""
    for ex in dataset.select(range(min(n, len(dataset)))):
        model(encode(tok, ex["sentence"], MAX_LEN))


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    tok = load_tokenizer()
    train, val = load_sst2()
    print(f"train / validation : {len(train)} / {len(val)}")
    ids = encode(tok, "the world best")
    print(f"encode sample      : {ids.tolist()}  shape={tuple(ids.shape)}")
    print(f"decoded            : {tok.convert_ids_to_tokens(ids.tolist())}")
