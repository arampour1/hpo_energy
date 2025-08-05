"""
Functions for data loading, model creation, evaluation and metric computation.
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
from codecarbon import EmissionsTracker
from config import HYPER_SPACE

# Lazy global dataset to support multiprocessing
DATASET = None

# Load SST2 dataset and tokenize
def load_data():
    ds = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(b): return tok(b["sentence"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds

# Build DistilBERT model for classification
def build_model(device):
    return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# F1 metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds, average="macro")}

# Train model on given hyperparams, return F1 and energy usage
def evaluate_config(lr: float, bs: int, ep: int, seed: int = 42,
                    device: torch.device = torch.device("cpu"),
                    subset: int = 2000, subset_val: int = 500):
    global DATASET
    bs = int(round(bs))
    ep = int(round(ep))
    if DATASET is None:
        DATASET = load_data()

    tracker = EmissionsTracker(project_name="DistilBERT-HPO",
                               measure_power_secs=1, save_to_file=False,
                               log_level="error")
    tracker.start()

    model = build_model(device)
    args = TrainingArguments(
        output_dir=f"/tmp/hpo-{seed}", overwrite_output_dir=True,
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        num_train_epochs=ep, learning_rate=lr,
        evaluation_strategy="epoch", logging_strategy="no", save_strategy="no",
        seed=seed, fp16=torch.cuda.is_available(), report_to=[]
    )
    trainer = Trainer(
        model, args,
        train_dataset=DATASET["train"].shuffle(seed=seed).select(range(subset)),
        eval_dataset=DATASET["validation"].shuffle(seed=seed).select(range(subset_val)),
        compute_metrics=compute_metrics
    )
    trainer.train()
    eval_res = trainer.evaluate()
    energy_kwh = tracker.stop()
    return eval_res["eval_f1"], energy_kwh
