#!/usr/bin/env python3
import os
import torch
import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import mean_squared_error

from clrp_utils import (
    seed_everything,
    create_folds,
    create_dataloaders,
    train_fold,
    oof_predictions,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
LR = 2.0e-5
WARMUP_RATIO = 0.0
EPOCHS = 3
SEED_VAL = 1325
VAL_STEP = 10
DEVICE = torch.device("cuda")
WEIGHT_DECAY = 0.01
NUM_FOLDS = 5
FOLDS_RANDOM_STATE = 1325
GRADIENT_CLIPPING = True
TRAIN_CSV = "~/datasets/commonlitreadabilityprize/train.csv"

model_cfg = {
    "model": "roberta-base",
    "weights_dir": "",
    "tokenizer": "roberta-base",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
}
print(model_cfg)

seed_everything(SEED_VAL)

tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"])

df = pd.read_csv(TRAIN_CSV)

df = create_folds(df, num_splits=NUM_FOLDS, random_state=FOLDS_RANDOM_STATE)


best_val_losses = list()

for fold in range(NUM_FOLDS):

    train_set, valid_set = df[df["kfold"] != fold], df[df["kfold"] == fold]

    train_dataloader, validation_dataloader = create_dataloaders(
        tokenizer,
        train_set,
        valid_set=valid_set,
        max_len=model_cfg["max_len"],
        train_batch_size=BATCH_SIZE,
        valid_batch_size=VAL_BATCH_SIZE,
    )

    if "bert" in model_cfg["model"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["model"],
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            hidden_dropout_prob=model_cfg["hidden_dropout_prob"],
            attention_probs_dropout_prob=model_cfg["attention_probs_dropout_prob"],
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["model"],
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            dropout=model_cfg["droput"],
            summary_last_dropout = model_cfg["summary_last_dropout"],
        )

    model = model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        weight_decay=WEIGHT_DECAY,
        eps=1e-6,
        correct_bias=False,
    )

    total_steps = len(train_dataloader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_RATIO * total_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = train_fold(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        validation_dataloader,
        DEVICE,
        fold,
        model_cfg["model"],
        epochs=EPOCHS,
        val_step=VAL_STEP,
        num_folds=NUM_FOLDS,
        gradient_clipping=GRADIENT_CLIPPING,
    )
    best_val_losses.append(best_val_loss)

    torch.cuda.empty_cache()
    del train_dataloader, validation_dataloader, model, optimizer, scheduler

print("\nBest Val Losses:")
for i, loss in enumerate(best_val_losses):
    print("Fold: {:}   Loss: {:.5f}".format(i, loss), flush=True)


oof_preds = oof_predictions(df, model_cfg, DEVICE)

oof_combined = np.zeros(len(df))
for fold in oof_preds:
    oof_combined[oof_preds[fold]["val_index"]] += oof_preds[fold]["preds"]

cv_score = np.sqrt(mean_squared_error(df.target.values, oof_combined))
print("CV score = {:.5f}".format(cv_score))
