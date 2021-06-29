import datetime
import time
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import KFold


def create_folds(data, num_splits=5, random_state=2021):
    """Create KFolds."""
    data["kfold"] = -1
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, "kfold"] = f
    return data


def seed_everything(seed_val=1325):
    """Seed everything."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def loss_fn(output, target):
    """RMSE loss function"""
    return torch.sqrt(torch.nn.MSELoss()(output, target))


def tokenize_excerpts(excerpts, tokenizer, max_len=256):
    """Tokenize all of the excerpts and map the tokens to their word IDs."""
    input_ids = []
    attention_masks = []

    for excerpt in excerpts:
        encoded_dict = tokenizer.encode_plus(
            excerpt,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
    return input_ids, attention_masks


def create_dataloaders(
    tokenizer,
    train_set,
    valid_set=None,
    max_len=256,
    train_batch_size=8,
    valid_batch_size=8,
):
    """Create Dataloadres."""
    excerpts = train_set.excerpt.values
    targets = train_set.target.values

    input_ids, attention_masks = tokenize_excerpts(excerpts, tokenizer, max_len)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    targets = torch.tensor(targets, dtype=torch.float32)

    train_dataset = TensorDataset(input_ids, attention_masks, targets)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_batch_size,
    )

    if valid_set is not None:
        excerpts = valid_set.excerpt.values
        targets = valid_set.target.values

        input_ids, attention_masks = tokenize_excerpts(excerpts, tokenizer, max_len)

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        targets = torch.tensor(targets, dtype=torch.float32)

        val_dataset = TensorDataset(input_ids, attention_masks, targets)

        validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=valid_batch_size,
        )

        return train_dataloader, validation_dataloader

    else:

        return train_dataloader, None


def oof_predictions(df, model_cfg, device, num_folds=5):
    """Calculate OOF predictions."""
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
            dropout=model_cfg["dropout"],
            summary_last_dropout = model_cfg["summary_last_dropout"],
        )
    model = model.to(device)

    oof_preds = dict()

    for fold in range(num_folds):
        model = model.to(device)
        model.load_state_dict(
            torch.load(model_cfg["weights_dir"] + f"model_{fold}.bin")
        )
        model.cuda()
        model.eval()

        val_index = df[df.kfold == fold].index.tolist()

        train_set, valid_set = df[df["kfold"] != fold], df[df["kfold"] == fold]

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"])

        train_dataloader, validation_dataloader = create_dataloaders(
            tokenizer, train_set, valid_set=valid_set, max_len=model_cfg["max_len"]
        )

        preds = []
        for batch in tqdm(validation_dataloader, desc=f"Model {fold}"):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                result = model(
                    input_ids,
                    token_type_ids=None,
                    attention_mask=input_mask,
                    labels=labels,
                    return_dict=True,
                )

            preds += result.logits.cpu().detach().numpy().tolist()

        del train_dataloader, validation_dataloader

        oof_preds[str(fold)] = {
            "val_index": val_index,
            "preds": np.array(preds).reshape([len(preds)]),
        }

    del model, tokenizer

    torch.cuda.empty_cache()

    return oof_preds


def train_fold(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    validation_dataloader,
    device,
    fold,
    model_name,
    epochs=3,
    val_step=20,
    num_folds=5,
    gradient_clipping=True,
):
    """Train one fold."""
    best_val_loss = np.Inf

    for epoch_i in range(0, epochs):
        print("")

        t0 = time.time()  # Measure staring time of epoch
        total_train_loss = 0.0  # Reset the total loss for this epoch.
        model.train()  # Put the model into training mode

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True,
            )

            if "roberta" or "xlnet" in model_name:
                loss = loss_fn(
                    result.logits, torch.reshape(b_labels, (b_labels.shape[0], 1))
                )
            else:
                loss = loss_fn(
                    result.logits,
                    b_labels,
                )

            total_train_loss += loss.item()
            loss.backward()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (
                step % val_step == val_step - 1 or step == len(train_dataloader) - 1
            ) and not step == 0:
                model.eval()
                total_eval_accuracy = 0.0
                total_eval_loss = 0.0
                for batch in validation_dataloader:
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    with torch.no_grad():
                        result = model(
                            b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True,
                        )
                    if "roberta" or "xlnet" in model_name:
                        loss = loss_fn(
                            result.logits,
                            torch.reshape(b_labels, (b_labels.shape[0], 1)),
                        )
                    else:
                        loss = loss_fn(
                            result.logits,
                            b_labels,
                        )
                    total_eval_loss += loss.item()
                    avg_val_loss = total_eval_loss / len(validation_dataloader)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), "model_" + str(fold) + ".bin")

                print(
                    "Fold {:} / {:}   Epoch {:} / {:}   Batch {:3} / {:3}   Val_Loss: {:.4f}   Best_Val_Loss: {:.4f}".format(
                        fold + 1,
                        num_folds,
                        epoch_i + 1,
                        epochs,
                        step + 1,
                        len(train_dataloader),
                        avg_val_loss,
                        best_val_loss,
                    ),
                    flush=True,
                )

        avg_train_loss = total_train_loss / len(train_dataloader)

        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Best Val Loss: {0:.4f}".format(best_val_loss))
        training_time = format_time(time.time() - t0)
        print("  Training epoch took: {:}".format(training_time), flush=True)
    return best_val_loss


def create_submission_dataloader(tokenizer, excerpts, batch_size=16):
    """Create dataloader for submission."""
    input_ids, attention_masks = tokenize_excerpts(excerpts, tokenizer)

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks, dim=0)

    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size
    )

    return prediction_dataloader


def make_predictions(model_cfg, test_csv, device, num_folds=5, batch_size=16):
    """Make predictions for one K-Fold ensemble"""
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"])
    df = pd.read_csv(test_csv)
    excerpts = df.excerpt.values
    prediction_dataloader = create_submission_dataloader(
        tokenizer, excerpts, batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["model"],
        num_labels=1,
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=model_cfg["hidden_dropout_prob"],
        attention_probs_dropout_prob=model_cfg["attention_probs_dropout_prob"],
    )

    avg_predictions = np.array([[0.0] * excerpts.shape[0]])

    for fold in range(num_folds):

        model = model.to(device)
        model.load_state_dict(
            torch.load(model_cfg["weights_dir"] + f"model_{fold}.bin")
        )
        model.cuda()
        model.eval()

        predictions = []

        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                result = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    return_dict=True,
                )
            logits = result.logits
            logits = logits.detach().cpu().numpy().reshape(len(logits)).tolist()
            predictions.extend(logits)
        avg_predictions += predictions

    avg_predictions = avg_predictions.reshape(-1)
    avg_predictions = avg_predictions / num_folds

    return avg_predictions
