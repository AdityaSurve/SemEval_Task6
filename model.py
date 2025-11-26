import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from utils.logger import Logger


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1_macro": f1, "accuracy": acc}


def main():
    logger = Logger()
    arrow_path = os.path.join(
        "dataset", "train", "train", "data-00000-of-00001.arrow")

    logger.announce(f"Loading dataset from {arrow_path}...")
    ds_raw = Dataset.from_file(arrow_path)
    df = ds_raw.to_pandas()

    logger.plain("Preprocessing data...")
    df['text'] = "Question: " + \
        df['question'].astype(str) + "\nAnswer: " + \
        df['interview_answer'].astype(str)

    label_list = sorted(list(df['label'].unique()))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    logger.success(f"Detected {len(label_list)} labels: {label_list}")

    df['label_id'] = df['label'].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df['label_id'],
        random_state=42
    )

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    logger.plain("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format("torch", columns=keep_cols)
    val_ds.set_format("torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="clarity_model_results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.announce("Starting training...")
    trainer.train()

    logger.plain("Evaluating...")
    eval_results = trainer.evaluate()

    logger.announce("Final Validation Results:")
    for key, value in eval_results.items():
        logger.success(f"{key}: {value}")


if __name__ == "__main__":
    main()
