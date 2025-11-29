import os
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report


def main():
    arrow_path = os.path.join(
        "dataset", "train", "train", "data-00000-of-00001.arrow")
    ds = Dataset.from_file(arrow_path)
    df = ds.to_pandas()

    df["text"] = "Question: " + df["question"] + "\nAnswer: " + df["interview_answer"]

    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    print(f"Dataset size: {len(df)}")
    print(f"Labels: {labels}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    model_name = "microsoft/deberta-v3-base" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
        )
    hf_dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    
    split = hf_dataset.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    train_ds = split["train"]
    val_ds = split["test"]

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"macro_f1": macro_f1, "accuracy": acc}

    training_args = TrainingArguments(
        output_dir="./clarity_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
        use_cpu=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nFinal evaluation:")
    results = trainer.evaluate()
    print(f"Macro F1: {results['eval_macro_f1']:.4f}")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = val_ds["label"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    trainer.save_model("./clarity_model_best")
    tokenizer.save_pretrained("./clarity_model_best")
    print("\nModel saved to ./clarity_model_best")


if __name__ == "__main__":
    main()


