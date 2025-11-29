import os
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5


def load_data():
    """Load the CLARITY dataset."""
    arrow_path = os.path.join(
        "dataset", "train", "train", "data-00000-of-00001.arrow")
    ds = Dataset.from_file(arrow_path)
    df = ds.to_pandas()
    df["text"] = (
        "Question: " + df["question"] +
        " [SEP] Answer: " + df["interview_answer"]
    )
    return df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("⚠️  WARNING: Running on CPU will be very slow. Use Google Colab with GPU!")
    df = load_data()
    print(f"Dataset size: {len(df)}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)
    print(f"\nLabels ({len(labels)}): {labels}")
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
    hf_df = df[["text", "label_id"]].rename(columns={"label_id": "labels"})
    dataset = Dataset.from_pandas(hf_df, preserve_index=False)
    split1 = dataset.train_test_split(
        test_size=0.2, seed=42, stratify_by_column="labels")
    split2 = split1["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="labels")
    datasets = DatasetDict({
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    })
    tokenized = datasets.map(tokenize, batched=True, remove_columns=["text"])
    print(f"\nDataset splits:")
    print(f"  Train: {len(tokenized['train'])}")
    print(f"  Val:   {len(tokenized['validation'])}")
    print(f"  Test:  {len(tokenized['test'])}")
    train_labels = tokenized["train"]["labels"]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(labels)),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"\nClass weights: {class_weights.tolist()}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds),
        }
    training_args = TrainingArguments(
        output_dir="./clarity_deberta",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        seed=42,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    trainer.train()
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_results = trainer.evaluate(tokenized["test"])
    print(f"\nTest Macro F1: {test_results['eval_macro_f1']:.4f}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    predictions = trainer.predict(tokenized["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = tokenized["test"]["labels"]
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=labels, digits=4))
    trainer.save_model("./clarity_deberta_best")
    tokenizer.save_pretrained("./clarity_deberta_best")
    print("\n✓ Model saved to ./clarity_deberta_best")


if __name__ == "__main__":
    main()
