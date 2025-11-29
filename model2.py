# zero shot classification (no training)

from transformers import pipeline
import pandas as pd
import os
from datasets import Dataset


def main():
    arrow_path = os.path.join(
        "dataset", "train", "train", "data-00000-of-00001.arrow")
    ds = Dataset.from_file(arrow_path)
    df = ds.to_pandas()

    df["text"] = "Question: " + df["question"] + \
        "\nAnswer: " + df["interview_answer"]
    labels = sorted(df["label"].unique())

    print(f"Dataset size: {len(df)} samples")
    print(f"Labels: {labels}")

    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device="cpu",
        batch_size=8
    )

    texts = df["text"].tolist()

    print("Running zero-shot classification...")
    results = classifier(texts, labels, batch_size=8)

    preds = [r["labels"][0] for r in results]

    df["pred"] = preds

    accuracy = (df["label"] == df["pred"]).mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nSample predictions:")
    print(df[["label", "pred"]].head(10))

    print("\nPer-class breakdown:")
    for label in labels:
        subset = df[df["label"] == label]
        correct = (subset["pred"] == label).sum()
        print(f"  {label}: {correct}/{len(subset)} correct")

if __name__ == "__main__":
    main()
