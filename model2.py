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

    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=-1
    )

    preds = []
    for t in df["text"].tolist():
        out = classifier(t, labels)
        preds.append(out["labels"][0])

    df["pred"] = preds
    print(df[["label", "pred"]].head())


if __name__ == "__main__":
    main()
