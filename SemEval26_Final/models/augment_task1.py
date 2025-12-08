"""
Data Augmentation for Task 1: Clarity Classification
Augments minority classes to balance the dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

logger = Logger()

THREE_CLASSES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
TARGET_PER_CLASS = 1500

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(text, n=3):
    words = text.split()
    if len(words) < 4:
        return text
    new_words = words.copy()
    random_indices = random.sample(range(len(words)), min(n, len(words)))
    for idx in random_indices:
        word = words[idx]
        if len(word) > 3:
            synonyms = get_synonyms(word.lower())
            if synonyms:
                new_words[idx] = random.choice(synonyms)
    return " ".join(new_words)


def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) < 10:
        return text
    new_words = [w for w in words if random.random() > p]
    if len(new_words) < len(words) * 0.5:
        return text
    return " ".join(new_words) if new_words else text


def random_swap(text, n=2):
    words = text.split()
    if len(words) < 4:
        return text
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return " ".join(new_words)


def random_insertion(text, n=2):
    words = text.split()
    if len(words) < 4:
        return text
    new_words = words.copy()
    for _ in range(n):
        word = random.choice(words)
        if len(word) > 3:
            synonyms = get_synonyms(word.lower())
            if synonyms:
                insert_pos = random.randint(0, len(new_words))
                new_words.insert(insert_pos, random.choice(synonyms))
    return " ".join(new_words)


def augment_text(text, technique):
    if technique == 'synonym':
        return synonym_replacement(text, n=random.randint(2, 5))
    elif technique == 'deletion':
        return random_deletion(text, p=random.uniform(0.05, 0.15))
    elif technique == 'swap':
        return random_swap(text, n=random.randint(1, 3))
    elif technique == 'insertion':
        return random_insertion(text, n=random.randint(1, 3))
    return text


def augment():
    logger.log("TASK 1: CLARITY - DATA AUGMENTATION", "announce")
    
    raw_path = os.path.join(DATASET_PATH, "raw_train.parquet")
    if not os.path.exists(raw_path):
        logger.log(f"Raw data not found at {raw_path}", "error")
        logger.log("Please place raw_train.parquet in the dataset folder", "warning")
        logger.log("This should contain clarity_label column", "warning")
        return
    
    df = pd.read_parquet(raw_path)
    
    logger.log(f"Loaded {len(df)} samples", "success")
    
    if "clarity_label" in df.columns:
        label_col = "clarity_label"
    else:
        logger.log("No clarity_label column found!", "error")
        return
    
    df["task1_label"] = df[label_col].map({
        "Clear Reply": "Clear Reply",
        "Ambivalent": "Ambivalent", 
        "Ambiguous": "Ambivalent",
        "Clear Non-Reply": "Clear Non-Reply"
    })
    
    df = df.dropna(subset=["task1_label"]).reset_index(drop=True)
    
    if "question" not in df.columns and "interview_question" in df.columns:
        df["question"] = df["interview_question"]
    
    logger.log("Original class distribution:", "plain")
    print(df["task1_label"].value_counts())
    
    techniques = ['synonym', 'deletion', 'swap', 'insertion']
    augmented_rows = []
    
    for label in THREE_CLASSES:
        subset = df[df["task1_label"] == label]
        current_count = len(subset)
        needed = TARGET_PER_CLASS - current_count
        
        logger.log(f"{label}: {current_count} samples", "plain")
        
        if needed <= 0:
            logger.log(f"  No augmentation needed", "success")
            continue
        
        logger.log(f"  Need {needed} more", "warning")
        
        aug_count = 0
        attempts = 0
        max_attempts = needed * 5
        
        while aug_count < needed and attempts < max_attempts:
            attempts += 1
            row = subset.sample(1).iloc[0]
            technique = random.choice(techniques)
            
            original_answer = str(row["interview_answer"])
            aug_answer = augment_text(original_answer, technique)
            
            if aug_answer != original_answer and len(aug_answer) > 50:
                new_row = row.copy()
                new_row["interview_answer"] = aug_answer
                new_row["is_augmented"] = True
                new_row["aug_technique"] = technique
                augmented_rows.append(new_row)
                aug_count += 1
                
                if aug_count % 100 == 0:
                    print(f"    Generated {aug_count}/{needed}")
        
        logger.log(f"  Generated {aug_count} samples", "success")
    
    df["is_augmented"] = False
    df["aug_technique"] = "original"
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        final_df = pd.concat([df, aug_df], ignore_index=True)
    else:
        final_df = df
    
    logger.log("FINAL DATASET", "announce")
    logger.log(f"Total samples: {len(final_df)}", "success")
    print(final_df["task1_label"].value_counts().sort_index())
    
    output_path = os.path.join(DATASET_PATH, "task1_train.parquet")
    output_csv_path = os.path.join(DATASET_PATH, "task1_train.csv")
    final_df.to_parquet(output_path, index=False)
    final_df.to_csv(output_csv_path, index=False)
    logger.log(f"Saved to {output_path}", "success")
    logger.log(f"Saved to {output_csv_path}", "success")
    
    print(f"\nAugmentation stats:")
    print(final_df["aug_technique"].value_counts())


if __name__ == "__main__":
    augment()


