import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
import os

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

NINE_CLASSES = [
    "Explicit", "Dodging", "Implicit", "General", "Deflection",
    "Declining to answer", "Claims ignorance", "Clarification", "Partial/half-answer"
]
TARGET_PER_CLASS = 1500


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


def main():
    print("Loading dataset...")
    df = pd.read_parquet("dataset/train_with_features.parquet")
    
    df = df[df["label"].isin(NINE_CLASSES)].reset_index(drop=True)
    
    print(f"\nLoaded {len(df)} samples")
    print(f"\nOriginal class distribution:")
    print(df["label"].value_counts())
    
    techniques = ['synonym', 'deletion', 'swap', 'insertion']
    augmented_rows = []
    
    for label in NINE_CLASSES:
        subset = df[df["label"] == label]
        current_count = len(subset)
        needed = TARGET_PER_CLASS - current_count
        
        print(f"\n{label}: {current_count} samples", end="")
        
        if needed <= 0:
            print(f" - No augmentation needed (already >= {TARGET_PER_CLASS})")
            continue
        
        print(f", need {needed} more")
        
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
                    print(f"  Generated {aug_count}/{needed}")
        
        print(f"  Final: {aug_count} augmented samples")
    
    df["is_augmented"] = False
    df["aug_technique"] = "original"
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        final_df = pd.concat([df, aug_df], ignore_index=True)
    else:
        final_df = df
    
    print(f"\n" + "="*60)
    print("FINAL DATASET")
    print("="*60)
    print(f"Total samples: {len(final_df)}")
    print(f"\nClass distribution:")
    print(final_df["label"].value_counts().sort_index())
    
    os.makedirs("augmented_data", exist_ok=True)
    
    final_df.to_parquet("augmented_data/train_augmented_9class.parquet", index=False)
    final_df.to_csv("augmented_data/train_augmented_9class.csv", index=False)
    
    print(f"\nSaved to:")
    print(f"  - augmented_data/train_augmented_9class.parquet")
    print(f"  - augmented_data/train_augmented_9class.csv")
    
    print(f"\nAugmentation stats:")
    print(final_df["aug_technique"].value_counts())


if __name__ == "__main__":
    main()

