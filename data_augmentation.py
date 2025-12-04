import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
import pyarrow.feather as feather
import os

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]
TARGET_PER_CLASS = 1500


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word and '_' not in lemma.name():
                synonyms.add(lemma.name())
    return list(synonyms)


def synonym_replacement(text, n=3):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([w for w in words if len(w) > 4]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word.lower())
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            new_words = [synonym if word.lower() == random_word.lower() else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)


def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) <= 1:
        return text
    new_words = [word for word in words if random.random() > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)


def random_swap(text, n=2):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            break
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return ' '.join(new_words)


def random_insertion(text, n=2):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        candidates = [w for w in words if len(w) > 4]
        if not candidates:
            candidates = words
        random_word = random.choice(candidates)
        synonyms = get_synonyms(random_word.lower())
        if synonyms:
            synonym = random.choice(synonyms)
            insert_idx = random.randint(0, len(new_words))
            new_words.insert(insert_idx, synonym)
    return ' '.join(new_words)


def augment_text(text, technique='synonym'):
    if technique == 'synonym':
        return synonym_replacement(text, n=3)
    elif technique == 'deletion':
        return random_deletion(text, p=0.1)
    elif technique == 'swap':
        return random_swap(text, n=2)
    elif technique == 'insertion':
        return random_insertion(text, n=2)
    return text


def load_raw_data():
    df = pd.read_parquet("dataset/train_with_features.parquet")
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()[:10]}...")
    
    print(f"\nAll labels:")
    print(df["label"].value_counts())
    
    df = df[df["label"].isin(FOUR_CLASSES)].reset_index(drop=True)
    print(f"\nAfter filtering to 4 classes: {len(df)} samples")
    
    return df


def main():
    print("="*60)
    print("DATA AUGMENTATION (RAW DATA)")
    print("="*60)
    
    df = load_raw_data()
    print(f"\nClass distribution:")
    print(df["label"].value_counts())
    
    techniques = ['synonym', 'deletion', 'swap', 'insertion']
    augmented_rows = []
    
    for label in FOUR_CLASSES:
        subset = df[df["label"] == label]
        current_count = len(subset)
        needed = TARGET_PER_CLASS - current_count
        
        print(f"\n{label}: {current_count} samples, need {max(0, needed)} more")
        
        if needed <= 0:
            print(f"  Skipping (already have enough)")
            continue
        
        aug_count = 0
        attempts = 0
        max_attempts = needed * 3
        
        while aug_count < needed and attempts < max_attempts:
            attempts += 1
            row = subset.sample(1).iloc[0]
            technique = random.choice(techniques)
            
            aug_answer = augment_text(str(row["interview_answer"]), technique)
            
            if aug_answer != row["interview_answer"] and len(aug_answer) > 50:
                new_row = {
                    "question": row["question"],
                    "interview_answer": aug_answer,
                    "label": row["label"],
                    "augmented": True,
                    "aug_technique": technique
                }
                if "interview_question" in row:
                    new_row["interview_question"] = row["interview_question"]
                if "multiple_questions" in row:
                    new_row["multiple_questions"] = row["multiple_questions"]
                if "affirmative_questions" in row:
                    new_row["affirmative_questions"] = row["affirmative_questions"]
                    
                augmented_rows.append(new_row)
                aug_count += 1
                
                if aug_count % 200 == 0:
                    print(f"  Generated {aug_count}/{needed}")
        
        print(f"  Final: Generated {aug_count} samples")
    
    df["augmented"] = False
    df["aug_technique"] = "original"
    
    keep_cols = ["question", "interview_answer", "label", "augmented", "aug_technique"]
    if "interview_question" in df.columns:
        keep_cols.insert(1, "interview_question")
    if "multiple_questions" in df.columns:
        keep_cols.append("multiple_questions")
    if "affirmative_questions" in df.columns:
        keep_cols.append("affirmative_questions")
    
    df_clean = df[[c for c in keep_cols if c in df.columns]].copy()
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        combined_df = pd.concat([df_clean, aug_df], ignore_index=True)
    else:
        combined_df = df_clean.copy()
    
    print("\n" + "="*60)
    print("AUGMENTATION RESULTS")
    print("="*60)
    print(f"\nOriginal (4 classes): {len(df)} samples")
    print(f"Total after augmentation: {len(combined_df)} samples")
    print(f"Added: {len(combined_df) - len(df)} samples")
    print(f"\nNew class distribution:")
    print(combined_df["label"].value_counts())
    
    os.makedirs("augmented_data", exist_ok=True)
    combined_df.to_parquet("augmented_data/train_augmented.parquet", index=False)
    combined_df.to_csv("augmented_data/train_augmented.csv", index=False)
    print(f"\nSaved to augmented_data/train_augmented.parquet")
    print(f"Saved to augmented_data/train_augmented.csv")
    
    print("\n" + "="*60)
    print("SAMPLE AUGMENTATIONS")
    print("="*60)
    
    for label in ["Partial/half-answer", "General"]:
        print(f"\n--- {label} ---")
        orig = df[df["label"] == label].iloc[0]
        print(f"Original: {str(orig['interview_answer'])[:150]}...")
        
        aug_samples = combined_df[(combined_df["label"] == label) & (combined_df["augmented"] == True)]
        if len(aug_samples) > 0:
            aug = aug_samples.iloc[0]
            print(f"Augmented ({aug['aug_technique']}): {str(aug['interview_answer'])[:150]}...")


if __name__ == "__main__":
    main()
