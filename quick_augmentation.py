from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import random

def quick_augment(dataset):
    """
    Simple oversampling - duplicate minority class samples
    """
    df = pd.DataFrame(dataset['train'])
    
    print("📊 Original distribution:")
    print(df['clarity_label'].value_counts())
    
    # Target: balance all classes to ~1800 samples each
    target = 1800
    
    augmented_data = []
    
    for label in df['clarity_label'].unique():
        label_samples = df[df['clarity_label'] == label]
        current = len(label_samples)
        
        print(f"\n{label}: {current} → {target}")
        
        # Add all original samples
        augmented_data.extend(label_samples.to_dict('records'))
        
        # Duplicate to reach target
        if current < target:
            needed = target - current
            # Randomly sample with replacement
            duplicates = label_samples.sample(n=needed, replace=True)
            augmented_data.extend(duplicates.to_dict('records'))
    
    augmented_df = pd.DataFrame(augmented_data)
    
    print("\n✅ Augmented distribution:")
    print(augmented_df['clarity_label'].value_counts())
    
    # Shuffle
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    augmented_dataset = Dataset.from_pandas(augmented_df)
    
    return DatasetDict({
        'train': augmented_dataset,
        'test': dataset['test']
    })

if __name__ == "__main__":
    dataset = load_dataset("ailsntua/QEvasion")
    augmented = quick_augment(dataset)
    augmented.save_to_disk("./augmented_balanced")
    print("\n✅ Saved to ./augmented_balanced")
