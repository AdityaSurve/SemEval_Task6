import torch
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import Config
import numpy as np

class ClarityDataset(Dataset):
    def __init__(self, df, tokenizer, le_clarity):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le_clarity = le_clarity
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        q_text = str(row['question'])
        a_text = str(row['interview_answer'])
        
        # Checking if summary exists
        summary = str(row['gpt3.5_summary']) if not pd.isna(row['gpt3.5_summary']) else ""
        
        sep = self.tokenizer.sep_token
        
        # Our Strategy: Question + Summary (if exists) + Answer
        # Summary helps model understand context better so that itll function better
        if summary and len(summary) > 10:
            combined_text = f"{q_text} {sep} {summary} {sep} {a_text}"
        else:
            combined_text = f"{q_text} {sep} {a_text}"
        
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LEN,
            return_tensors='pt'
        )
        
        c_label = self.le_clarity.transform([row['clarity_label']])[0]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_clarity': torch.tensor(c_label, dtype=torch.long)
        }

def get_dataloaders():
    print("** Downloading dataset 'ailsntua/QEvasion'...")
    dataset = load_dataset("ailsntua/QEvasion")
    
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    print(f"** Original train size: {len(train_df)} **")
    print(f"** Test size: {len(test_df)} **")
    
    # Check class distribution
    print("\n** Training set class distribution:")
    print(train_df['clarity_label'].value_counts())
    
    # Create validation split from training data (20%)
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['clarity_label']  # Keeping class distribution
    )
    
    print(f"** Split into train: {len(train_df)}, val: {len(val_df)} **")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Fit label encoder on all possible labels
    all_clarity = pd.concat([
        train_df['clarity_label'], 
        val_df['clarity_label'],
        test_df['clarity_label']
    ]).unique()
    all_clarity = [x for x in all_clarity if pd.notna(x)]
    
    le_clarity = LabelEncoder().fit(all_clarity)
    
    print(f"\n** Clarity labels: {le_clarity.classes_} **")
    
    # Calculating class weights for imbalanced classes
    train_labels = train_df['clarity_label'].values
    label_counts = pd.Series(train_labels).value_counts()
    total = len(train_labels)
    class_weights = {le_clarity.transform([label])[0]: total / count 
                     for label, count in label_counts.items()}
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(le_clarity.classes_))], 
                                       dtype=torch.float)
    
    print(f"**** Class weights: {class_weights_tensor} ****")
    
    # Creating datasets
    train_dataset = ClarityDataset(train_df, tokenizer, le_clarity)
    val_dataset = ClarityDataset(val_df, tokenizer, le_clarity)
    test_dataset = ClarityDataset(test_df, tokenizer, le_clarity)
    
    # Creating dataloaders 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  
        pin_memory=False  
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, le_clarity, class_weights_tensor
