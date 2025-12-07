import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_from_disk
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from config import Config
import os

class QADataset(TorchDataset):
    def __init__(self, data, tokenizer, max_len, le_clarity):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.le_clarity = le_clarity
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        q_text = str(row['question'])
        a_text = str(row['interview_answer'])
        label = row['clarity_label']
        
        # Combine question and answer
        sep = self.tokenizer.sep_token
        
        # Strategy: Question + Summary (if exists) + Answer
        summary = str(row.get('gpt3.5_summary', ''))
        if summary and len(summary) > 10:
            text = f"{q_text} {sep} {summary} {sep} {a_text}"
        else:
            text = f"{q_text} {sep} {a_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Encode label
        label_encoded = self.le_clarity.transform([label])[0]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_encoded, dtype=torch.long)
        }

def get_dataloaders():
    # FORCE USE OF AUGMENTED DATASET
    if not os.path.exists("./augmented_balanced"):
        raise FileNotFoundError("Augmented dataset not found! Run quick_augmentation.py first.")
    
    print("** 🚀 Loading AUGMENTED dataset from ./augmented_balanced **")
    dataset = load_from_disk("./augmented_balanced")
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"** Augmented train size: {len(train_data)} **")
    print(f"** Test size: {len(test_data)} **")
    
    # Convert to DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    print("\n** Training set class distribution:")
    print(train_df['clarity_label'].value_counts())
    
    # Train-validation split
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['clarity_label']
    )
    
    print(f"** Split into train: {len(train_df)}, val: {len(val_df)} **")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Label encoding
    le_clarity = LabelEncoder()
    le_clarity.fit(train_df['clarity_label'])
    
    print(f"\n** Clarity labels: {le_clarity.classes_} **")
    
    # Calculate class weights for BALANCED dataset
    class_counts = train_df['clarity_label'].value_counts()
    total = len(train_df)
    class_weights = torch.tensor([
        total / (len(le_clarity.classes_) * class_counts[label]) 
        for label in le_clarity.classes_
    ], dtype=torch.float)
    
    print(f"**** Class weights (balanced): {class_weights} ****")
    
    # Create datasets
    train_dataset = QADataset(train_df.reset_index(drop=True), tokenizer, Config.MAX_LEN, le_clarity)
    val_dataset = QADataset(val_df.reset_index(drop=True), tokenizer, Config.MAX_LEN, le_clarity)
    test_dataset = QADataset(test_df.reset_index(drop=True), tokenizer, Config.MAX_LEN, le_clarity)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader, le_clarity, class_weights
