"""
Model 6 Hybrid: DeBERTa + Engineered Features
==============================================
Combines transformer embeddings with linguistic features for better performance.

Usage:
    python model6_hybrid.py
    python model6_hybrid.py --epochs 5 --batch_size 8
    python model6_hybrid.py --four_classes  # Only use 4 main labels
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup
)

# ============================================
# CONFIG
# ============================================
class Config:
    # Data
    FEATURES_PATH = "dataset/train_with_features.parquet"
    MODEL_DIR = "models"
    
    # Model
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 8
    EPOCHS = 4
    LR = 2e-5
    FEATURE_LR = 1e-3  # Higher LR for feature layers
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # Labels
    FOUR_CLASSES = ['Dodging', 'Explicit', 'General', 'Partial/half-answer']
    
    RANDOM_STATE = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# DATASET
# ============================================
class HybridDataset(Dataset):
    """Dataset combining text and engineered features."""
    
    def __init__(self, texts, features, labels, tokenizer, max_length):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================
# MODEL
# ============================================
class HybridClassifier(nn.Module):
    """
    Hybrid model combining DeBERTa embeddings with engineered features.
    
    Architecture:
    - DeBERTa: text -> [CLS] embedding (768 dim)
    - Features: engineered features -> MLP (hidden_dim)
    - Combine: concat -> MLP -> output classes
    """
    
    def __init__(self, model_name, num_features, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer_dim = self.transformer.config.hidden_size  # 768
        
        # Feature processing MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Combined classifier
        combined_dim = self.transformer_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.feature_mlp, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask, features):
        # Get transformer output
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding
        cls_embedding = transformer_output.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # Process features
        feature_embedding = self.feature_mlp(features)  # (batch, hidden_dim)
        
        # Concatenate and classify
        combined = torch.cat([cls_embedding, feature_embedding], dim=1)
        logits = self.classifier(combined)
        
        return logits


# ============================================
# TRAINING
# ============================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask, features)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, predictions, average='macro')
    acc = accuracy_score(true_labels, predictions)
    
    return avg_loss, f1, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, predictions, average='macro')
    acc = accuracy_score(true_labels, predictions)
    
    return avg_loss, f1, acc, predictions, true_labels


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Train hybrid DeBERTa + Features model')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LR)
    parser.add_argument('--four_classes', action='store_true', 
                        help='Use only 4 main classes (Dodging, Explicit, General, Partial)')
    parser.add_argument('--model_name', type=str, default=Config.MODEL_NAME)
    args = parser.parse_args()
    
    print("=" * 60)
    print("HYBRID MODEL: DeBERTa + Engineered Features")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {args.model_name}")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Load data with features
    print("\n[1/5] Loading data...")
    df = pd.read_parquet(Config.FEATURES_PATH)
    print(f"Loaded {len(df)} samples")
    
    # Filter to 4 classes if requested
    if args.four_classes:
        print(f"Filtering to 4 main classes: {Config.FOUR_CLASSES}")
        df = df[df['label'].isin(Config.FOUR_CLASSES)].reset_index(drop=True)
        print(f"Filtered to {len(df)} samples")
    
    # Prepare text (Question + Answer)
    df['text'] = "Question: " + df['question'] + " [SEP] Answer: " + df['interview_answer']
    
    # Feature columns
    feature_cols = [
        'qa_similarity', 'topic_shift_score', 'keyword_overlap', 'entity_overlap',
        'first_sentence_similarity', 'answer_length_tokens', 'answer_length_chars',
        'answer_to_question_len_ratio', 'num_sentences', 'hedge_score', 'filler_score',
        'vague_word_count', 'modal_verb_count', 'num_numbers', 'num_named_entities',
        'specificity_score', 'concreteness_score', 'sentiment_compound',
        'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
        'emotion_confidence', 'pos_ratio_verbs', 'pos_ratio_nouns',
        'pos_ratio_pronouns', 'num_clauses', 'starts_with_thanks', 'pivot_score',
        'negation_count', 'deflection_score', 'ttr', 'entropy_score'
    ]
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Using {len(feature_cols)} features")
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].fillna(0))
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df['label'])
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {le.classes_}")
    print(f"Distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    
    # Split data
    print("\n[2/5] Splitting data...")
    texts = df['text'].tolist()
    
    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        range(len(df)), labels, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=labels
    )
    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx, y_temp, test_size=0.5, random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train: {len(X_train_idx)}, Val: {len(X_val_idx)}, Test: {len(X_test_idx)}")
    
    # Create datasets
    print("\n[3/5] Creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = HybridDataset(
        [texts[i] for i in X_train_idx],
        features[X_train_idx],
        y_train,
        tokenizer, Config.MAX_LENGTH
    )
    val_dataset = HybridDataset(
        [texts[i] for i in X_val_idx],
        features[X_val_idx],
        y_val,
        tokenizer, Config.MAX_LENGTH
    )
    test_dataset = HybridDataset(
        [texts[i] for i in X_test_idx],
        features[X_test_idx],
        y_test,
        tokenizer, Config.MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    print("\n[4/5] Initializing model...")
    model = HybridClassifier(
        args.model_name,
        num_features=len(feature_cols),
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer with different LRs for transformer vs feature layers
    transformer_params = list(model.transformer.parameters())
    feature_params = list(model.feature_mlp.parameters()) + list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': transformer_params, 'lr': args.lr},
        {'params': feature_params, 'lr': Config.FEATURE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Loss with class weights
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(Config.DEVICE)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    print("\n[5/5] Training...")
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, Config.DEVICE
        )
        print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
        
        val_loss, val_f1, val_acc, _, _ = evaluate(
            model, val_loader, criterion, Config.DEVICE
        )
        print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"  ★ New best model! Val F1: {val_f1:.4f}")
    
    # Load best model and evaluate on test
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    model.load_state_dict(best_model_state)
    test_loss, test_f1, test_acc, test_preds, test_true = evaluate(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=le.classes_))
    
    # Save model
    model_path = f"{Config.MODEL_DIR}/model6_hybrid_best.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'label_encoder': le,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'config': {
            'model_name': args.model_name,
            'num_features': len(feature_cols),
            'num_classes': num_classes,
            'hidden_dim': Config.HIDDEN_DIM,
            'dropout': Config.DROPOUT
        },
        'metrics': {
            'test_f1': test_f1,
            'test_acc': test_acc,
            'best_val_f1': best_val_f1
        }
    }, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
