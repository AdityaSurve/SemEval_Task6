import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")


class Config:
    FEATURES_PATH = "dataset/train_with_features.parquet"
    MODEL_DIR = "models"
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    HIDDEN_DIM = 384
    DROPOUT = 0.3
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 2e-5  # Lower LR for transformer
    FEATURE_LR = 1e-3  # Higher LR for feature heads
    GRADIENT_ACCUMULATION = 4
    WARMUP_RATIO = 0.1
    RANDOM_STATE = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]
    N_FOLDS = 5
    USE_KFOLD = False  # Set True for k-fold training
    MIXUP_ALPHA = 0.2
    LABEL_SMOOTHING = 0.1
    FOCAL_GAMMA = 2.0
    RDROP_ALPHA = 0.5  # R-Drop regularization


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, 
            weight=self.alpha, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class HybridDataset(Dataset):
    def __init__(self, texts, features, labels, tokenizer, max_length, augment=False):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Simple text augmentation for minority classes during training
        if self.augment and self.labels[idx] == 3:  # Partial/half-answer class
            if np.random.random() < 0.3:
                words = text.split()
                if len(words) > 5:
                    # Random word dropout
                    drop_idx = np.random.choice(len(words), size=max(1, len(words)//10), replace=False)
                    words = [w for i, w in enumerate(words) if i not in drop_idx]
                    text = ' '.join(words)
        
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch, seq_len, hidden]
        # attention_mask: [batch, seq_len]
        weights = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
        weights = weights.masked_fill(attention_mask == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)  # [batch, seq_len]
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # [batch, hidden]
        return pooled


class MultiSampleDropout(nn.Module):
    """Multi-sample dropout for better regularization"""
    def __init__(self, classifier, dropout_rates=[0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        self.classifier = classifier
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, x):
        outputs = torch.stack([self.classifier(dropout(x)) for dropout in self.dropouts], dim=0)
        return outputs.mean(dim=0)


class HybridClassifier(nn.Module):
    def __init__(self, model_name, num_features, num_classes, hidden_dim=384, dropout=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.transformer_dim = self.transformer.config.hidden_size
        
        # Attention pooling for better sequence representation
        self.attention_pool = AttentionPooling(self.transformer_dim)
        
        # Layer-wise weighted combination of hidden states
        self.layer_weights = nn.Parameter(torch.ones(13) / 13)  # DeBERTa has 12 layers + embeddings
        
        # Enhanced feature MLP with residual connection
        self.feature_proj = nn.Linear(num_features, hidden_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention between text and features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Project transformer output to hidden_dim
        self.text_proj = nn.Linear(self.transformer_dim * 2, hidden_dim)  # *2 for CLS + attention pooling
        
        # Final classifier with multi-sample dropout
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        classifier = nn.Linear(hidden_dim, num_classes)
        self.classifier = MultiSampleDropout(classifier)
        
        self._init_weights()

    def _init_weights(self):
        for module in [self.feature_proj, self.feature_mlp, self.text_proj, self.pre_classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        if submodule.bias is not None:
                            nn.init.zeros_(submodule.bias)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Weighted combination of all hidden states
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
        stacked = torch.stack(hidden_states, dim=0)  # (num_layers, batch, seq, hidden)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_hidden = (stacked * weights).sum(dim=0)  # (batch, seq, hidden)
        
        # CLS token
        cls_output = weighted_hidden[:, 0]
        
        # Attention pooling
        attn_pooled = self.attention_pool(weighted_hidden, attention_mask)
        
        # Combine CLS and attention pooling
        text_repr = torch.cat([cls_output, attn_pooled], dim=-1)
        text_repr = self.text_proj(text_repr)  # (batch, hidden_dim)
        
        # Feature processing with residual
        feat_proj = self.feature_proj(features)
        feat_repr = self.feature_mlp(feat_proj) + feat_proj  # Residual connection
        
        # Cross-attention: text attends to features
        text_repr_expanded = text_repr.unsqueeze(1)  # (batch, 1, hidden)
        feat_repr_expanded = feat_repr.unsqueeze(1)  # (batch, 1, hidden)
        cross_out, _ = self.cross_attention(text_repr_expanded, feat_repr_expanded, feat_repr_expanded)
        cross_out = cross_out.squeeze(1)  # (batch, hidden)
        
        # Concatenate all representations
        combined = torch.cat([text_repr, cross_out], dim=-1)
        
        # Classification
        pre_logits = self.pre_classifier(combined)
        logits = self.classifier(pre_logits)
        
        return logits


def mixup_data(x_ids, x_mask, x_feats, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_ids.size(0)
    index = torch.randperm(batch_size).to(x_ids.device)

    # For features, we can do proper mixup
    mixed_feats = lam * x_feats + (1 - lam) * x_feats[index]
    
    # For text, we keep original (mixup on embeddings would require model changes)
    return x_ids, x_mask, mixed_feats, y, y[index], lam


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps, use_mixup=True, rdrop_alpha=0.5):
    model.train()
    total_loss = 0
    preds, trues = [], []
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        feats = batch["features"].to(device)
        labels = batch["labels"].to(device)
        
        # Mixup augmentation
        if use_mixup and np.random.random() < 0.5:
            ids, mask, feats, labels_a, labels_b, lam = mixup_data(ids, mask, feats, labels, Config.MIXUP_ALPHA)
            logits = model(ids, mask, feats)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            # R-Drop: run forward twice and minimize KL divergence
            logits1 = model(ids, mask, feats)
            logits2 = model(ids, mask, feats)
            
            ce_loss = (criterion(logits1, labels) + criterion(logits2, labels)) / 2
            
            # KL divergence between two forward passes
            p = F.log_softmax(logits1, dim=-1)
            q = F.log_softmax(logits2, dim=-1)
            kl_loss = F.kl_div(p, q.exp(), reduction='batchmean') + F.kl_div(q, p.exp(), reduction='batchmean')
            
            loss = ce_loss + rdrop_alpha * kl_loss / 2
            logits = (logits1 + logits2) / 2
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        if use_mixup and 'labels_a' in dir():
            trues.extend(labels_a.cpu().numpy())
        else:
            trues.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})
    
    return total_loss / len(dataloader), f1_score(trues, preds, average="macro")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(ids, mask, feats)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(dataloader), 
        f1_score(trues, preds, average="macro"), 
        accuracy_score(trues, preds), 
        preds, 
        trues,
        torch.cat(all_logits, dim=0)
    )


def get_class_weights(labels, power=1.5):
    """Calculate class weights with adjustable power for more aggressive balancing"""
    class_counts = np.bincount(labels)
    weights = len(labels) / (len(np.unique(labels)) * class_counts)
    weights = weights ** power  # More aggressive weighting
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels):
    """Create weighted sampler for oversampling minority classes"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def train_single_fold(train_idx, val_idx, df, features, labels, texts, tokenizer, fold=0):
    """Train a single fold"""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*50}")
    
    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    
    # Print class distribution
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Val class distribution: {np.bincount(y_val)}")
    
    train_data = HybridDataset(
        [texts[i] for i in train_idx], X_train, y_train, 
        tokenizer, Config.MAX_LENGTH, augment=True
    )
    val_data = HybridDataset(
        [texts[i] for i in val_idx], X_val, y_val, 
        tokenizer, Config.MAX_LENGTH, augment=False
    )
    
    # Use weighted sampler for training
    sampler = create_weighted_sampler(y_train)
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE)
    
    # Initialize model
    model = HybridClassifier(
        Config.MODEL_NAME, 
        num_features=X_train.shape[1], 
        num_classes=len(Config.FOUR_CLASSES), 
        hidden_dim=Config.HIDDEN_DIM, 
        dropout=Config.DROPOUT
    )
    model.to(Config.DEVICE)
    
    # Apply LoRA with higher rank
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model.transformer = get_peft_model(model.transformer, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.transformer.parameters(), "lr": Config.LR, "weight_decay": 0.01},
        {"params": list(model.attention_pool.parameters()) + 
                   list(model.feature_proj.parameters()) +
                   list(model.feature_mlp.parameters()) + 
                   list(model.cross_attention.parameters()) +
                   list(model.text_proj.parameters()) +
                   list(model.pre_classifier.parameters()) +
                   list(model.classifier.parameters()), 
         "lr": Config.FEATURE_LR, "weight_decay": 0.01}
    ])
    
    total_steps = len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * Config.WARMUP_RATIO), 
        num_training_steps=total_steps
    )
    
    # Focal loss with aggressive class weights
    weights = get_class_weights(y_train, power=1.5).to(Config.DEVICE)
    criterion = FocalLoss(alpha=weights, gamma=Config.FOCAL_GAMMA, label_smoothing=Config.LABEL_SMOOTHING)
    
    best_f1 = 0
    best_state = None
    patience = 4
    no_improve = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            Config.DEVICE, Config.GRADIENT_ACCUMULATION,
            use_mixup=True, rdrop_alpha=Config.RDROP_ALPHA
        )
        
        val_loss, val_f1, val_acc, _, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"New best F1: {best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.to(Config.DEVICE)
    
    return model, best_f1


def main():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(Config.FEATURES_PATH)
    df = df[df["label"].isin(Config.FOUR_CLASSES)].reset_index(drop=True)
    
    # Enhanced text representation
    df["text"] = (
        "Question: " + df["question"].fillna("") + 
        " [SEP] Answer: " + df["interview_answer"].fillna("")
    )
    
    # Get features
    feature_cols = [c for c in df.columns if c not in [
        "question", "interview_question", "interview_answer",
        "label", "url", "inaudible", "multiple_questions", 
        "affirmative_questions", "text"
    ]]
    
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].fillna(0))
    
    le = LabelEncoder()
    labels = le.fit_transform(df["label"])
    texts = df["text"].tolist()
    
    print(f"Dataset size: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Class distribution: {np.bincount(labels)}")
    print(f"Classes: {le.classes_}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    if Config.USE_KFOLD:
        # K-Fold cross-validation
        kfold = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        fold_f1s = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
            model, best_f1 = train_single_fold(
                train_idx, val_idx, df, features, labels, texts, tokenizer, fold
            )
            fold_f1s.append(best_f1)
            
            # Save fold model
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "label_encoder": le,
                "feature_cols": feature_cols
            }, f"{Config.MODEL_DIR}/model8_fold{fold}.pt")
        
        print(f"\n{'='*50}")
        print(f"K-Fold Results:")
        print(f"Mean F1: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")
        print(f"Fold F1s: {fold_f1s}")
        
    else:
        # Single train/val/test split
        from sklearn.model_selection import train_test_split
        
        train_idx, temp_idx = train_test_split(
            range(len(df)), test_size=0.3, stratify=labels, random_state=Config.RANDOM_STATE
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=Config.RANDOM_STATE
        )
        
        # Train model
        model, _ = train_single_fold(
            np.array(train_idx), np.array(val_idx), 
            df, features, labels, texts, tokenizer, fold=0
        )
        
        # Final evaluation on test set
        print(f"\n{'='*50}")
        print("Final Test Evaluation")
        print(f"{'='*50}")
        
        X_test = features[test_idx]
        y_test = labels[test_idx]
        
        test_data = HybridDataset(
            [texts[i] for i in test_idx], X_test, y_test,
            tokenizer, Config.MAX_LENGTH, augment=False
        )
        test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE)
        
        weights = get_class_weights(labels[train_idx], power=1.5).to(Config.DEVICE)
        criterion = FocalLoss(alpha=weights, gamma=Config.FOCAL_GAMMA, label_smoothing=Config.LABEL_SMOOTHING)
        
        test_loss, test_f1, test_acc, preds, trues, _ = evaluate(
            model, test_loader, criterion, Config.DEVICE
        )
        
        print(f"\nAccuracy: {test_acc:.4f}")
        print(f"Macro F1: {test_f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(trues, preds, target_names=le.classes_))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(trues, preds))
        
        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "label_encoder": le,
            "feature_cols": feature_cols
        }, f"{Config.MODEL_DIR}/model8_final.pt")
        
        print(f"\nModel saved to {Config.MODEL_DIR}/model8_final.pt")


if __name__ == "__main__":
    main()
