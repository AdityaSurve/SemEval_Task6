import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False


class Config:
    FEATURES_PATH = "dataset/train_with_features.parquet"
    MODEL_DIR = "models"
    
    FOUR_CLASSES = ['Dodging', 'Explicit', 'General', 'Partial/half-answer']
    
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 512
    HIDDEN_DIM = 512
    DROPOUT = 0.2
    
    BATCH_SIZE = 4
    EPOCHS = 6
    LR = 1e-5
    FEATURE_LR = 5e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    GRADIENT_ACCUMULATION = 2
    
    RANDOM_STATE = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HybridDataset(Dataset):
    def __init__(self, texts, features, labels, tokenizer, max_length):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
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


class HybridClassifier(nn.Module):
    def __init__(self, model_name, num_features, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer_dim = self.transformer.config.hidden_size  # 1024 for large
        
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
        for layer in self.transformer.encoder.layer[:12]:
            for param in layer.parameters():
                param.requires_grad = False
        
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
        
        self.attention = nn.Sequential(
            nn.Linear(self.transformer_dim, 1),
            nn.Softmax(dim=1)
        )
        
        combined_dim = self.transformer_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, features):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output.last_hidden_state
        
        attn_weights = self.attention(hidden_states)
        pooled = torch.sum(hidden_states * attn_weights, dim=1)

        feature_emb = self.feature_mlp(features)
        combined = torch.cat([pooled, feature_emb], dim=1)
        return self.classifier(combined)
    
    def get_embeddings(self, input_ids, attention_mask, features):
        with torch.no_grad():
            output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = output.last_hidden_state
            attn_weights = self.attention(hidden_states)
            pooled = torch.sum(hidden_states * attn_weights, dim=1)
            feature_emb = self.feature_mlp(features)
            return torch.cat([pooled, feature_emb], dim=1)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask, features)
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    return total_loss / len(dataloader), f1_score(true_labels, predictions, average='macro')


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            
            total_loss += loss.item()
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (total_loss / len(dataloader), 
            f1_score(true_labels, predictions, average='macro'),
            accuracy_score(true_labels, predictions),
            predictions, true_labels, np.array(all_probs))


def get_transformer_probs(model, dataloader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            logits = model(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def train_traditional_models(X_train, y_train, X_val, y_val):
    models = {}
    
    print("\n" + "="*50)
    print("Training Traditional ML Models")
    print("="*50)
    
    if HAS_XGB:
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=Config.RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pred = xgb_model.predict(X_val)
        print(f"  XGBoost Val F1: {f1_score(y_val, val_pred, average='macro'):.4f}")
        models['xgboost'] = xgb_model
    
    if HAS_LGB:
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=Config.RANDOM_STATE,
            class_weight='balanced',
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        val_pred = lgb_model.predict(X_val)
        print(f"  LightGBM Val F1: {f1_score(y_val, val_pred, average='macro'):.4f}")
        models['lightgbm'] = lgb_model
    
    print("\nTraining RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=Config.RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    val_pred = rf_model.predict(X_val)
    print(f"  RandomForest Val F1: {f1_score(y_val, val_pred, average='macro'):.4f}")
    models['random_forest'] = rf_model
    
    return models

def ensemble_predict(transformer_probs, ml_models, X_features, weights=None):
    if weights is None:
        n_ml = len(ml_models)
        weights = {'transformer': 0.5}
        ml_weight = 0.5 / n_ml
        for name in ml_models:
            weights[name] = ml_weight
    
    ensemble_probs = transformer_probs * weights['transformer']
    
    for name, model in ml_models.items():
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_features)
        else:
            probs = model.predict_proba(X_features)
        ensemble_probs += probs * weights.get(name, 0.1)
    
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
    
    return np.argmax(ensemble_probs, axis=1), ensemble_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LR)
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENSEMBLE MODEL: DeBERTa-large + XGBoost + LightGBM + RF")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Transformer: {Config.MODEL_NAME}")
    print(f"Classes: {Config.FOUR_CLASSES}")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    print("\n[1/6] Loading data...")
    df = pd.read_parquet(Config.FEATURES_PATH)
    
    df = df[df['label'].isin(Config.FOUR_CLASSES)].reset_index(drop=True)
    print(f"Samples: {len(df)} (4 classes only)")
    
    df['text'] = "Question: " + df['question'] + " [SEP] Answer: " + df['interview_answer']
    
    feature_cols = [c for c in df.columns if c not in 
                    ['question', 'interview_question', 'interview_answer', 
                     'label', 'url', 'inaudible', 'multiple_questions', 
                     'affirmative_questions', 'text']]
    print(f"Features: {len(feature_cols)}")
    
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].fillna(0))
    
    le = LabelEncoder()
    le.fit(Config.FOUR_CLASSES)
    labels = le.transform(df['label'])
    print(f"Label distribution: {dict(zip(le.classes_, np.bincount(labels)))}")
    
    print("\n[2/6] Splitting data...")
    texts = df['text'].tolist()
    
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        range(len(df)), labels, test_size=0.3, 
        random_state=Config.RANDOM_STATE, stratify=labels
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5,
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    X_train_feat = features[train_idx]
    X_val_feat = features[val_idx]
    X_test_feat = features[test_idx]
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    print("\n[3/6] Training ML models...")
    ml_models = train_traditional_models(X_train_feat, y_train, X_val_feat, y_val)
    
    print("\n[4/6] Preparing transformer data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = HybridDataset(
        [texts[i] for i in train_idx], X_train_feat, y_train, tokenizer, Config.MAX_LENGTH
    )
    val_dataset = HybridDataset(
        [texts[i] for i in val_idx], X_val_feat, y_val, tokenizer, Config.MAX_LENGTH
    )
    test_dataset = HybridDataset(
        [texts[i] for i in test_idx], X_test_feat, y_test, tokenizer, Config.MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print("\n[5/6] Training DeBERTa-large hybrid...")
    model = HybridClassifier(
        Config.MODEL_NAME,
        num_features=len(feature_cols),
        num_classes=len(Config.FOUR_CLASSES),
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    transformer_params = [p for p in model.transformer.parameters() if p.requires_grad]
    other_params = list(model.feature_mlp.parameters()) + list(model.classifier.parameters()) + list(model.attention.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': transformer_params, 'lr': args.lr},
        {'params': other_params, 'lr': Config.FEATURE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * args.epochs // Config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * Config.WARMUP_RATIO), total_steps
    )
    
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(len(y_train) / (len(Config.FOUR_CLASSES) * class_counts), 
                                  dtype=torch.float32).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_f1 = 0
    best_state = None
    patience = 3
    no_improve = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            Config.DEVICE, Config.GRADIENT_ACCUMULATION
        )
        print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        
        val_loss, val_f1, val_acc, _, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
        print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  ★ New best! Val F1: {val_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    model.to(Config.DEVICE)
    
    print("\n[6/6] Ensemble Evaluation...")
    transformer_probs = get_transformer_probs(model, test_loader, Config.DEVICE)
    transformer_preds = np.argmax(transformer_probs, axis=1)
    transformer_f1 = f1_score(y_test, transformer_preds, average='macro')
    print(f"\nTransformer alone - F1: {transformer_f1:.4f}")
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    
    weight_configs = [
        {'transformer': 0.5, 'xgboost': 0.2, 'lightgbm': 0.2, 'random_forest': 0.1},
        {'transformer': 0.6, 'xgboost': 0.15, 'lightgbm': 0.15, 'random_forest': 0.1},
        {'transformer': 0.4, 'xgboost': 0.25, 'lightgbm': 0.25, 'random_forest': 0.1},
    ]
    
    best_ensemble_f1 = 0
    best_config = None
    best_preds = None
    
    for config in weight_configs:
        available_weights = {'transformer': config['transformer']}
        for name in ml_models:
            if name in config:
                available_weights[name] = config[name]
        
        total = sum(available_weights.values())
        available_weights = {k: v/total for k, v in available_weights.items()}
        
        preds, probs = ensemble_predict(transformer_probs, ml_models, X_test_feat, available_weights)
        f1 = f1_score(y_test, preds, average='macro')
        acc = accuracy_score(y_test, preds)
        
        print(f"\nWeights: {available_weights}")
        print(f"  F1: {f1:.4f}, Accuracy: {acc:.4f}")
        
        if f1 > best_ensemble_f1:
            best_ensemble_f1 = f1
            best_config = available_weights
            best_preds = preds
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS (Best Ensemble)")
    print("="*60)
    print(f"Best weights: {best_config}")
    print(f"\nAccuracy: {accuracy_score(y_test, best_preds):.4f}")
    print(f"Macro F1: {best_ensemble_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, best_preds)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'Macro F1':<10} {'Accuracy':<10}")
    print("-"*45)
    print(f"{'DeBERTa-large Hybrid':<25} {transformer_f1:.4f}     {accuracy_score(y_test, transformer_preds):.4f}")
    
    for name, ml_model in ml_models.items():
        ml_preds = ml_model.predict(X_test_feat)
        ml_f1 = f1_score(y_test, ml_preds, average='macro')
        ml_acc = accuracy_score(y_test, ml_preds)
        print(f"{name:<25} {ml_f1:.4f}     {ml_acc:.4f}")
    
    print(f"{'ENSEMBLE':<25} {best_ensemble_f1:.4f}     {accuracy_score(y_test, best_preds):.4f}")

    save_path = f"{Config.MODEL_DIR}/model6_ensemble.pt"
    torch.save({
        'transformer_state': best_state,
        'ml_models': ml_models,
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'best_weights': best_config,
        'metrics': {'ensemble_f1': best_ensemble_f1, 'transformer_f1': transformer_f1}
    }, save_path)
    print(f"\nEnsemble saved to: {save_path}")


if __name__ == "__main__":
    main()