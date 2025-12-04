import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType


class Config:
    FEATURES_PATH = "dataset/train_with_features.parquet"
    MODEL_DIR = "models"
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    HIDDEN_DIM = 256
    DROPOUT = 0.2
    BATCH_SIZE = 8
    EPOCHS = 6
    LR = 1e-4
    FEATURE_LR = 5e-4
    GRADIENT_ACCUMULATION = 2
    WARMUP_RATIO = 0.1
    RANDOM_STATE = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]


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
        encoding = self.tokenizer(str(
            self.texts[idx]), truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class HybridClassifier(nn.Module):
    def __init__(self, model_name, num_features, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(
            model_name, output_hidden_states=True)
        self.transformer_dim = self.transformer.config.hidden_size
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
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, features):
        out = self.transformer(input_ids=input_ids,
                               attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        f = self.feature_mlp(features)
        x = torch.cat([cls, f], dim=1)
        return self.classifier(x)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps):
    model.train()
    total_loss = 0
    preds, trues = [], []
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(dataloader)):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        feats = batch["features"].to(device)
        labels = batch["labels"].to(device)
        logits = model(ids, mask, feats)
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), f1_score(trues, preds, average="macro")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids, mask, feats)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), f1_score(trues, preds, average="macro"), accuracy_score(trues, preds), preds, trues


def main():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    df = pd.read_parquet(Config.FEATURES_PATH)
    df = df[df["label"].isin(Config.FOUR_CLASSES)].reset_index(drop=True)
    df["text"] = "Question: " + df["question"] + \
        " [SEP] Answer: " + df["interview_answer"]

    feature_cols = [c for c in df.columns if c not in ["question", "interview_question", "interview_answer",
                                                       "label", "url", "inaudible", "multiple_questions", "affirmative_questions", "text"]]
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].fillna(0))

    le = LabelEncoder()
    labels = le.fit_transform(df["label"])

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        range(len(df)), labels, test_size=0.3, stratify=labels, random_state=42)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    X_train = features[train_idx]
    X_val = features[val_idx]
    X_test = features[test_idx]
    texts = df["text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_data = HybridDataset(
        [texts[i] for i in train_idx], X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_data = HybridDataset(
        [texts[i] for i in val_idx], X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_data = HybridDataset(
        [texts[i] for i in test_idx], X_test, y_test, tokenizer, Config.MAX_LENGTH)

    train_loader = DataLoader(
        train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE)

    model = HybridClassifier(Config.MODEL_NAME, num_features=len(
        feature_cols), num_classes=len(Config.FOUR_CLASSES), hidden_dim=256, dropout=0.2)    model.to(Config.DEVICE)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model.transformer = get_peft_model(model.transformer, lora_config)

    optimizer = torch.optim.AdamW([
        {"params": model.transformer.parameters(), "lr": Config.LR},
        {"params": list(model.feature_mlp.parameters()) +
         list(model.classifier.parameters()), "lr": Config.FEATURE_LR}
    ])
    total_steps = len(train_loader) * \
        Config.EPOCHS // Config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * Config.WARMUP_RATIO), total_steps)

    class_counts = np.bincount(y_train)
    weights = torch.tensor(len(y_train) / (4 * class_counts),
                           dtype=torch.float32).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = 0
    best_state = None
    patience = 3
    no_improve = 0

    for epoch in range(Config.EPOCHS):
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, Config.DEVICE, Config.GRADIENT_ACCUMULATION)
        val_loss, val_f1, val_acc, _, _ = evaluate(
            model, val_loader, criterion, Config.DEVICE)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.to(Config.DEVICE)

    test_loss, test_f1, test_acc, preds, trues = evaluate(
        model, test_loader, criterion, Config.DEVICE)
    print("Accuracy:", test_acc)
    print("Macro F1:", test_f1)
    print(classification_report(trues, preds, target_names=le.classes_))
    print(confusion_matrix(trues, preds))

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": feature_cols
    }, f"{Config.MODEL_DIR}/final_lora_hybrid.pt")


if __name__ == "__main__":
    main()
