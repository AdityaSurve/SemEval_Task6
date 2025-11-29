import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ClarityClassifier(nn.Module):
    """Neural network that learns to classify Q-A pairs."""

    def __init__(self, embedding_dim, num_classes, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Question encoder
        self.q_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Answer encoder
        self.a_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Interaction layer - learns relationship between Q and A
        # Input: q_encoded, a_encoded, element-wise product, absolute difference
        interaction_dim = hidden_dims[0] * 4

        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dims[2], num_classes),
        )

    def forward(self, q_emb, a_emb):
        # Encode Q and A separately
        q_enc = self.q_encoder(q_emb)
        a_enc = self.a_encoder(a_emb)

        # Interaction features - let model learn what relationships matter
        interaction = torch.cat([
            q_enc,                      # Question representation
            a_enc,                      # Answer representation
            q_enc * a_enc,              # Element-wise interaction
            torch.abs(q_enc - a_enc),  # Difference (what's missing/added)
        ], dim=-1)

        return self.classifier(interaction)


def get_embeddings(texts, model, batch_size=32):
    """Get embeddings for a list of texts."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for q_emb, a_emb, labels in loader:
        q_emb, a_emb, labels = q_emb.to(device), a_emb.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(q_emb, a_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for q_emb, a_emb, labels in loader:
            q_emb, a_emb, labels = q_emb.to(device), a_emb.to(device), labels.to(device)

            outputs = model(q_emb, a_emb)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), macro_f1, all_preds, all_labels


def main():
    device = torch.device("cpu")
    arrow_path = os.path.join("dataset", "train", "train", "data-00000-of-00001.arrow")
    ds = Dataset.from_file(arrow_path)
    df = ds.to_pandas()

    print(f"Dataset size: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    le = LabelEncoder()
    labels = le.fit_transform(df["label"])
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    print("\nLoading sentence transformer...")
    sbert = SentenceTransformer("all-mpnet-base-v2") 
    embedding_dim = 768

    print("Encoding questions...")
    q_embeddings = get_embeddings(df["question"].tolist(), sbert)

    print("Encoding answers...")
    a_embeddings = get_embeddings(df["interview_answer"].tolist(), sbert)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42, stratify=labels[train_idx])

    q_train = torch.FloatTensor(q_embeddings[train_idx])
    a_train = torch.FloatTensor(a_embeddings[train_idx])
    y_train = torch.LongTensor(labels[train_idx])
    q_val = torch.FloatTensor(q_embeddings[val_idx])
    a_val = torch.FloatTensor(a_embeddings[val_idx])
    y_val = torch.LongTensor(labels[val_idx])
    q_test = torch.FloatTensor(q_embeddings[test_idx])
    a_test = torch.FloatTensor(a_embeddings[test_idx])
    y_test = torch.LongTensor(labels[test_idx])

    train_loader = DataLoader(
        TensorDataset(q_train, a_train, y_train),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(q_val, a_val, y_val),
        batch_size=64
    )
    test_loader = DataLoader(
        TensorDataset(q_test, a_test, y_test),
        batch_size=64
    )

    class_counts = np.bincount(labels[train_idx])
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes

    model = ClarityClassifier(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_f1 = 0
    patience_counter = 0
    max_patience = 15

    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_clarity_model.pt")
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 5 == 0 or marker:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | Best: {best_f1:.4f}{marker}")

        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load("best_clarity_model.pt"))
    test_loss, test_f1, preds, true_labels = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print(f"FINAL TEST RESULTS (Macro F1: {test_f1:.4f})")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=le.classes_))


if __name__ == "__main__":
    main()


