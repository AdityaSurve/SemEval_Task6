import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import spacy
import joblib
import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]


class SemanticFeatureExtractor:
    def __init__(self):
        print("Loading Sentence Transformer...")
        self.sbert = SentenceTransformer("all-mpnet-base-v2")

        self.hedge_words = self._load_file("external_datasets/hedges.txt")
        self.filler_words = {"um", "uh", "er", "ah",
                             "like", "basically", "actually", "literally"}
        self.pivot_phrases = ["let me be clear", "the fact is", "the truth is",
                              "what matters is", "the important thing", "the real question", "at the end of the day"]
        self.thanks_phrases = ["thank you", "thanks for",
                               "great question", "good question", "i appreciate"]
        self.direct_yes = ["yes", "yeah", "yep", "yea", "absolutely",
                           "definitely", "certainly", "of course", "exactly", "correct"]
        self.direct_no = ["no", "nope", "nah",
                          "never", "not at all", "absolutely not"]
        self.wh_words = {"what", "who", "where", "when", "why", "how", "which"}

    def _load_file(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return set(line.strip().lower() for line in f if line.strip())
        return set()

    def extract_features(self, question, answer):
        question = str(question) if question else ""
        answer = str(answer) if answer else ""
        q_lower = question.lower().strip()
        a_lower = answer.lower().strip()
        q_words = q_lower.split()
        a_words = a_lower.split()

        q_emb = self.sbert.encode(question, convert_to_numpy=True)
        a_emb = self.sbert.encode(answer, convert_to_numpy=True)

        features = {}

        features["semantic_similarity"] = 1 - \
            cosine(q_emb, a_emb) if np.any(q_emb) and np.any(a_emb) else 0

        first_sentence = answer.split(
            '.')[0] if '.' in answer else answer[:100]
        first_emb = self.sbert.encode(first_sentence, convert_to_numpy=True)
        features["first_sent_similarity"] = 1 - \
            cosine(q_emb, first_emb) if np.any(first_emb) else 0

        features["q_len"] = len(q_words)
        features["a_len"] = len(a_words)
        features["a_q_ratio"] = len(a_words) / max(len(q_words), 1)

        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ",
                           "will ", "would ", "could ", "should ", "can ", "has ", "have ", "had "]
        features["is_yes_no_q"] = int(
            any(q_lower.startswith(s) for s in yes_no_starters))
        features["is_wh_q"] = int(any(q_lower.startswith(w)
                                  for w in self.wh_words))

        first_words = a_words[:3] if len(a_words) >= 3 else a_words
        features["starts_yes"] = int(
            any(w in self.direct_yes for w in first_words))
        features["starts_no"] = int(
            any(w in self.direct_no for w in first_words))
        features["starts_direct"] = int(
            features["starts_yes"] or features["starts_no"])

        features["yes_no_q_with_direct"] = int(
            features["is_yes_no_q"] and features["starts_direct"])
        features["yes_no_q_without_direct"] = int(
            features["is_yes_no_q"] and not features["starts_direct"])
        features["wh_q_short_answer"] = int(
            features["is_wh_q"] and len(a_words) < 20)

        features["hedge_count"] = sum(
            1 for h in self.hedge_words if h in a_lower)
        features["hedge_ratio"] = features["hedge_count"] / \
            max(len(a_words), 1)
        features["has_hedges"] = int(features["hedge_count"] > 0)

        features["pivot_count"] = sum(
            1 for p in self.pivot_phrases if p in a_lower)
        features["has_pivot"] = int(features["pivot_count"] > 0)

        features["thanks_start"] = int(
            any(a_lower.startswith(t) for t in self.thanks_phrases))

        features["filler_count"] = sum(
            1 for f in self.filler_words if f in a_words)
        features["filler_ratio"] = features["filler_count"] / \
            max(len(a_words), 1)

        features["answer_has_question"] = int("?" in answer)
        features["ends_with_question"] = int(answer.strip().endswith("?"))

        q_doc = nlp(q_lower[:2000])
        a_doc = nlp(a_lower[:2000])

        q_content = {
            t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {
            t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        overlap = q_content & a_content
        features["lexical_overlap"] = len(
            overlap) / max(len(q_content), 1) if q_content else 0

        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["entity_overlap"] = len(
            q_entities & a_entities) / max(len(q_entities), 1) if q_entities else 0
        features["new_entities"] = len(a_entities - q_entities)

        features["first_person"] = sum(1 for w in a_words if w in {
                                       "i", "me", "my", "mine", "myself"})
        features["uses_we"] = int("we" in a_words or "our" in a_words)
        features["uses_you"] = int("you" in a_words or "your" in a_words)

        features["very_short"] = int(len(a_words) < 15)
        features["short"] = int(len(a_words) < 40)
        features["long"] = int(len(a_words) > 100)
        features["very_long"] = int(len(a_words) > 200)

        features["explicit_signal"] = (
            features["starts_direct"] * 3 +
            features["semantic_similarity"] * 5 +
            features["lexical_overlap"] * 3 +
            features["entity_overlap"] * 2 +
            features["yes_no_q_with_direct"] * 3 -
            features["hedge_ratio"] * 10
        )

        features["dodging_signal"] = (
            features["has_pivot"] * 5 +
            features["thanks_start"] * 4 +
            features["answer_has_question"] * 3 +
            features["yes_no_q_without_direct"] * 4 +
            (1 - features["semantic_similarity"]) * 5 +
            features["uses_you"] * 2 -
            features["starts_direct"] * 3
        )

        features["partial_signal"] = (
            features["hedge_count"] * 2 +
            features["has_hedges"] * 3 +
            int(0.3 < features["semantic_similarity"] < 0.6) * 3 +
            int(0.1 < features["lexical_overlap"] < 0.4) * 2
        )

        features["general_signal"] = (
            (1 - features["entity_overlap"]) * 3 +
            features["new_entities"] * 0.5 +
            features["long"] * 2 +
            features["uses_we"] * 2 +
            int(features["lexical_overlap"] < 0.2) * 2
        )

        return features, q_emb, a_emb


def load_data():
    df = pd.read_parquet("dataset/train_with_features.parquet")
    df = df[df["label"].isin(FOUR_CLASSES)].reset_index(drop=True)
    return df


def main():
    df = load_data()
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    extractor = SemanticFeatureExtractor()

    print("Extracting features...")
    features_list = []
    q_embeddings = []
    a_embeddings = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  {idx}/{len(df)}")
        feats, q_emb, a_emb = extractor.extract_features(
            row["question"], row["interview_answer"])
        features_list.append(feats)
        q_embeddings.append(q_emb)
        a_embeddings.append(a_emb)

    feature_df = pd.DataFrame(features_list)
    q_emb_array = np.array(q_embeddings)
    a_emb_array = np.array(a_embeddings)

    print(f"\nLinguistic features: {len(feature_df.columns)}")
    print(f"Embedding dim: {q_emb_array.shape[1]}")

    X_ling = feature_df.values
    X_emb = np.hstack([q_emb_array, a_emb_array])
    X_combined = np.hstack([X_ling, X_emb])

    print(f"Total features: {X_combined.shape[1]}")

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"Classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {np.bincount(y_train_bal)}")

    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            num_leaves=50, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric="mlogloss"
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            random_state=42, verbose=0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=15, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=42
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
        results[name] = {"model": model, "f1": f1, "acc": acc, "pred": y_pred}

    print("\n" + "="*60)
    print("ENSEMBLE")
    print("="*60)

    top3 = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)[:3]
    print(f"Top 3: {[n for n, _ in top3]}")

    probas = []
    for name, res in top3:
        probas.append(res["model"].predict_proba(X_test_scaled))

    avg_proba = np.mean(probas, axis=0)
    ensemble_pred = np.argmax(avg_proba, axis=1)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average="macro")
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"\nEnsemble F1: {ensemble_f1:.4f} | Acc: {ensemble_acc:.4f}")
    results["Ensemble"] = {"f1": ensemble_f1,
                           "acc": ensemble_acc, "pred": ensemble_pred}

    print("\n" + "="*60)
    print("FINAL RANKING")
    print("="*60)
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, res) in enumerate(sorted_results):
        print(f"{i+1}. {name}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}")

    best_name = sorted_results[0][0]
    best_pred = sorted_results[0][1]["pred"]

    print("\n" + "="*60)
    print(f"BEST: {best_name}")
    print("="*60)
    print(classification_report(y_test, best_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))

    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    print("\nTop linguistic features by importance:")
    feature_names = feature_df.columns.tolist()

    best_model = sorted_results[0][1].get("model")
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_[:len(feature_names)]
        indices = np.argsort(importances)[::-1][:15]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "models": {n: r["model"] for n, r in results.items() if "model" in r},
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names
    }, "models/model10_semantic.joblib")
    print("\nModel saved!")


if __name__ == "__main__":
    main()
