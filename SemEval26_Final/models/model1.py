from utils.logger import Logger
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import spacy
import joblib
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

logger = Logger()
nlp = spacy.load("en_core_web_sm")

THREE_CLASSES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

DATASET_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "dataset")
ARTIFACTS_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "artifacts")


class ClarityFeatureExtractor:
    def __init__(self):
        logger.log(
            "Loading Sentence Transformer (mpnet-base - higher quality)...", "plain")
        self.sbert = SentenceTransformer(
            "all-mpnet-base-v2")

    def extract(self, question, answer):
        q = str(question) if question else ""
        a = str(answer) if answer else ""
        q_lower = q.lower().strip()
        a_lower = a.lower().strip()
        q_words = q_lower.split()
        a_words = a_lower.split()

        q_emb = self.sbert.encode(q)
        a_emb = self.sbert.encode(a[:1500])

        combined_emb = np.concatenate([q_emb, a_emb])
        diff_emb = q_emb - a_emb

        semantic_sim = 1 - \
            cosine(q_emb, a_emb) if np.any(q_emb) and np.any(a_emb) else 0

        first_sent = a.split('.')[0] if '.' in a else a[:150]
        first_emb = self.sbert.encode(first_sent)
        first_sent_sim = 1 - \
            cosine(q_emb, first_emb) if np.any(first_emb) else 0

        sentences = [s.strip() for s in a.split('.') if s.strip()]
        if len(sentences) > 1:
            last_emb = self.sbert.encode(sentences[-1])
            last_sent_sim = 1 - cosine(q_emb, last_emb)
        else:
            last_sent_sim = first_sent_sim

        ling_features = {}

        ling_features["semantic_sim"] = semantic_sim
        ling_features["first_sent_sim"] = first_sent_sim
        ling_features["last_sent_sim"] = last_sent_sim
        ling_features["sim_drop"] = first_sent_sim - last_sent_sim

        ling_features["q_len_words"] = len(q_words)
        ling_features["a_len_words"] = len(a_words)
        ling_features["a_q_ratio"] = len(a_words) / max(len(q_words), 1)
        ling_features["a_is_short"] = int(len(a_words) < 80)
        ling_features["a_is_very_short"] = int(len(a_words) < 40)

        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ",
                           "were ", "will ", "would ", "could ", "should ", "can "]
        ling_features["is_yes_no_q"] = int(
            any(q_lower.startswith(s) for s in yes_no_starters))
        ling_features["is_wh_q"] = int(any(q_lower.startswith(
            w) for w in ["what", "who", "where", "when", "why", "how", "which"]))

        direct_yes = {"yes", "yeah", "yep", "absolutely", "definitely",
                      "certainly", "correct", "exactly", "right", "sure", "of course"}
        direct_no = {"no", "nope", "nah", "never", "not"}
        first_word = a_words[0] if a_words else ""
        first_5 = set(a_words[:5]) if len(a_words) >= 5 else set(a_words)

        ling_features["starts_yes"] = int(
            first_word in direct_yes or len(first_5 & direct_yes) > 0)
        ling_features["starts_no"] = int(first_word in direct_no)
        ling_features["starts_direct"] = int(
            ling_features["starts_yes"] or ling_features["starts_no"])
        ling_features["starts_well"] = int(first_word == "well")
        ling_features["starts_i"] = int(first_word == "i")

        ling_features["yes_no_q_direct_a"] = int(
            ling_features["is_yes_no_q"] and ling_features["starts_direct"])

        q_doc = nlp(q_lower[:500])
        a_doc = nlp(a_lower[:2000])

        q_content = {
            t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {
            t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        ling_features["lexical_overlap"] = len(
            q_content & a_content) / max(len(q_content), 1) if q_content else 0

        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        first_sent_lower = first_sent.lower()
        ling_features["first_sent_has_q_noun"] = int(
            any(n in first_sent_lower for n in q_nouns))

        ling_features["sentence_count"] = len(sentences)

        refusal_phrases = ["i can't", "i cannot", "i won't", "no comment", "i'm not going to",
                           "i don't know", "not prepared to", "decline to"]
        ling_features["has_refusal"] = int(
            any(r in a_lower for r in refusal_phrases))

        ignorance = ["i don't know", "i'm not aware",
                     "no idea", "not sure", "i haven't"]
        ling_features["claims_ignorance"] = int(
            any(i in a_lower for i in ignorance))

        hedges = ["maybe", "perhaps", "possibly",
                  "probably", "might", "could", "somewhat"]
        ling_features["hedge_count"] = sum(1 for h in hedges if h in a_lower)

        commitment = ["i will", "i'm going to",
                      "we will", "we're going to", "i promise"]
        ling_features["commitment_count"] = sum(
            1 for c in commitment if c in a_lower)

        ling_features["has_question_in_answer"] = int("?" in a)
        ling_features["ends_question"] = int(a.strip().endswith("?"))

        return {
            "q_emb": q_emb,
            "a_emb": a_emb,
            "combined_emb": combined_emb,
            "diff_emb": diff_emb,
            "ling": ling_features
        }


def train():
    logger.log("TASK 1: CLARITY-LEVEL CLASSIFICATION (3 CLASSES)", "announce")

    train_path = os.path.join(DATASET_PATH, "task1_train.parquet")
    if not os.path.exists(train_path):
        logger.log(f"Training data not found at {train_path}", "error")
        logger.log(
            "Please place task1_train.parquet in the dataset folder", "warning")
        return

    df = pd.read_parquet(train_path)

    if "clarity_label" in df.columns:
        df["label"] = df["clarity_label"]
    elif "task1_label" in df.columns:
        df["label"] = df["task1_label"]

    if "question" not in df.columns and "interview_question" in df.columns:
        df["question"] = df["interview_question"]

    df = df[df["label"].isin(THREE_CLASSES)].reset_index(drop=True)

    logger.log(f"Dataset size: {len(df)}", "success")
    logger.log("Class distribution:", "plain")
    print(df["label"].value_counts())

    extractor = ClarityFeatureExtractor()

    logger.log("Extracting features...", "plain")
    q_embs = []
    a_embs = []
    combined_embs = []
    diff_embs = []
    ling_features_list = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"  {idx}/{len(df)}")
        feats = extractor.extract(row["question"], row["interview_answer"])
        q_embs.append(feats["q_emb"])
        a_embs.append(feats["a_emb"])
        combined_embs.append(feats["combined_emb"])
        diff_embs.append(feats["diff_emb"])
        ling_features_list.append(feats["ling"])

    Q_emb = np.array(q_embs)
    A_emb = np.array(a_embs)
    Combined_emb = np.array(combined_embs)
    Diff_emb = np.array(diff_embs)
    ling_df = pd.DataFrame(ling_features_list)

    logger.log(f"Q embeddings shape: {Q_emb.shape}", "plain")
    logger.log(f"A embeddings shape: {A_emb.shape}", "plain")
    logger.log(f"Linguistic features: {ling_df.shape[1]}", "plain")

    le = LabelEncoder()
    le.fit(THREE_CLASSES)
    y = le.transform(df["label"])

    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(df)), y, test_size=0.2, random_state=42, stratify=y
    )

    logger.log(
        f"Train: {len(X_train_idx)}, Test: {len(X_test_idx)}", "success")

    logger.log("PREPARING FEATURES", "announce")

    pca_qa = PCA(n_components=150)
    X_emb_qa_pca = pca_qa.fit_transform(Combined_emb)

    pca_diff = PCA(n_components=100)
    X_diff_pca = pca_diff.fit_transform(Diff_emb)

    X_combined = np.hstack([X_emb_qa_pca, X_diff_pca, ling_df.values])
    logger.log(f"Total features: {X_combined.shape[1]}", "success")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_combined[X_train_idx])
    X_test = scaler.transform(X_combined[X_test_idx])

    logger.log("TRAINING MODELS", "announce")

    models = {
        "LightGBM": lgb.LGBMClassifier(n_estimators=600, max_depth=10, learning_rate=0.03, class_weight='balanced', random_state=42, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.03, random_state=42, eval_metric='mlogloss'),
        "CatBoost": CatBoostClassifier(iterations=600, depth=8, learning_rate=0.03, random_state=42, verbose=False, auto_class_weights="Balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        logger.log(f"Training {name}...", "plain")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        logger.log(f"  F1: {f1:.4f} | Acc: {acc:.4f}", "success")
        results[name] = {"f1": f1, "acc": acc}
        trained_models[name] = model

    logger.log("ENSEMBLE METHODS", "announce")

    estimators = [(n, m) for n, m in trained_models.items()]

    logger.log("Soft Voting...", "plain")
    vote_clf = VotingClassifier(estimators=estimators, voting='soft')
    vote_clf.fit(X_train, y_train)
    y_pred = vote_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"  F1: {f1:.4f} | Acc: {acc:.4f}", "success")
    results["Voting Ensemble"] = {"f1": f1, "acc": acc, "pred": y_pred}

    logger.log("Stacking...", "plain")
    stack_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight='balanced', max_iter=500),
        cv=5
    )
    stack_clf.fit(X_train, y_train)
    y_pred = stack_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"  F1: {f1:.4f} | Acc: {acc:.4f}", "success")
    results["Stacking"] = {"f1": f1, "acc": acc, "pred": y_pred}

    logger.log("FINAL RANKING", "announce")
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, res) in enumerate(sorted_results):
        print(f"{i+1}. {name}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}")

    best_name = sorted_results[0][0]
    best_pred = results[best_name].get("pred")

    if best_pred is None:
        best_pred = trained_models[best_name].predict(X_test)

    logger.log(f"BEST MODEL: {best_name}", "announce")
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))

    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_PATH, "task1_model.joblib")
    joblib.dump({
        "trained_models": trained_models,
        "vote_clf": vote_clf,
        "stack_clf": stack_clf,
        "scaler": scaler,
        "pca_qa": pca_qa,
        "pca_diff": pca_diff,
        "label_encoder": le,
        "ling_feature_names": list(ling_df.columns),
    }, model_path)
    logger.log(f"Model saved to {model_path}", "success")


def predict(question: str, answer: str) -> str:
    model_path = os.path.join(ARTIFACTS_PATH, "task1_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please run train() first.")

    artifacts = joblib.load(model_path)
    extractor = ClarityFeatureExtractor()

    feats = extractor.extract(question, answer)

    combined_emb = feats["combined_emb"].reshape(1, -1)
    diff_emb = feats["diff_emb"].reshape(1, -1)
    ling = np.array([list(feats["ling"].values())])

    X_emb_qa_pca = artifacts["pca_qa"].transform(combined_emb)
    X_diff_pca = artifacts["pca_diff"].transform(diff_emb)
    X_combined = np.hstack([X_emb_qa_pca, X_diff_pca, ling])
    X_scaled = artifacts["scaler"].transform(X_combined)

    y_pred = artifacts["vote_clf"].predict(X_scaled)
    label = artifacts["label_encoder"].inverse_transform(y_pred)[0]

    return label


if __name__ == "__main__":
    train()
