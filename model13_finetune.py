import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from collections import Counter
import spacy
import joblib
import warnings
import re
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]


class TopFeatureExtractor:
    def __init__(self):
        print("Loading Sentence Transformer...")
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        
    def extract(self, question, answer):
        q = str(question) if question else ""
        a = str(answer) if answer else ""
        q_lower = q.lower().strip()
        a_lower = a.lower().strip()
        q_words = q_lower.split()
        a_words = a_lower.split()
        
        features = {}
        
        q_emb = self.sbert.encode(q)
        a_emb = self.sbert.encode(a[:1500])
        features["semantic_sim"] = 1 - cosine(q_emb, a_emb) if np.any(q_emb) and np.any(a_emb) else 0
        
        first_sent = a.split('.')[0] if '.' in a else a[:150]
        first_emb = self.sbert.encode(first_sent)
        features["first_sent_sim"] = 1 - cosine(q_emb, first_emb) if np.any(first_emb) else 0
        
        sentences = [s.strip() for s in a.split('.') if s.strip()]
        if len(sentences) > 1:
            last_emb = self.sbert.encode(sentences[-1])
            features["last_sent_sim"] = 1 - cosine(q_emb, last_emb)
            features["topic_drift"] = features["first_sent_sim"] - features["last_sent_sim"]
        else:
            features["last_sent_sim"] = features["first_sent_sim"]
            features["topic_drift"] = 0
        
        features["q_len_words"] = len(q_words)
        features["a_len_words"] = len(a_words)
        features["q_has_multiple_questions"] = int(q.count("?") > 1)
        features["q_has_and"] = int(" and " in q_lower)
        features["q_is_complex"] = int(len(q_words) > 20 or features["q_has_and"] or features["q_has_multiple_questions"])
        
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ", "will ", "would ", "could ", "should ", "can ", "has ", "have ", "had "]
        features["is_yes_no_q"] = int(any(q_lower.startswith(s) for s in yes_no_starters))
        features["is_wh_q"] = int(any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which"]))
        features["wh_q_short_a"] = int(features["is_wh_q"] and len(a_words) < 30)
        
        direct_yes = {"yes", "yeah", "yep", "absolutely", "definitely", "certainly", "correct", "exactly", "right", "sure"}
        direct_no = {"no", "nope", "nah", "never", "not"}
        first_word = a_words[0] if a_words else ""
        first_3 = a_words[:3] if len(a_words) >= 3 else a_words
        features["starts_yes"] = int(first_word in direct_yes or any(w in direct_yes for w in first_3))
        features["starts_no"] = int(first_word in direct_no)
        features["starts_direct"] = int(features["starts_yes"] or features["starts_no"])
        features["starts_well_look"] = int(a_lower.startswith("well, look") or a_lower.startswith("well look"))
        
        q_doc = nlp(q_lower[:500])
        a_doc = nlp(a_lower[:2000])
        
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        features["lexical_overlap"] = len(q_content & a_content) / max(len(q_content), 1) if q_content else 0
        features["lexical_overlap_count"] = len(q_content & a_content)
        
        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["entity_overlap"] = len(q_entities & a_entities) / max(len(q_entities), 1) if q_entities else 0
        features["entity_overlap_count"] = len(q_entities & a_entities)
        features["q_entity_count"] = len(q_entities)
        
        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        first_sent_lower = first_sent.lower()
        features["first_sent_has_q_noun"] = int(any(n in first_sent_lower for n in q_nouns))
        
        uncertainty_modals = ["would", "could", "might", "may", "should"]
        features["uncertainty_modal_count"] = sum(1 for m in uncertainty_modals if m in a_words)
        
        features["short_sentences"] = sum(1 for s in sentences if len(s.split()) < 10)
        features["sentence_count"] = len(sentences)
        
        hesitation = ["i mean", "you know", "sort of", "kind of", "basically", "actually", "honestly"]
        features["has_hesitation"] = int(any(h in a_lower for h in hesitation))
        
        comparatives = ["but", "however", "although", "though", "whereas", "nevertheless", "yet"]
        features["has_contrast"] = int(any(c in a_lower for c in comparatives))
        
        features["ends_with_question"] = int(a.strip().endswith("?"))
        features["num_count"] = len(re.findall(r'\d+', a))
        
        commitment = ["promise", "commit", "guarantee", "ensure", "definitely", "certainly", "absolutely"]
        features["commitment_count"] = sum(1 for c in commitment if c in a_lower)
        
        features["present_count"] = sum(1 for w in ["is", "are", "am", "now", "today", "currently"] if w in a_words)
        
        features["specific_quantity_count"] = len(re.findall(r'\b\d+\b', a))
        
        features["has_conditional"] = int(any(c in a_lower for c in ["if ", "unless ", "whether "]))
        
        return features


def load_data():
    df = pd.read_parquet("dataset/train_with_features.parquet")
    df = df[df["label"].isin(FOUR_CLASSES)].reset_index(drop=True)
    return df


def main():
    print("="*60)
    print("MODEL WITH TOP DISCRIMINATIVE FEATURES")
    print("="*60)
    
    df = load_data()
    print(f"\nDataset size: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
    extractor = TopFeatureExtractor()
    
    print("Extracting top features...")
    features_list = []
    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"  {idx}/{len(df)}")
        feats = extractor.extract(row["question"], row["interview_answer"])
        features_list.append(feats)
    
    feature_df = pd.DataFrame(features_list)
    feature_names = feature_df.columns.tolist()
    X = feature_df.values
    
    print(f"\nTotal features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    le = LabelEncoder()
    le.classes_ = np.array(FOUR_CLASSES)
    y = le.transform(df["label"])
    print(f"Classes: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
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
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=40, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric="mlogloss"
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.05,
            random_state=42, verbose=0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
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
    print(f"Top 3 for ensemble: {[n for n, _ in top3]}")
    
    probas = []
    for name, res in top3:
        probas.append(res["model"].predict_proba(X_test_scaled))
    
    avg_proba = np.mean(probas, axis=0)
    ensemble_pred = np.argmax(avg_proba, axis=1)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average="macro")
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble F1: {ensemble_f1:.4f} | Acc: {ensemble_acc:.4f}")
    results["Ensemble"] = {"f1": ensemble_f1, "acc": ensemble_acc, "pred": ensemble_pred}
    
    print("\n" + "="*60)
    print("FINAL RANKING")
    print("="*60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, res) in enumerate(sorted_results):
        print(f"{i+1}. {name}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}")
    
    best_name = sorted_results[0][0]
    best_pred = sorted_results[0][1]["pred"]
    
    print("\n" + "="*60)
    print(f"BEST: {best_name}")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, target_names=FOUR_CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    best_model = sorted_results[0][1].get("model")
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nTop features by importance:")
        for i, idx in enumerate(indices[:20]):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "models": {n: r["model"] for n, r in results.items() if "model" in r},
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names
    }, "models/model13_top_features.joblib")
    print("\nModel saved to models/model13_top_features.joblib")


if __name__ == "__main__":
    main()




