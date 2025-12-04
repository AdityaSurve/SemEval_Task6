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
import spacy
import joblib
import warnings
import re
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
THREE_CLASSES = ["Dodging", "Explicit", "Indirect"]


class FeatureExtractor:
    def __init__(self):
        print("Loading Sentence Transformer...")
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.hedge_words = self._load_lexicon("external_datasets/lexicons/hedges.txt")
        self.direct_yes = {"yes", "yeah", "yep", "absolutely", "definitely", "certainly", "correct", "exactly", "right", "sure"}
        self.direct_no = {"no", "nope", "nah", "never", "not"}
        self.pivot_phrases = ["let me", "the fact is", "the truth is", "what matters", "the important thing", 
                             "the real question", "at the end of the day", "look,", "well, look"]
        self.thanks_phrases = ["thank you", "thanks for", "great question", "good question"]
        
    def _load_lexicon(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return set(line.strip().lower() for line in f if line.strip())
        return {"maybe", "perhaps", "possibly", "might", "could", "seem", "appear", "somewhat"}
    
    def extract(self, question, answer):
        question = str(question) if question else ""
        answer = str(answer) if answer else ""
        q_lower = question.lower().strip()
        a_lower = answer.lower().strip()
        q_words = q_lower.split()
        a_words = a_lower.split()
        
        q_emb = self.sbert.encode(question, convert_to_numpy=True)
        a_emb = self.sbert.encode(answer[:1500], convert_to_numpy=True)
        
        features = {}
        
        features["semantic_sim"] = 1 - cosine(q_emb, a_emb) if np.any(q_emb) and np.any(a_emb) else 0
        
        first_sent = answer.split('.')[0] if '.' in answer else answer[:150]
        first_emb = self.sbert.encode(first_sent, convert_to_numpy=True)
        features["first_sent_sim"] = 1 - cosine(q_emb, first_emb) if np.any(first_emb) else 0
        
        features["q_len_words"] = len(q_words)
        features["a_len_words"] = len(a_words)
        features["q_len_chars"] = len(question)
        features["a_len_chars"] = len(answer)
        features["a_q_word_ratio"] = len(a_words) / max(len(q_words), 1)
        features["a_q_char_ratio"] = len(answer) / max(len(question), 1)
        
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ", "will ", "would ", 
                          "could ", "should ", "can ", "has ", "have ", "had "]
        features["is_yes_no_q"] = int(any(q_lower.startswith(s) for s in yes_no_starters))
        features["is_wh_q"] = int(any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which"]))
        features["q_has_question_mark"] = int("?" in question)
        
        first_3 = a_words[:3] if len(a_words) >= 3 else a_words
        first_word = a_words[0] if a_words else ""
        features["starts_yes"] = int(first_word in self.direct_yes or any(w in self.direct_yes for w in first_3))
        features["starts_no"] = int(first_word in self.direct_no or (len(a_words) > 1 and a_words[0] == "no"))
        features["starts_direct"] = int(features["starts_yes"] or features["starts_no"])
        features["starts_well"] = int(first_word == "well" or a_lower.startswith("well,"))
        features["starts_look"] = int(first_word == "look" or a_lower.startswith("look,"))
        features["starts_i"] = int(first_word == "i")
        features["starts_we"] = int(first_word == "we")
        features["starts_so"] = int(first_word == "so")
        
        features["yes_no_q_direct_a"] = int(features["is_yes_no_q"] and features["starts_direct"])
        features["yes_no_q_no_direct"] = int(features["is_yes_no_q"] and not features["starts_direct"])
        
        features["has_pivot"] = int(any(p in a_lower for p in self.pivot_phrases))
        features["has_thanks"] = int(any(t in a_lower for t in self.thanks_phrases))
        features["has_question_in_answer"] = int("?" in answer)
        features["ends_with_question"] = int(answer.strip().endswith("?"))
        features["question_count_in_answer"] = answer.count("?")
        
        features["hedge_count"] = sum(1 for h in self.hedge_words if h in a_words)
        features["hedge_ratio"] = features["hedge_count"] / max(len(a_words), 1)
        features["has_hedges"] = int(features["hedge_count"] > 0)
        
        q_doc = nlp(q_lower[:1500])
        a_doc = nlp(a_lower[:1500])
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        features["lexical_overlap"] = len(q_content & a_content) / max(len(q_content), 1) if q_content else 0
        features["lexical_overlap_count"] = len(q_content & a_content)
        
        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["entity_overlap"] = len(q_entities & a_entities) / max(len(q_entities), 1) if q_entities else 0
        features["entity_overlap_count"] = len(q_entities & a_entities)
        features["new_entities"] = len(a_entities - q_entities)
        features["q_entity_count"] = len(q_entities)
        features["a_entity_count"] = len(a_entities)
        
        features["first_person_count"] = sum(1 for w in a_words if w in {"i", "me", "my", "mine", "myself"})
        features["first_person_ratio"] = features["first_person_count"] / max(len(a_words), 1)
        features["uses_we"] = int("we" in a_words or "our" in a_words)
        features["uses_you"] = int("you" in a_words or "your" in a_words)
        features["uses_they"] = int("they" in a_words or "their" in a_words)
        
        features["num_count"] = len(re.findall(r'\d+', answer))
        features["has_numbers"] = int(features["num_count"] > 0)
        
        features["sentence_count"] = len([s for s in answer.split('.') if s.strip()])
        features["avg_sentence_len"] = len(a_words) / max(features["sentence_count"], 1)
        
        features["a_very_short"] = int(len(a_words) < 20)
        features["a_short"] = int(len(a_words) < 50)
        features["a_medium"] = int(50 <= len(a_words) <= 150)
        features["a_long"] = int(len(a_words) > 150)
        features["a_very_long"] = int(len(a_words) > 300)
        
        features["dodging_signal"] = (
            (1 - features["semantic_sim"]) * 10 +
            features["has_pivot"] * 5 +
            features["has_thanks"] * 4 +
            features["yes_no_q_no_direct"] * 5 +
            features["has_question_in_answer"] * 3 +
            features["uses_you"] * 2 -
            features["starts_direct"] * 8 -
            features["lexical_overlap"] * 5
        )
        
        features["explicit_signal"] = (
            features["semantic_sim"] * 10 +
            features["first_sent_sim"] * 5 +
            features["starts_direct"] * 8 +
            features["yes_no_q_direct_a"] * 5 +
            features["lexical_overlap"] * 5 +
            features["entity_overlap"] * 3 -
            features["hedge_ratio"] * 10 -
            features["has_pivot"] * 3
        )
        
        features["indirect_signal"] = (
            features["hedge_ratio"] * 10 +
            features["a_long"] * 3 +
            features["uses_we"] * 2 +
            (1 - features["entity_overlap"]) * 3 +
            int(0.25 < features["semantic_sim"] < 0.45) * 3
        )
        
        return features, q_emb, a_emb


def load_data():
    df = pd.read_parquet("dataset/custom/train_3class.parquet")
    return df


def main():
    df = load_data()
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
    extractor = FeatureExtractor()
    
    print("Extracting features...")
    features_list = []
    q_embeddings = []
    a_embeddings = []
    
    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"  {idx}/{len(df)}")
        feats, q_emb, a_emb = extractor.extract(row["question"], row["interview_answer"])
        features_list.append(feats)
        q_embeddings.append(q_emb)
        a_embeddings.append(a_emb)
    
    feature_df = pd.DataFrame(features_list)
    q_emb_array = np.array(q_embeddings)
    a_emb_array = np.array(a_embeddings)
    
    feature_names = feature_df.columns.tolist()
    X_features = feature_df.values
    X_embeddings = np.hstack([q_emb_array, a_emb_array])
    X_combined = np.hstack([X_features, X_embeddings])
    
    print(f"\nLinguistic features: {len(feature_names)}")
    print(f"Embedding features: {X_embeddings.shape[1]}")
    print(f"Total features: {X_combined.shape[1]}")
    
    le = LabelEncoder()
    le.classes_ = np.array(THREE_CLASSES)
    y = le.transform(df["label"])
    print(f"Classes: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
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
    print(classification_report(y_test, best_pred, target_names=THREE_CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    print("\n" + "="*60)
    print("TOP FEATURES")
    print("="*60)
    best_model = sorted_results[0][1].get("model")
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_[:len(feature_names)]
        indices = np.argsort(importances)[::-1][:20]
        print("\nTop 20 linguistic features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "models": {n: r["model"] for n, r in results.items() if "model" in r},
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names
    }, "models/model12_3class.joblib")
    print("\nModel saved to models/model12_3class.joblib")


if __name__ == "__main__":
    main()
