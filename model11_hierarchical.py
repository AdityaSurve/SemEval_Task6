import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import lightgbm as lgb
import xgboost as xgb
import spacy
import joblib
import warnings
import re
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]


class FeatureExtractor:
    def __init__(self):
        print("Loading Sentence Transformer...")
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.hedge_words = self._load_lexicon("external_datasets/lexicons/hedges.txt")
        self.direct_yes = {"yes", "yeah", "yep", "absolutely", "definitely", "certainly", "correct", "exactly", "right"}
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
        features["a_q_ratio"] = len(a_words) / max(len(q_words), 1)
        
        features["q_has_and"] = int(" and " in q_lower)
        features["q_multi_question"] = int(q_lower.count("?") > 1)
        features["q_is_complex"] = int(len(q_words) > 20 or features["q_has_and"] or features["q_multi_question"])
        features["q_question_count"] = q_lower.count("?")
        features["q_comma_count"] = q_lower.count(",")
        
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ", "will ", "would ", 
                          "could ", "should ", "can ", "has ", "have ", "had "]
        features["is_yes_no_q"] = int(any(q_lower.startswith(s) for s in yes_no_starters))
        features["is_wh_q"] = int(any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which"]))
        
        first_3 = a_words[:3] if len(a_words) >= 3 else a_words
        first_word = a_words[0] if a_words else ""
        features["starts_yes"] = int(first_word in self.direct_yes or any(w in self.direct_yes for w in first_3))
        features["starts_no"] = int(first_word in self.direct_no or (len(a_words) > 1 and a_words[0] == "no"))
        features["starts_direct"] = int(features["starts_yes"] or features["starts_no"])
        features["starts_well"] = int(first_word == "well" or a_lower.startswith("well,"))
        features["starts_look"] = int(first_word == "look" or a_lower.startswith("look,"))
        
        features["yes_no_q_direct_a"] = int(features["is_yes_no_q"] and features["starts_direct"])
        features["yes_no_q_no_direct"] = int(features["is_yes_no_q"] and not features["starts_direct"])
        
        features["has_pivot"] = int(any(p in a_lower for p in self.pivot_phrases))
        features["has_thanks"] = int(any(t in a_lower for t in self.thanks_phrases))
        features["has_question_in_answer"] = int("?" in answer)
        features["ends_with_question"] = int(answer.strip().endswith("?"))
        
        features["hedge_count"] = sum(1 for h in self.hedge_words if h in a_words)
        features["hedge_ratio"] = features["hedge_count"] / max(len(a_words), 1)
        
        q_doc = nlp(q_lower[:1500])
        a_doc = nlp(a_lower[:1500])
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        features["lexical_overlap"] = len(q_content & a_content) / max(len(q_content), 1) if q_content else 0
        
        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["entity_overlap"] = len(q_entities & a_entities) / max(len(q_entities), 1) if q_entities else 0
        features["new_entities"] = len(a_entities - q_entities)
        features["q_entity_count"] = len(q_entities)
        
        features["first_person_count"] = sum(1 for w in a_words if w in {"i", "me", "my", "mine", "myself"})
        features["uses_we"] = int("we" in a_words or "our" in a_words)
        features["uses_you"] = int("you" in a_words or "your" in a_words)
        
        features["num_count"] = len(re.findall(r'\d+', answer))
        features["has_numbers"] = int(features["num_count"] > 0)
        
        features["a_very_short"] = int(len(a_words) < 20)
        features["a_short"] = int(len(a_words) < 50)
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
        
        features["partial_signal"] = (
            features["q_is_complex"] * 5 +
            features["q_multi_question"] * 8 +
            features["q_has_and"] * 3 +
            (features["q_len_words"] / 10) * 2 +
            features["starts_well"] * 2 +
            int(0.25 < features["semantic_sim"] < 0.5) * 3
        )
        
        features["general_signal"] = (
            (1 - features["entity_overlap"]) * 3 +
            features["a_long"] * 2 +
            features["uses_we"] * 2 +
            features["hedge_ratio"] * 5 +
            int(features["semantic_sim"] > 0.3) * 2 -
            features["q_is_complex"] * 3
        )
        
        return features, q_emb, a_emb


def load_data():
    df = pd.read_parquet("dataset/train_with_features.parquet")
    df = df[df["label"].isin(FOUR_CLASSES)].reset_index(drop=True)
    return df


class HierarchicalClassifier:
    def __init__(self):
        self.clf_dodging = None
        self.clf_explicit = None
        self.clf_general_partial = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        
        print("\n" + "="*60)
        print("LEVEL 1: Dodging vs Non-Dodging")
        print("="*60)
        y_dodging = (y == 0).astype(int)
        
        smote1 = SMOTE(random_state=42, k_neighbors=3)
        X_bal1, y_bal1 = smote1.fit_resample(X_scaled, y_dodging)
        
        self.clf_dodging = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1
        )
        self.clf_dodging.fit(X_bal1, y_bal1)
        
        dodging_pred = self.clf_dodging.predict(X_scaled)
        print(f"Dodging detection - Acc: {accuracy_score(y_dodging, dodging_pred):.4f}")
        print(f"Dodging F1: {f1_score(y_dodging, dodging_pred):.4f}")
        
        print("\n" + "="*60)
        print("LEVEL 2: Explicit vs (General+Partial) [non-dodging only]")
        print("="*60)
        non_dodging_mask = y != 0
        X_non_dodging = X_scaled[non_dodging_mask]
        y_non_dodging = y[non_dodging_mask]
        y_explicit = (y_non_dodging == 1).astype(int)
        
        smote2 = SMOTE(random_state=42, k_neighbors=3)
        X_bal2, y_bal2 = smote2.fit_resample(X_non_dodging, y_explicit)
        
        self.clf_explicit = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1
        )
        self.clf_explicit.fit(X_bal2, y_bal2)
        
        explicit_pred = self.clf_explicit.predict(X_non_dodging)
        print(f"Explicit detection - Acc: {accuracy_score(y_explicit, explicit_pred):.4f}")
        print(f"Explicit F1: {f1_score(y_explicit, explicit_pred):.4f}")
        
        print("\n" + "="*60)
        print("LEVEL 3: General vs Partial [non-explicit, non-dodging]")
        print("="*60)
        gen_par_mask = (y == 2) | (y == 3)
        X_gen_par = X_scaled[gen_par_mask]
        y_gen_par = y[gen_par_mask]
        y_partial = (y_gen_par == 3).astype(int)
        
        smote3 = SMOTE(random_state=42, k_neighbors=2)
        X_bal3, y_bal3 = smote3.fit_resample(X_gen_par, y_partial)
        
        self.clf_general_partial = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            class_weight="balanced", random_state=42, verbose=-1
        )
        self.clf_general_partial.fit(X_bal3, y_bal3)
        
        gp_pred = self.clf_general_partial.predict(X_gen_par)
        print(f"General/Partial detection - Acc: {accuracy_score(y_partial, gp_pred):.4f}")
        print(f"Partial F1: {f1_score(y_partial, gp_pred):.4f}")
        
        self._print_feature_importance()
        
    def _print_feature_importance(self):
        print("\n" + "="*60)
        print("KEY FEATURES PER LEVEL")
        print("="*60)
        
        for name, clf in [("Dodging", self.clf_dodging), 
                          ("Explicit", self.clf_explicit), 
                          ("General/Partial", self.clf_general_partial)]:
            print(f"\n{name} classifier - Top 10 features:")
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            for i, idx in enumerate(indices):
                print(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros(len(X), dtype=int)
        
        dodging_proba = self.clf_dodging.predict_proba(X_scaled)[:, 1]
        is_dodging = dodging_proba > 0.5
        predictions[is_dodging] = 0
        
        non_dodging_mask = ~is_dodging
        if non_dodging_mask.sum() > 0:
            explicit_proba = self.clf_explicit.predict_proba(X_scaled[non_dodging_mask])[:, 1]
            is_explicit = explicit_proba > 0.5
            
            non_dodging_indices = np.where(non_dodging_mask)[0]
            predictions[non_dodging_indices[is_explicit]] = 1
            
            remaining_mask = np.zeros(len(X), dtype=bool)
            remaining_mask[non_dodging_indices[~is_explicit]] = True
            
            if remaining_mask.sum() > 0:
                partial_proba = self.clf_general_partial.predict_proba(X_scaled[remaining_mask])[:, 1]
                is_partial = partial_proba > 0.5
                
                remaining_indices = np.where(remaining_mask)[0]
                predictions[remaining_indices[is_partial]] = 3
                predictions[remaining_indices[~is_partial]] = 2
        
        return predictions
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        n_samples = len(X)
        proba = np.zeros((n_samples, 4))
        
        dodging_proba = self.clf_dodging.predict_proba(X_scaled)[:, 1]
        proba[:, 0] = dodging_proba
        
        non_dodging_proba = 1 - dodging_proba
        explicit_proba = self.clf_explicit.predict_proba(X_scaled)[:, 1]
        proba[:, 1] = non_dodging_proba * explicit_proba
        
        non_explicit_proba = 1 - explicit_proba
        partial_proba = self.clf_general_partial.predict_proba(X_scaled)[:, 1]
        
        proba[:, 2] = non_dodging_proba * non_explicit_proba * (1 - partial_proba)
        proba[:, 3] = non_dodging_proba * non_explicit_proba * partial_proba
        
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


def main():
    df = load_data()
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
    extractor = FeatureExtractor()
    
    print("Extracting features...")
    features_list = []
    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"  {idx}/{len(df)}")
        feats, _, _ = extractor.extract(row["question"], row["interview_answer"])
        features_list.append(feats)
    
    feature_df = pd.DataFrame(features_list)
    feature_names = feature_df.columns.tolist()
    X = feature_df.values
    
    print(f"\nTotal features: {len(feature_names)}")
    
    le = LabelEncoder()
    le.classes_ = np.array(FOUR_CLASSES)
    y = le.transform(df["label"])
    print(f"Classes: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    clf = HierarchicalClassifier()
    clf.fit(X_train, y_train, feature_names)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=FOUR_CLASSES))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n" + "="*60)
    print("FLAT CLASSIFIER COMPARISON")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    
    flat_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42, verbose=-1
    )
    flat_clf.fit(X_train_bal, y_train_bal)
    y_pred_flat = flat_clf.predict(X_test_scaled)
    
    flat_acc = accuracy_score(y_test, y_pred_flat)
    flat_f1 = f1_score(y_test, y_pred_flat, average="macro")
    
    print(f"\nFlat LightGBM - Accuracy: {flat_acc:.4f}, Macro F1: {flat_f1:.4f}")
    print(f"Hierarchical  - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    print(f"\nImprovement: F1 +{(f1-flat_f1)*100:.2f}%, Acc +{(acc-flat_acc)*100:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "hierarchical_clf": clf,
        "flat_clf": flat_clf,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names
    }, "models/model11_hierarchical.joblib")
    print("\nModel saved to models/model11_hierarchical.joblib")


if __name__ == "__main__":
    main()
