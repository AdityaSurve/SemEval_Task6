import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import spacy
import joblib
import warnings
import re
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

THREE_CLASSES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


class HierarchicalFeatureExtractor:
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
            mid_idx = len(sentences) // 2
            mid_emb = self.sbert.encode(sentences[mid_idx])
            features["mid_sent_sim"] = 1 - cosine(q_emb, mid_emb)
        else:
            features["last_sent_sim"] = features["first_sent_sim"]
            features["topic_drift"] = 0
            features["mid_sent_sim"] = features["first_sent_sim"]
        
        features["avg_sent_sim"] = (features["first_sent_sim"] + features["mid_sent_sim"] + features["last_sent_sim"]) / 3
        features["max_sent_sim"] = max(features["first_sent_sim"], features["mid_sent_sim"], features["last_sent_sim"])
        features["min_sent_sim"] = min(features["first_sent_sim"], features["mid_sent_sim"], features["last_sent_sim"])
        
        features["q_len_words"] = len(q_words)
        features["a_len_words"] = len(a_words)
        features["a_len_chars"] = len(a)
        features["a_q_ratio"] = len(a_words) / max(len(q_words), 1)
        features["a_is_short"] = int(len(a_words) < 100)
        features["a_is_very_short"] = int(len(a_words) < 50)
        features["a_is_long"] = int(len(a_words) > 300)
        
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ", "will ", "would ", "could ", "should ", "can "]
        features["is_yes_no_q"] = int(any(q_lower.startswith(s) for s in yes_no_starters))
        features["is_wh_q"] = int(any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which"]))
        
        direct_yes = {"yes", "yeah", "yep", "absolutely", "definitely", "certainly", "correct", "exactly", "right", "sure"}
        direct_no = {"no", "nope", "nah", "never", "not"}
        first_word = a_words[0] if a_words else ""
        first_3 = a_words[:3] if len(a_words) >= 3 else a_words
        
        features["starts_yes"] = int(first_word in direct_yes or any(w in direct_yes for w in first_3))
        features["starts_no"] = int(first_word in direct_no)
        features["starts_direct"] = int(features["starts_yes"] or features["starts_no"])
        features["starts_well"] = int(first_word == "well" or a_lower.startswith("well,"))
        features["starts_i"] = int(first_word == "i")
        features["starts_look"] = int(first_word == "look")
        
        features["yes_no_q_direct_a"] = int(features["is_yes_no_q"] and features["starts_direct"])
        
        q_doc = nlp(q_lower[:500])
        a_doc = nlp(a_lower[:2000])
        
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        features["lexical_overlap"] = len(q_content & a_content) / max(len(q_content), 1) if q_content else 0
        features["lexical_overlap_count"] = len(q_content & a_content)
        
        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["entity_overlap"] = len(q_entities & a_entities) / max(len(q_entities), 1) if q_entities else 0
        features["a_entity_count"] = len(a_entities)
        
        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        first_sent_lower = first_sent.lower()
        features["first_sent_has_q_noun"] = int(any(n in first_sent_lower for n in q_nouns))
        features["q_noun_coverage"] = sum(1 for n in q_nouns if n in a_lower) / max(len(q_nouns), 1)
        
        uncertainty_modals = ["would", "could", "might", "may", "should"]
        features["uncertainty_modal_count"] = sum(1 for m in uncertainty_modals if m in a_words)
        
        features["sentence_count"] = len(sentences)
        features["avg_sentence_len"] = len(a_words) / max(len(sentences), 1)
        
        hesitation = ["i mean", "you know", "sort of", "kind of", "basically", "actually", "honestly"]
        features["hesitation_count"] = sum(1 for h in hesitation if h in a_lower)
        
        comparatives = ["but", "however", "although", "though", "nevertheless", "yet"]
        features["contrast_count"] = sum(1 for c in comparatives if c in a_lower)
        
        features["has_question_in_answer"] = int("?" in a)
        features["ends_with_question"] = int(a.strip().endswith("?"))
        
        pivot_phrases = ["let me", "the fact is", "the truth is", "what matters", "first of all", "look,", "the real question"]
        features["pivot_count"] = sum(1 for p in pivot_phrases if p in a_lower)
        
        hedges = ["maybe", "perhaps", "possibly", "probably", "might", "could", "somewhat"]
        features["hedge_count"] = sum(1 for h in hedges if h in a_lower)
        
        vague_words = ["something", "things", "stuff", "everything", "anything", "people", "everyone"]
        features["vague_count"] = sum(1 for v in vague_words if v in a_words)
        
        features["i_count"] = sum(1 for w in a_words if w == "i")
        features["we_count"] = sum(1 for w in a_words if w == "we")
        
        return features


class HierarchicalClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.stage1_clf = None  # Non-Reply vs Rest
        self.stage2_clf = None  # Clear Reply vs Ambivalent
        
    def fit(self, X, y, feature_names):
        X_scaled = self.scaler.fit_transform(X)
        
        print("\n--- Stage 1: Clear Non-Reply vs Rest ---")
        y_stage1 = (y == 2).astype(int)  # 2 = Clear Non-Reply
        
        stage1_models = [
            ("lgb", lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.03, class_weight="balanced", random_state=42, verbose=-1)),
            ("xgb", xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03, random_state=42, eval_metric="logloss")),
            ("rf", RandomForestClassifier(n_estimators=500, max_depth=15, class_weight="balanced", random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=500, max_depth=15, class_weight="balanced", random_state=42)),
        ]
        
        self.stage1_clf = VotingClassifier(estimators=stage1_models, voting='soft')
        self.stage1_clf.fit(X_scaled, y_stage1)
        
        stage1_pred = self.stage1_clf.predict(X_scaled)
        stage1_acc = accuracy_score(y_stage1, stage1_pred)
        print(f"  Stage 1 Train Acc: {stage1_acc:.4f}")
        
        print("\n--- Stage 2: Clear Reply vs Ambivalent ---")
        mask_not_nonreply = (y != 2)
        X_stage2 = X_scaled[mask_not_nonreply]
        y_stage2 = y[mask_not_nonreply]
        y_stage2_binary = (y_stage2 == 0).astype(int)  # 0 = Clear Reply
        
        stage2_models = [
            ("lgb", lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.03, class_weight="balanced", random_state=42, verbose=-1)),
            ("xgb", xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03, random_state=42, eval_metric="logloss")),
            ("cat", CatBoostClassifier(iterations=500, depth=8, learning_rate=0.03, random_state=42, verbose=False, auto_class_weights="Balanced")),
            ("rf", RandomForestClassifier(n_estimators=500, max_depth=15, class_weight="balanced", random_state=42)),
        ]
        
        self.stage2_clf = VotingClassifier(estimators=stage2_models, voting='soft')
        self.stage2_clf.fit(X_stage2, y_stage2_binary)
        
        stage2_pred = self.stage2_clf.predict(X_stage2)
        stage2_acc = accuracy_score(y_stage2_binary, stage2_pred)
        print(f"  Stage 2 Train Acc: {stage2_acc:.4f}")
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros(len(X), dtype=int)
        
        stage1_proba = self.stage1_clf.predict_proba(X_scaled)[:, 1]
        is_nonreply = stage1_proba > 0.5
        predictions[is_nonreply] = 2  # Clear Non-Reply
        
        not_nonreply_mask = ~is_nonreply
        if not_nonreply_mask.sum() > 0:
            X_rest = X_scaled[not_nonreply_mask]
            stage2_proba = self.stage2_clf.predict_proba(X_rest)[:, 1]
            is_clear_reply = stage2_proba > 0.5
            
            rest_indices = np.where(not_nonreply_mask)[0]
            predictions[rest_indices[is_clear_reply]] = 0   # Clear Reply
            predictions[rest_indices[~is_clear_reply]] = 1  # Ambivalent
        
        return predictions
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        n_samples = len(X)
        proba = np.zeros((n_samples, 3))
        
        stage1_proba = self.stage1_clf.predict_proba(X_scaled)
        proba[:, 2] = stage1_proba[:, 1]  # P(Non-Reply)
        
        rest_proba = stage1_proba[:, 0]  # P(not Non-Reply)
        stage2_proba = self.stage2_clf.predict_proba(X_scaled)
        
        proba[:, 0] = rest_proba * stage2_proba[:, 1]  # P(Clear Reply)
        proba[:, 1] = rest_proba * stage2_proba[:, 0]  # P(Ambivalent)
        
        return proba


def main():
    print("="*70)
    print("TASK 1: HIERARCHICAL CLARITY CLASSIFICATION")
    print("="*70)
    
    df = pd.read_parquet("augmented_data/task1_train_augmented.parquet")
    df = df[df["task1_label"].isin(THREE_CLASSES)].reset_index(drop=True)
    
    print(f"\nDataset size: {len(df)}")
    print(f"\nClass distribution:")
    print(df["task1_label"].value_counts())
    
    extractor = HierarchicalFeatureExtractor()
    
    print("\nExtracting features...")
    features_list = []
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"  {idx}/{len(df)}")
        feats = extractor.extract(row["question"], row["interview_answer"])
        features_list.append(feats)
    
    feature_df = pd.DataFrame(features_list)
    feature_names = feature_df.columns.tolist()
    X = feature_df.values
    
    print(f"\nTotal features: {len(feature_names)}")
    
    le = LabelEncoder()
    le.fit(THREE_CLASSES)
    y = le.transform(df["task1_label"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    print("\n" + "="*70)
    print("TRAINING HIERARCHICAL CLASSIFIER")
    print("="*70)
    
    hier_clf = HierarchicalClassifier()
    hier_clf.fit(X_train, y_train, feature_names)
    
    y_pred_hier = hier_clf.predict(X_test)
    f1_hier = f1_score(y_test, y_pred_hier, average="macro")
    acc_hier = accuracy_score(y_test, y_pred_hier)
    print(f"\nHierarchical: F1={f1_hier:.4f}, Acc={acc_hier:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING FLAT CLASSIFIERS FOR COMPARISON")
    print("="*70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    flat_models = {
        "LightGBM": lgb.LGBMClassifier(n_estimators=600, max_depth=12, learning_rate=0.03, class_weight="balanced", random_state=42, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=600, max_depth=10, learning_rate=0.03, random_state=42, eval_metric="mlogloss"),
        "CatBoost": CatBoostClassifier(iterations=600, depth=10, learning_rate=0.03, random_state=42, verbose=False, auto_class_weights="Balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=600, max_depth=18, class_weight="balanced", random_state=42),
    }
    
    results = {"Hierarchical": {"f1": f1_hier, "acc": acc_hier, "pred": y_pred_hier}}
    trained_models = {}
    
    for name, model in flat_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
        results[name] = {"f1": f1, "acc": acc, "pred": y_pred}
        trained_models[name] = model
    
    print("\n--- Flat Ensemble ---")
    flat_estimators = [(n, m) for n, m in flat_models.items()]
    flat_vote = VotingClassifier(estimators=flat_estimators, voting='soft')
    flat_vote.fit(X_train_scaled, y_train)
    y_pred = flat_vote.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Flat Ensemble"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n--- Combined (Hier + Flat avg) ---")
    hier_proba = hier_clf.predict_proba(X_test)
    flat_proba = flat_vote.predict_proba(X_test_scaled)
    combined_proba = 0.4 * hier_proba + 0.6 * flat_proba
    y_pred_combined = np.argmax(combined_proba, axis=1)
    f1_comb = f1_score(y_test, y_pred_combined, average="macro")
    acc_comb = accuracy_score(y_test, y_pred_combined)
    print(f"  F1: {f1_comb:.4f} | Acc: {acc_comb:.4f}")
    results["Combined"] = {"f1": f1_comb, "acc": acc_comb, "pred": y_pred_combined}
    
    print("\n" + "="*70)
    print("FINAL RANKING")
    print("="*70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, res) in enumerate(sorted_results):
        print(f"{i+1}. {name}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}")
    
    best_name = sorted_results[0][0]
    best_pred = sorted_results[0][1]["pred"]
    
    print("\n" + "="*70)
    print(f"BEST: {best_name}")
    print("="*70)
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "hier_clf": hier_clf,
        "flat_models": trained_models,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
    }, "models/task1_hierarchical.joblib")
    print("\nModel saved to models/task1_hierarchical.joblib")


if __name__ == "__main__":
    main()


