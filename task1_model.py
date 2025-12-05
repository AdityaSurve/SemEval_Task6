import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import spacy
import joblib
import warnings
import re
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

THREE_CLASSES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


class ClarityFeatureExtractor:
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
        features["a_q_ratio"] = len(a_words) / max(len(q_words), 1)
        features["q_has_multiple_questions"] = int(q.count("?") > 1)
        features["q_has_and"] = int(" and " in q_lower)
        features["q_is_complex"] = int(len(q_words) > 20 or features["q_has_and"] or features["q_has_multiple_questions"])
        
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
        
        features["yes_no_q_direct_a"] = int(features["is_yes_no_q"] and features["starts_direct"])
        features["yes_no_q_indirect_a"] = int(features["is_yes_no_q"] and not features["starts_direct"])
        
        q_doc = nlp(q_lower[:500])
        a_doc = nlp(a_lower[:2000])
        
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        features["lexical_overlap"] = len(q_content & a_content) / max(len(q_content), 1) if q_content else 0
        features["lexical_overlap_count"] = len(q_content & a_content)
        
        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["entity_overlap"] = len(q_entities & a_entities) / max(len(q_entities), 1) if q_entities else 0
        features["q_entity_count"] = len(q_entities)
        features["a_entity_count"] = len(a_entities)
        
        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        first_sent_lower = first_sent.lower()
        features["first_sent_has_q_noun"] = int(any(n in first_sent_lower for n in q_nouns))
        
        uncertainty_modals = ["would", "could", "might", "may", "should"]
        features["uncertainty_modal_count"] = sum(1 for m in uncertainty_modals if m in a_words)
        
        features["sentence_count"] = len(sentences)
        features["avg_sentence_len"] = len(a_words) / max(len(sentences), 1)
        
        hesitation = ["i mean", "you know", "sort of", "kind of", "basically", "actually", "honestly"]
        features["has_hesitation"] = int(any(h in a_lower for h in hesitation))
        features["hesitation_count"] = sum(1 for h in hesitation if h in a_lower)
        
        comparatives = ["but", "however", "although", "though", "whereas", "nevertheless", "yet"]
        features["has_contrast"] = int(any(c in a_lower for c in comparatives))
        features["contrast_count"] = sum(1 for c in comparatives if c in a_lower)
        
        features["has_question_in_answer"] = int("?" in a)
        features["num_count"] = len(re.findall(r'\d+', a))
        features["has_numbers"] = int(features["num_count"] > 0)
        
        pivot_phrases = ["let me", "the fact is", "the truth is", "what matters", "first of all", "look,", "the real question"]
        features["has_pivot"] = int(any(p in a_lower for p in pivot_phrases))
        features["pivot_count"] = sum(1 for p in pivot_phrases if p in a_lower)
        
        negations = ["not", "no", "never", "nothing", "nobody", "don't", "doesn't", "didn't", "won't", "can't"]
        features["negation_count"] = sum(1 for n in negations if n in a_words)
        
        features["i_count"] = sum(1 for w in a_words if w == "i")
        features["we_count"] = sum(1 for w in a_words if w == "we")
        
        hedges = ["maybe", "perhaps", "possibly", "probably", "might", "could", "somewhat", "relatively"]
        features["hedge_count"] = sum(1 for h in hedges if h in a_words)
        features["hedge_ratio"] = features["hedge_count"] / max(len(a_words), 1)
        
        features["clear_reply_score"] = (
            features["semantic_sim"] * 10 +
            features["starts_direct"] * 5 +
            features["lexical_overlap"] * 5 +
            features["first_sent_has_q_noun"] * 3 -
            features["hedge_count"] * 2 -
            features["hesitation_count"] * 2
        )
        
        features["non_reply_score"] = (
            (1 - features["semantic_sim"]) * 10 +
            features["has_pivot"] * 5 +
            features["topic_drift"] * 5 -
            features["starts_direct"] * 5 -
            features["lexical_overlap"] * 5
        )
        
        features["ambivalent_score"] = (
            features["hedge_count"] * 3 +
            features["hesitation_count"] * 3 +
            features["uncertainty_modal_count"] * 2 +
            features["has_contrast"] * 2 -
            abs(features["semantic_sim"] - 0.5) * 5
        )
        
        return features


def main():
    print("="*70)
    print("TASK 1: CLARITY CLASSIFICATION MODEL")
    print("="*70)
    
    df = pd.read_parquet("augmented_data/task1_train_augmented.parquet")
    df = df[df["task1_label"].isin(THREE_CLASSES)].reset_index(drop=True)
    
    print(f"\nDataset size: {len(df)}")
    print(f"\nClass distribution:")
    print(df["task1_label"].value_counts())
    
    extractor = ClarityFeatureExtractor()
    
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("="*70)
    
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 700),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
        return scores.mean()
    
    print("\nTuning LightGBM...")
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_lgb.optimize(objective_lgb, n_trials=25, show_progress_bar=True)
    best_lgb_params = study_lgb.best_params
    print(f"Best LightGBM F1: {study_lgb.best_value:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    models = {
        "LightGBM (tuned)": lgb.LGBMClassifier(**best_lgb_params, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        "CatBoost": CatBoostClassifier(iterations=500, depth=6, learning_rate=0.03, random_state=42, verbose=False, auto_class_weights="Balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
        results[name] = {"f1": f1, "acc": acc}
        trained_models[name] = model
    
    print("\n" + "="*70)
    print("ENSEMBLE METHODS")
    print("="*70)
    
    top3 = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)[:3]
    top3_names = [n for n, _ in top3]
    print(f"Top 3 for ensemble: {top3_names}")
    
    estimators = [(name, trained_models[name]) for name in top3_names]
    
    print("\n--- Soft Voting ---")
    soft_vote = VotingClassifier(estimators=estimators, voting='soft')
    soft_vote.fit(X_train_scaled, y_train)
    y_pred = soft_vote.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Soft Voting"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n--- Hard Voting ---")
    hard_vote = VotingClassifier(estimators=estimators, voting='hard')
    hard_vote.fit(X_train_scaled, y_train)
    y_pred = hard_vote.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Hard Voting"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n--- Stacking ---")
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    stack.fit(X_train_scaled, y_train)
    y_pred = stack.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Stacking"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n" + "="*70)
    print("FINAL RANKING")
    print("="*70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, res) in enumerate(sorted_results):
        print(f"{i+1}. {name}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}")
    
    best_name = sorted_results[0][0]
    best_pred = results[best_name].get("pred")
    
    if best_pred is None:
        best_pred = trained_models[best_name].predict(X_test_scaled)
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_name}")
    print("="*70)
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "trained_models": trained_models,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
    }, "models/task1_model.joblib")
    print("\nModel saved to models/task1_model.joblib")


if __name__ == "__main__":
    main()

