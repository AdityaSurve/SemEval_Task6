import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import spacy
import joblib
import optuna
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)
nlp = spacy.load("en_core_web_sm")

THREE_CLASSES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


class CleanFeatureExtractor:
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
            mid_idx = len(sentences) // 2
            mid_emb = self.sbert.encode(sentences[mid_idx])
            features["mid_sent_sim"] = 1 - cosine(q_emb, mid_emb)
        else:
            features["last_sent_sim"] = features["first_sent_sim"]
            features["mid_sent_sim"] = features["first_sent_sim"]
        
        features["sim_consistency"] = 1 - np.std([features["first_sent_sim"], features["mid_sent_sim"], features["last_sent_sim"]])
        
        features["q_len_words"] = len(q_words)
        features["a_len_words"] = len(a_words)
        features["a_len_chars"] = len(a)
        features["a_q_ratio"] = len(a_words) / max(len(q_words), 1)
        features["a_is_short"] = int(len(a_words) < 80)
        features["a_is_very_short"] = int(len(a_words) < 40)
        
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ", "will ", "would ", "could ", "should ", "can "]
        features["is_yes_no_q"] = int(any(q_lower.startswith(s) for s in yes_no_starters))
        features["is_wh_q"] = int(any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which"]))
        
        direct_yes = {"yes", "yeah", "yep", "absolutely", "definitely", "certainly", "correct", "exactly", "right", "sure", "of course"}
        direct_no = {"no", "nope", "nah", "never", "not"}
        first_word = a_words[0] if a_words else ""
        first_5 = set(a_words[:5]) if len(a_words) >= 5 else set(a_words)
        
        features["starts_yes"] = int(first_word in direct_yes or len(first_5 & direct_yes) > 0)
        features["starts_no"] = int(first_word in direct_no)
        features["starts_direct"] = int(features["starts_yes"] or features["starts_no"])
        features["starts_well"] = int(first_word == "well" or a_lower.startswith("well,"))
        features["starts_i"] = int(first_word == "i")
        features["starts_look"] = int(first_word in ["look", "listen"])
        
        features["yes_no_q_direct_a"] = int(features["is_yes_no_q"] and features["starts_direct"])
        
        q_doc = nlp(q_lower[:500])
        a_doc = nlp(a_lower[:2000])
        
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        features["lexical_overlap"] = len(q_content & a_content) / max(len(q_content), 1) if q_content else 0
        
        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        first_sent_lower = first_sent.lower()
        features["first_sent_has_q_noun"] = int(any(n in first_sent_lower for n in q_nouns))
        features["q_noun_coverage"] = sum(1 for n in q_nouns if n in a_lower) / max(len(q_nouns), 1)
        
        features["sentence_count"] = len(sentences)
        features["avg_sentence_len"] = len(a_words) / max(len(sentences), 1)
        
        refusal_phrases = ["i can't", "i cannot", "i won't", "i will not", "no comment", "i'm not going to", 
                          "i don't know", "i have no", "not prepared to", "decline to", "not appropriate"]
        features["refusal_count"] = sum(1 for r in refusal_phrases if r in a_lower)
        features["has_refusal"] = int(features["refusal_count"] > 0)
        
        clarify_phrases = ["what do you mean", "could you clarify", "i'm not sure what", "which", "are you asking"]
        features["asks_clarification"] = int(any(c in a_lower for c in clarify_phrases))
        
        ignorance_phrases = ["i don't know", "i'm not aware", "i haven't", "i didn't", "no idea", "not sure"]
        features["claims_ignorance"] = int(any(i in a_lower for i in ignorance_phrases))
        
        hedges = ["maybe", "perhaps", "possibly", "probably", "might", "could", "somewhat", "kind of", "sort of"]
        features["hedge_count"] = sum(1 for h in hedges if h in a_lower)
        
        uncertainty_modals = ["would", "could", "might", "may", "should"]
        features["modal_count"] = sum(1 for m in uncertainty_modals if m in a_words)
        
        commitment = ["i will", "i'm going to", "we will", "we're going to", "i am committed", "i promise"]
        features["commitment_count"] = sum(1 for c in commitment if c in a_lower)
        
        deflection = ["let me tell you", "the real question", "what's important", "what matters"]
        features["deflection_count"] = sum(1 for d in deflection if d in a_lower)
        
        features["i_count"] = sum(1 for w in a_words if w == "i")
        features["we_count"] = sum(1 for w in a_words if w == "we")
        features["first_person_ratio"] = (features["i_count"] + features["we_count"]) / max(len(a_words), 1)
        
        features["has_question_in_answer"] = int("?" in a)
        features["ends_with_question"] = int(a.strip().endswith("?"))
        
        features["clear_reply_score"] = (
            features["semantic_sim"] * 2 +
            features["starts_direct"] * 3 +
            features["first_sent_has_q_noun"] * 2 +
            features["lexical_overlap"] * 2 +
            features["commitment_count"] -
            features["hedge_count"] * 0.5
        )
        
        features["non_reply_score"] = (
            features["has_refusal"] * 3 +
            features["claims_ignorance"] * 2 +
            features["asks_clarification"] * 2 +
            features["a_is_very_short"] * 2 +
            (1 - features["semantic_sim"]) * 2 +
            features["ends_with_question"]
        )
        
        features["ambivalent_score"] = (
            features["hedge_count"] +
            features["modal_count"] * 0.5 +
            features["deflection_count"] * 2 +
            (1 - features["starts_direct"]) +
            (1 - features["first_sent_has_q_noun"])
        )
        
        return features


def main():
    print("="*70)
    print("TASK 1: CLEAN CLARITY CLASSIFICATION (NO AUGMENTATION)")
    print("="*70)
    
    raw_path = "dataset/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(raw_path)
    
    label_mapping = {
        "Clear Reply": "Clear Reply",
        "Ambivalent": "Ambivalent",
        "Clear Non-Reply": "Clear Non-Reply"
    }
    df["task1_label"] = df["clarity_label"].map(label_mapping)
    df = df[df["task1_label"].isin(THREE_CLASSES)].reset_index(drop=True)
    
    if "question" not in df.columns:
        df["question"] = df["interview_question"]
    
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"\nClass distribution:")
    print(df["task1_label"].value_counts())
    
    extractor = CleanFeatureExtractor()
    
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
    print("OPTUNA TUNING (LightGBM)")
    print("="*70)
    
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        }
        model = lgb.LGBMClassifier(**params, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        return scores.mean()
    
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=40, show_progress_bar=True)
    best_lgb_params = study_lgb.best_params
    print(f"Best LightGBM CV F1: {study_lgb.best_value:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    models = {
        "LightGBM (tuned)": lgb.LGBMClassifier(**best_lgb_params, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'),
        "CatBoost": CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3, random_state=42, verbose=False, auto_class_weights="Balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
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
        results[name] = {"f1": f1, "acc": acc, "pred": y_pred}
        trained_models[name] = model
    
    print("\n--- Soft Voting Ensemble ---")
    top_models = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)[:4]
    estimators = [(n, trained_models[n]) for n, _ in top_models]
    vote_clf = VotingClassifier(estimators=estimators, voting='soft')
    vote_clf.fit(X_train_scaled, y_train)
    y_pred = vote_clf.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Voting Ensemble"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n--- Stacking Ensemble ---")
    stack_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=500),
        cv=5
    )
    stack_clf.fit(X_train_scaled, y_train)
    y_pred = stack_clf.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Stacking Ensemble"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n" + "="*70)
    print("FINAL RANKING")
    print("="*70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, res) in enumerate(sorted_results):
        print(f"{i+1}. {name}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}")
    
    best_name = sorted_results[0][0]
    best_pred = sorted_results[0][1]["pred"]
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_name}")
    print("="*70)
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "models": trained_models,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
    }, "models/task1_clean.joblib")
    print("\nModel saved to models/task1_clean.joblib")


if __name__ == "__main__":
    main()
