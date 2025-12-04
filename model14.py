import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import Counter
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
FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]


class ComprehensiveFeatureExtractor:
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
        features["starts_well"] = int(first_word == "well" or a_lower.startswith("well,"))
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
        features["a_entity_count"] = len(a_entities)
        
        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        first_sent_lower = first_sent.lower()
        features["first_sent_has_q_noun"] = int(any(n in first_sent_lower for n in q_nouns))
        
        certainty_modals = ["will", "must", "shall", "can"]
        uncertainty_modals = ["would", "could", "might", "may", "should"]
        features["certainty_modal_count"] = sum(1 for m in certainty_modals if m in a_words)
        features["uncertainty_modal_count"] = sum(1 for m in uncertainty_modals if m in a_words)
        
        features["short_sentences"] = sum(1 for s in sentences if len(s.split()) < 10)
        features["sentence_count"] = len(sentences)
        features["avg_sentence_len"] = len(a_words) / max(len(sentences), 1)
        
        hesitation = ["i mean", "you know", "sort of", "kind of", "basically", "actually", "honestly"]
        features["has_hesitation"] = int(any(h in a_lower for h in hesitation))
        features["hesitation_count"] = sum(1 for h in hesitation if h in a_lower)
        
        comparatives = ["but", "however", "although", "though", "whereas", "nevertheless", "yet"]
        features["has_contrast"] = int(any(c in a_lower for c in comparatives))
        features["contrast_count"] = sum(1 for c in comparatives if c in a_lower)
        
        features["ends_with_question"] = int(a.strip().endswith("?"))
        features["has_question_in_answer"] = int("?" in a)
        features["num_count"] = len(re.findall(r'\d+', a))
        features["has_numbers"] = int(features["num_count"] > 0)
        
        commitment = ["promise", "commit", "guarantee", "ensure", "definitely", "certainly", "absolutely"]
        features["commitment_count"] = sum(1 for c in commitment if c in a_lower)
        
        features["present_count"] = sum(1 for w in ["is", "are", "am", "now", "today", "currently"] if w in a_words)
        features["specific_quantity_count"] = len(re.findall(r'\b\d+\b', a))
        features["has_conditional"] = int(any(c in a_lower for c in ["if ", "unless ", "whether "]))
        
        negations = ["not", "no", "never", "nothing", "nobody", "don't", "doesn't", "didn't", "won't", "can't"]
        features["negation_count"] = sum(1 for n in negations if n in a_words)
        
        causal = ["because", "since", "therefore", "thus", "hence", "as a result", "due to"]
        features["causal_count"] = sum(1 for c in causal if c in a_lower)
        
        pivot_phrases = ["let me", "the fact is", "the truth is", "what matters", "first of all", "look,"]
        features["has_pivot"] = int(any(p in a_lower for p in pivot_phrases))
        
        vague_words = ["something", "things", "stuff", "everything", "anything", "people", "everyone"]
        features["vague_count"] = sum(1 for v in vague_words if v in a_words)
        
        features["i_count"] = sum(1 for w in a_words if w == "i")
        features["we_count"] = sum(1 for w in a_words if w == "we")
        features["you_count"] = sum(1 for w in a_words if w == "you")
        
        return features


def load_augmented_data():
    df = pd.read_parquet("augmented_data/train_augmented.parquet")
    df = df[df["label"].isin(FOUR_CLASSES)].reset_index(drop=True)
    return df


def get_feature_importance(model, n_features):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_).mean(axis=0)
    else:
        return np.ones(n_features) / n_features


def main():
    print("="*70)
    print("MODEL WITH AUGMENTED DATA + PER-MODEL FEATURE SELECTION")
    print("="*70)
    
    df = load_augmented_data()
    print(f"\nDataset size: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
    extractor = ComprehensiveFeatureExtractor()
    
    print("Extracting features...")
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
    le.classes_ = np.array(FOUR_CLASSES)
    y = le.transform(df["label"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("PER-MODEL FEATURE SELECTION + TRAINING")
    print("="*70)
    
    model_configs = {
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        "CatBoost": lambda: CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, random_state=42, verbose=0),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        "Logistic Regression": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }
    
    trained_models = {}
    model_features = {}
    results = {}
    
    for model_name, model_fn in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        temp_model = model_fn()
        temp_model.fit(X_train_scaled, y_train)
        importances = get_feature_importance(temp_model, len(feature_names))
        
        top_k = min(40, len(feature_names))
        top_indices = np.argsort(importances)[::-1][:top_k]
        selected_features = [feature_names[i] for i in top_indices]
        
        X_train_selected = X_train_scaled[:, top_indices]
        X_test_selected = X_test_scaled[:, top_indices]
        
        model = model_fn()
        model.fit(X_train_selected, y_train)
        
        y_pred = model.predict(X_test_selected)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  Features: {top_k}")
        print(f"  Test F1: {f1:.4f} | Acc: {acc:.4f}")
        print(f"  Top features: {selected_features[:5]}")
        
        trained_models[model_name] = model
        model_features[model_name] = top_indices
        results[model_name] = {"f1": f1, "acc": acc, "pred": y_pred, "features": top_indices}
    
    print("\n" + "="*70)
    print("ENSEMBLE MODELS")
    print("="*70)
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    top5_names = [name for name, _ in sorted_models[:5]]
    print(f"\nTop 5 models: {top5_names}")
    
    all_top_indices = set()
    for name in top5_names:
        all_top_indices.update(model_features[name])
    all_top_indices = sorted(list(all_top_indices))
    
    X_train_ensemble = X_train_scaled[:, all_top_indices]
    X_test_ensemble = X_test_scaled[:, all_top_indices]
    
    ensemble_estimators = []
    for name in top5_names:
        model = model_configs[name]()
        model.fit(X_train_ensemble, y_train)
        ensemble_estimators.append((name, model))
    
    print("\n--- Soft Voting Ensemble ---")
    soft_voting = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    soft_voting.fit(X_train_ensemble, y_train)
    y_pred_soft = soft_voting.predict(X_test_ensemble)
    f1_soft = f1_score(y_test, y_pred_soft, average="macro")
    acc_soft = accuracy_score(y_test, y_pred_soft)
    print(f"  F1: {f1_soft:.4f} | Acc: {acc_soft:.4f}")
    results["Soft Voting"] = {"f1": f1_soft, "acc": acc_soft, "pred": y_pred_soft}
    
    print("\n--- Hard Voting Ensemble ---")
    hard_voting = VotingClassifier(estimators=ensemble_estimators, voting='hard')
    hard_voting.fit(X_train_ensemble, y_train)
    y_pred_hard = hard_voting.predict(X_test_ensemble)
    f1_hard = f1_score(y_test, y_pred_hard, average="macro")
    acc_hard = accuracy_score(y_test, y_pred_hard)
    print(f"  F1: {f1_hard:.4f} | Acc: {acc_hard:.4f}")
    results["Hard Voting"] = {"f1": f1_hard, "acc": acc_hard, "pred": y_pred_hard}
    
    print("\n--- Average Probability Ensemble ---")
    probas = []
    for name in top5_names:
        model = trained_models[name]
        idx = model_features[name]
        proba = model.predict_proba(X_test_scaled[:, idx])
        probas.append(proba)
    avg_proba = np.mean(probas, axis=0)
    y_pred_avg = np.argmax(avg_proba, axis=1)
    f1_avg = f1_score(y_test, y_pred_avg, average="macro")
    acc_avg = accuracy_score(y_test, y_pred_avg)
    print(f"  F1: {f1_avg:.4f} | Acc: {acc_avg:.4f}")
    results["Avg Probability"] = {"f1": f1_avg, "acc": acc_avg, "pred": y_pred_avg}
    
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
    print(classification_report(y_test, best_pred, target_names=FOUR_CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "trained_models": trained_models,
        "model_features": model_features,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
        "results": {k: {"f1": v["f1"], "acc": v["acc"]} for k, v in results.items()}
    }, "models/model14_augmented.joblib")
    print("\nModel saved to models/model14_augmented.joblib")


if __name__ == "__main__":
    main()
