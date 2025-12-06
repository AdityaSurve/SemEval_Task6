import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
import optuna
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)
nlp = spacy.load("en_core_web_sm")

THREE_CLASSES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


class EmbeddingFeatureExtractor:
    def __init__(self):
        print("Loading Sentence Transformer (mpnet-base - higher quality)...")
        self.sbert = SentenceTransformer("all-mpnet-base-v2")  # Better model, 768d
        
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
        
        semantic_sim = 1 - cosine(q_emb, a_emb) if np.any(q_emb) and np.any(a_emb) else 0
        
        first_sent = a.split('.')[0] if '.' in a else a[:150]
        first_emb = self.sbert.encode(first_sent)
        first_sent_sim = 1 - cosine(q_emb, first_emb) if np.any(first_emb) else 0
        
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
        
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ", "will ", "would ", "could ", "should ", "can "]
        ling_features["is_yes_no_q"] = int(any(q_lower.startswith(s) for s in yes_no_starters))
        ling_features["is_wh_q"] = int(any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which"]))
        
        direct_yes = {"yes", "yeah", "yep", "absolutely", "definitely", "certainly", "correct", "exactly", "right", "sure", "of course"}
        direct_no = {"no", "nope", "nah", "never", "not"}
        first_word = a_words[0] if a_words else ""
        first_5 = set(a_words[:5]) if len(a_words) >= 5 else set(a_words)
        
        ling_features["starts_yes"] = int(first_word in direct_yes or len(first_5 & direct_yes) > 0)
        ling_features["starts_no"] = int(first_word in direct_no)
        ling_features["starts_direct"] = int(ling_features["starts_yes"] or ling_features["starts_no"])
        ling_features["starts_well"] = int(first_word == "well")
        ling_features["starts_i"] = int(first_word == "i")
        
        ling_features["yes_no_q_direct_a"] = int(ling_features["is_yes_no_q"] and ling_features["starts_direct"])
        
        q_doc = nlp(q_lower[:500])
        a_doc = nlp(a_lower[:2000])
        
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha}
        ling_features["lexical_overlap"] = len(q_content & a_content) / max(len(q_content), 1) if q_content else 0
        
        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        first_sent_lower = first_sent.lower()
        ling_features["first_sent_has_q_noun"] = int(any(n in first_sent_lower for n in q_nouns))
        
        ling_features["sentence_count"] = len(sentences)
        
        refusal_phrases = ["i can't", "i cannot", "i won't", "no comment", "i'm not going to", 
                          "i don't know", "not prepared to", "decline to"]
        ling_features["has_refusal"] = int(any(r in a_lower for r in refusal_phrases))
        
        hedges = ["maybe", "perhaps", "possibly", "probably", "might", "could", "somewhat"]
        ling_features["hedge_count"] = sum(1 for h in hedges if h in a_lower)
        
        commitment = ["i will", "i'm going to", "we will", "we're going to", "i promise"]
        ling_features["commitment_count"] = sum(1 for c in commitment if c in a_lower)
        
        ling_features["has_question_in_answer"] = int("?" in a)
        
        return {
            "q_emb": q_emb,
            "a_emb": a_emb,
            "combined_emb": combined_emb,
            "diff_emb": diff_emb,
            "ling": ling_features
        }


def main():
    print("="*70)
    print("TASK 1: EMBEDDING + LINGUISTIC FEATURES")
    print("="*70)
    
    aug_path = "augmented_data/task1_train_augmented.parquet"
    if os.path.exists(aug_path):
        print("Using augmented data...")
        df = pd.read_parquet(aug_path)
        label_col = "task1_label"
    else:
        print("Using original data...")
        raw_path = "dataset/data/train-00000-of-00001.parquet"
        df = pd.read_parquet(raw_path)
        df["task1_label"] = df["clarity_label"]
        label_col = "task1_label"
        if "question" not in df.columns:
            df["question"] = df["interview_question"]
    
    df = df[df[label_col].isin(THREE_CLASSES)].reset_index(drop=True)
    
    print(f"\nDataset size: {len(df)}")
    print(f"\nClass distribution:")
    print(df[label_col].value_counts())
    
    extractor = EmbeddingFeatureExtractor()
    
    print("\nExtracting features...")
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
    
    print(f"\nQ embeddings shape: {Q_emb.shape}")
    print(f"A embeddings shape: {A_emb.shape}")
    print(f"Linguistic features: {ling_df.shape[1]}")
    
    le = LabelEncoder()
    le.fit(THREE_CLASSES)
    y = le.transform(df[label_col])
    
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(df)), y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train_idx)}, Test: {len(X_test_idx)}")
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT FEATURE COMBINATIONS")
    print("="*70)
    
    results = {}
    
    print("\n--- Option 1: Linguistic features only ---")
    X_ling = ling_df.values
    scaler_ling = StandardScaler()
    X_train_ling = scaler_ling.fit_transform(X_ling[X_train_idx])
    X_test_ling = scaler_ling.transform(X_ling[X_test_idx])
    
    clf_ling = lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)
    clf_ling.fit(X_train_ling, y_train)
    y_pred = clf_ling.predict(X_test_ling)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Linguistic only"] = {"f1": f1, "acc": acc}
    
    print("\n--- Option 2: Q+A embeddings (1536d) + PCA ---")
    X_emb_qa = Combined_emb
    pca_qa = PCA(n_components=150)  # More components for larger embeddings
    X_emb_qa_pca = pca_qa.fit_transform(X_emb_qa)
    scaler_qa = StandardScaler()
    X_train_qa = scaler_qa.fit_transform(X_emb_qa_pca[X_train_idx])
    X_test_qa = scaler_qa.transform(X_emb_qa_pca[X_test_idx])
    
    clf_qa = lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)
    clf_qa.fit(X_train_qa, y_train)
    y_pred = clf_qa.predict(X_test_qa)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Q+A embeddings"] = {"f1": f1, "acc": acc}
    
    print("\n--- Option 3: Diff embeddings (768d) + PCA ---")
    pca_diff = PCA(n_components=100)  # More components for larger embeddings
    X_diff_pca = pca_diff.fit_transform(Diff_emb)
    scaler_diff = StandardScaler()
    X_train_diff = scaler_diff.fit_transform(X_diff_pca[X_train_idx])
    X_test_diff = scaler_diff.transform(X_diff_pca[X_test_idx])
    
    clf_diff = lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)
    clf_diff.fit(X_train_diff, y_train)
    y_pred = clf_diff.predict(X_test_diff)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Diff embeddings"] = {"f1": f1, "acc": acc}
    
    print("\n--- Option 4: Embeddings + Linguistic (COMBINED) ---")
    X_combined = np.hstack([X_emb_qa_pca, X_diff_pca, ling_df.values])
    scaler_comb = StandardScaler()
    X_train_comb = scaler_comb.fit_transform(X_combined[X_train_idx])
    X_test_comb = scaler_comb.transform(X_combined[X_test_idx])
    
    clf_comb = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)
    clf_comb.fit(X_train_comb, y_train)
    y_pred = clf_comb.predict(X_test_comb)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Combined (Emb+Ling)"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n--- Option 5: Combined + Ensemble ---")
    models = {
        "LightGBM": lgb.LGBMClassifier(n_estimators=600, max_depth=10, learning_rate=0.03, class_weight='balanced', random_state=42, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.03, random_state=42, eval_metric='mlogloss'),
        "CatBoost": CatBoostClassifier(iterations=600, depth=8, learning_rate=0.03, random_state=42, verbose=False, auto_class_weights="Balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train_comb, y_train)
        trained[name] = model
    
    estimators = [(n, m) for n, m in trained.items()]
    vote_clf = VotingClassifier(estimators=estimators, voting='soft')
    vote_clf.fit(X_train_comb, y_train)
    y_pred = vote_clf.predict(X_test_comb)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f}")
    results["Combined + Ensemble"] = {"f1": f1, "acc": acc, "pred": y_pred}
    
    print("\n--- Option 6: Stacking ---")
    stack_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=500),
        cv=5
    )
    stack_clf.fit(X_train_comb, y_train)
    y_pred = stack_clf.predict(X_test_comb)
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
    if "pred" in sorted_results[0][1]:
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
        "vote_clf": vote_clf,
        "stack_clf": stack_clf,
        "scaler": scaler_comb,
        "pca_qa": pca_qa,
        "pca_diff": pca_diff,
        "label_encoder": le,
    }, "models/task1_embeddings.joblib")
    print("\nModel saved to models/task1_embeddings.joblib")


if __name__ == "__main__":
    main()
