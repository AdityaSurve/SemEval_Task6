import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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

NINE_CLASSES = [
    "Explicit", "Dodging", "Implicit", "General", "Deflection",
    "Declining to answer", "Claims ignorance", "Clarification", "Partial/half-answer"
]


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
        features["q_question_count"] = q.count("?")
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
        features["starts_i_dont_know"] = int(a_lower.startswith("i don't know") or a_lower.startswith("i do not know"))
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
        
        features["has_question_in_answer"] = int("?" in a)
        features["question_in_answer_count"] = a.count("?")
        features["num_count"] = len(re.findall(r'\d+', a))
        features["has_numbers"] = int(features["num_count"] > 0)
        
        commitment = ["promise", "commit", "guarantee", "ensure", "definitely", "certainly", "absolutely"]
        features["commitment_count"] = sum(1 for c in commitment if c in a_lower)
        
        ignorance_phrases = ["i don't know", "i do not know", "i'm not sure", "i am not sure", "not certain", "can't say", "cannot say", "no idea"]
        features["claims_ignorance"] = int(any(p in a_lower for p in ignorance_phrases))
        features["ignorance_count"] = sum(1 for p in ignorance_phrases if p in a_lower)
        
        decline_phrases = ["i can't answer", "i cannot answer", "i won't", "i will not", "decline to", "refuse to", "not going to answer", "no comment"]
        features["declines"] = int(any(p in a_lower for p in decline_phrases))
        features["decline_count"] = sum(1 for p in decline_phrases if p in a_lower)
        
        deflection_phrases = ["you should ask", "that's a question for", "talk to", "ask them", "they would know", "not my", "someone else"]
        features["deflects"] = int(any(p in a_lower for p in deflection_phrases))
        features["deflection_count"] = sum(1 for p in deflection_phrases if p in a_lower)
        
        clarify_phrases = ["what do you mean", "could you clarify", "can you explain", "i'm not sure what", "which", "are you asking"]
        features["asks_clarification"] = int(any(p in a_lower for p in clarify_phrases))
        
        implicit_markers = ["implies", "suggests", "indicates", "means", "essentially", "basically", "in other words"]
        features["has_implicit_markers"] = int(any(m in a_lower for m in implicit_markers))
        
        features["has_conditional"] = int(any(c in a_lower for c in ["if ", "unless ", "whether "]))
        
        negations = ["not", "no", "never", "nothing", "nobody", "don't", "doesn't", "didn't", "won't", "can't"]
        features["negation_count"] = sum(1 for n in negations if n in a_words)
        
        pivot_phrases = ["let me", "the fact is", "the truth is", "what matters", "first of all", "look,", "the real question"]
        features["has_pivot"] = int(any(p in a_lower for p in pivot_phrases))
        features["pivot_count"] = sum(1 for p in pivot_phrases if p in a_lower)
        
        features["i_count"] = sum(1 for w in a_words if w == "i")
        features["we_count"] = sum(1 for w in a_words if w == "we")
        features["you_count"] = sum(1 for w in a_words if w == "you")
        features["they_count"] = sum(1 for w in a_words if w == "they")
        
        return features


def main():
    print("="*70)
    print("9-CLASS MODEL WITH AUGMENTED DATA")
    print("="*70)
    
    df = pd.read_parquet("augmented_data/train_augmented_9class.parquet")
    df = df[df["label"].isin(NINE_CLASSES)].reset_index(drop=True)
    
    print(f"\nDataset size: {len(df)}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts())
    
    extractor = ComprehensiveFeatureExtractor()
    
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
    le.fit(NINE_CLASSES)
    y = le.transform(df["label"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    models = {
        "LightGBM": lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.03, num_leaves=60, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        "CatBoost": CatBoostClassifier(iterations=500, depth=8, learning_rate=0.03, random_state=42, verbose=False, auto_class_weights="Balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
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
    
    top4 = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)[:4]
    top4_names = [n for n, _ in top4]
    print(f"Top 4 for ensemble: {top4_names}")
    
    estimators = [(name, trained_models[name]) for name in top4_names]
    
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
    
    print("\n--- Stacking (LogReg meta) ---")
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
    }, "models/model16_9class.joblib")
    print("\nModel saved to models/model16_9class.joblib")


if __name__ == "__main__":
    main()

