# Feature-engineered model for CLARITY task

import os
import re
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import spacy

nlp = spacy.load("en_core_web_sm")

class ClarityFeatureExtractor:
    def __init__(self):
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.hedge_words = {
            "maybe", "perhaps", "possibly", "probably", "might", "could",
            "would", "should", "somewhat", "relatively", "fairly", "rather",
            "kind of", "sort of", "in a way", "to some extent", "arguably"
        }
        self.deflection_phrases = [
            "let me tell you", "the real question is", "what matters is",
            "the important thing", "i think what you're really asking",
            "look,", "listen,", "here's the thing", "the fact is",
            "what i would say", "i'll tell you what", "you know what"
        ]
        self.evasion_starters = [
            "well,", "so,", "you know,", "i mean,", "actually,",
            "to be honest", "frankly", "honestly"
        ]
        self.direct_indicators = {
            "yes", "no", "absolutely", "definitely", "certainly",
            "of course", "exactly", "correct", "right", "wrong"
        }
        self.wh_words = {"what", "who", "where", "when", "why", "how", "which"}

    def extract_features(self, question: str, answer: str) -> dict:
        q_doc = nlp(question.lower())
        a_doc = nlp(answer.lower())
        features = {}
        features["q_length_chars"] = len(question)
        features["a_length_chars"] = len(answer)
        features["q_length_words"] = len(q_doc)
        features["a_length_words"] = len(a_doc)
        features["a_q_length_ratio"] = len(a_doc) / max(len(q_doc), 1)
        features["q_num_sentences"] = len(list(q_doc.sents))
        features["a_num_sentences"] = len(list(a_doc.sents))
        features["is_yes_no_q"] = self._is_yes_no_question(question)
        features["is_wh_q"] = any(token.text in self.wh_words for token in q_doc)
        features["q_has_or"] = " or " in question.lower()

        q_lemmas = {token.lemma_ for token in q_doc if not token.is_stop and not token.is_punct}
        a_lemmas = {token.lemma_ for token in a_doc if not token.is_stop and not token.is_punct}
        overlap = q_lemmas & a_lemmas
        features["lexical_overlap"] = len(overlap) / max(len(q_lemmas), 1)
        features["q_coverage"] = len(overlap) / max(len(q_lemmas), 1)

        q_emb = self.sbert.encode(question, convert_to_numpy=True)
        a_emb = self.sbert.encode(answer, convert_to_numpy=True)
        features["semantic_similarity"] = 1 - cosine(q_emb, a_emb)

        a_lower = answer.lower()
        features["hedge_count"] = sum(1 for h in self.hedge_words if h in a_lower)
        features["hedge_ratio"] = features["hedge_count"] / max(len(a_doc), 1)

        features["deflection_count"] = sum(1 for d in self.deflection_phrases if d in a_lower)
        features["has_deflection"] = int(features["deflection_count"] > 0)
        features["starts_with_evasion"] = int(any(a_lower.strip().startswith(e) for e in self.evasion_starters))

        first_words = a_lower.split()[:3] if a_lower.split() else []
        features["has_direct_start"] = int(any(w in self.direct_indicators for w in first_words))
        features["direct_indicator_count"] = sum(1 for d in self.direct_indicators if d in a_lower)
        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["q_entity_count"] = len(q_entities)
        features["a_entity_count"] = len(a_entities)
        features["entity_overlap"] = len(q_entities & a_entities) / max(len(q_entities), 1) if q_entities else 0
        pronouns = {"i", "we", "you", "they", "it", "he", "she", "them", "us"}
        a_pronouns = sum(1 for token in a_doc if token.text in pronouns)
        features["pronoun_ratio"] = a_pronouns / max(len(a_doc), 1)
        adj_adv_count = sum(1 for token in a_doc if token.pos_ in {"ADJ", "ADV"})
        features["adj_adv_ratio"] = adj_adv_count / max(len(a_doc), 1)
        features["avg_sentence_length"] = len(a_doc) / max(features["a_num_sentences"], 1)
        first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
        features["first_person_ratio"] = sum(1 for t in a_doc if t.text in first_person) / max(len(a_doc), 1
        features["answer_has_question"] = int("?" in answer)
        negations = {"not", "no", "never", "nothing", "nobody", "none", "neither", "n't", "dont", "don't"}
        features["negation_count"] = sum(1 for t in a_doc if t.text in negations or t.dep_ == "neg")
        return features

    def _is_yes_no_question(self, question: str) -> int:
        """Detect if question expects yes/no answer."""
        q_lower = question.lower().strip()
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ",
                          "will ", "would ", "could ", "should ", "can ", "has ", "have ", "had "]
        return int(any(q_lower.startswith(s) for s in yes_no_starters))


def main():
    arrow_path = os.path.join("dataset", "train", "train", "data-00000-of-00001.arrow")
    ds = Dataset.from_file(arrow_path)
    df = ds.to_pandas()

    print(f"Dataset size: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    print("Extracting features (this may take a few minutes)...")
    extractor = ClarityFeatureExtractor()

    feature_list = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        features = extractor.extract_features(row["question"], row["interview_answer"])
        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)
    print(f"\nExtracted {len(feature_df.columns)} features:")
    print(feature_df.columns.tolist())
    X = feature_df.values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    }

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    best_model = None
    best_f1 = 0

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
        print(f"CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Test Macro F1: {test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = (name, clf)



    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model[0]} (Macro F1: {best_f1:.4f})")
    print("=" * 60)

    y_pred = best_model[1].predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    if hasattr(best_model[1], "feature_importances_"):
        print("\nTop 15 Most Important Features:")
        importances = best_model[1].feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_df.columns[idx]}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()

