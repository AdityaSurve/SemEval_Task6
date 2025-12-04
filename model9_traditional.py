import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import spacy
import textstat
import joblib
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from lexicon_loader import LexiconLoader

nlp = spacy.load("en_core_web_sm")
FOUR_CLASSES = ["Dodging", "Explicit", "General", "Partial/half-answer"]


class AdvancedFeatureExtractor:
    def __init__(self):
        self.lexicons = LexiconLoader()
        self.lexicons.print_status()
        
        self.hedge_words = self.lexicons.hedge_words if self.lexicons.hedge_words else self._default_hedges()
        self.filler_words = self.lexicons.filler_words
        self.filler_phrases = self.lexicons.filler_phrases
        self.negations = self.lexicons.negation_words
        self.modal_verbs = self.lexicons.modal_verbs
        self.vague_words = self.lexicons.vague_words
        self.vague_phrases = self.lexicons.vague_phrases
        self.pivot_phrases = self.lexicons.pivot_phrases
        self.thanks_starters = self.lexicons.thanks_starters
        
        self.nrc_emotions = self.lexicons.nrc_emotions
        self.nrc_positive = self.lexicons.nrc_sentiments.get('positive', set())
        self.nrc_negative = self.lexicons.nrc_sentiments.get('negative', set())
        self.mpqa_strong = self.lexicons.mpqa_strong_subjective
        self.mpqa_weak = self.lexicons.mpqa_weak_subjective
        
        self.direct_yes_no = ["yes", "no", "yeah", "nope", "yep", "nah", "yea", "naw"]
        self.strong_affirmatives = ["absolutely", "definitely", "certainly", "of course", "exactly", "correct", "right", "sure", "indeed", "precisely", "undoubtedly"]
        self.strong_negatives = ["never", "absolutely not", "no way", "not at all", "definitely not", "certainly not"]
        self.wh_words = {"what", "who", "where", "when", "why", "how", "which", "whom", "whose"}
        self.evasion_starters = ["well,", "so,", "you know,", "i mean,", "actually,", "to be honest", "frankly", "honestly", "look,", "see,", "okay so"]
    
    def _default_hedges(self):
        return {"maybe", "perhaps", "possibly", "probably", "might", "could", "would", "should", "somewhat", "relatively", "fairly", "rather", "kind of", "sort of", "in a way", "to some extent", "arguably", "i think", "i believe", "i guess", "i suppose", "it seems", "it appears", "seemingly", "apparently"}

    def extract_features(self, question: str, answer: str) -> dict:
        question = str(question) if question else ""
        answer = str(answer) if answer else ""
        q_lower = question.lower().strip()
        a_lower = answer.lower().strip()
        q_doc = nlp(q_lower[:5000])
        a_doc = nlp(a_lower[:5000])
        q_words = q_lower.split()
        a_words = a_lower.split()
        features = {}
        features["q_len_words"] = len(q_words)
        features["a_len_words"] = len(a_words)
        features["q_len_chars"] = len(question)
        features["a_len_chars"] = len(answer)
        features["a_q_word_ratio"] = len(a_words) / max(len(q_words), 1)
        features["a_q_char_ratio"] = len(answer) / max(len(question), 1)
        features["log_a_len"] = np.log1p(len(a_words))
        q_sents = list(q_doc.sents)
        a_sents = list(a_doc.sents)
        features["q_num_sentences"] = len(q_sents)
        features["a_num_sentences"] = len(a_sents)
        features["avg_sentence_len"] = len(a_words) / max(len(a_sents), 1)
        features["sentence_len_std"] = np.std([len(list(s)) for s in a_sents]) if len(a_sents) > 1 else 0
        features["is_yes_no_q"] = int(self._is_yes_no_question(q_lower))
        features["is_wh_q"] = int(any(w in q_words[:5] for w in self.wh_words))
        features["q_has_or"] = int(" or " in q_lower)
        features["is_how_much_many"] = int("how much" in q_lower or "how many" in q_lower)
        features["is_why_q"] = int(any(q_lower.startswith(s) for s in ["why ", "why?"]))
        features["is_what_q"] = int(any(q_lower.startswith(s) for s in ["what ", "what?"]))
        features["is_how_q"] = int(q_lower.startswith("how "))
        features["is_when_q"] = int(q_lower.startswith("when "))
        features["is_where_q"] = int(q_lower.startswith("where "))
        features["is_who_q"] = int(q_lower.startswith("who "))
        features["q_ends_with_question"] = int(question.strip().endswith("?"))
        features["q_is_complex"] = int(len(q_words) > 15 or " and " in q_lower)
        first_a_words = a_words[:5] if len(a_words) >= 5 else a_words
        first_a_text = " ".join(first_a_words)
        features["starts_yes"] = int(first_a_words[0] in ["yes", "yeah", "yep", "yea"] if first_a_words else 0)
        features["starts_no"] = int(first_a_words[0] in ["no", "nope", "nah", "naw"] if first_a_words else 0)
        features["starts_yes_no"] = int(features["starts_yes"] or features["starts_no"])
        features["starts_strong_affirm"] = int(any(first_a_text.startswith(s) for s in self.strong_affirmatives))
        features["starts_strong_neg"] = int(any(first_a_text.startswith(s) for s in self.strong_negatives))
        features["starts_evasion"] = int(any(a_lower.startswith(e) for e in self.evasion_starters))
        features["has_direct_answer"] = int(features["starts_yes_no"] or features["starts_strong_affirm"] or features["starts_strong_neg"])
        features["direct_in_first_10_words"] = int(any(w in self.direct_yes_no for w in a_words[:10]))
        features["hedge_count"] = sum(1 for h in self.hedge_words if h in a_lower)
        features["has_hedges"] = int(features["hedge_count"] > 0)
        features["hedge_ratio"] = features["hedge_count"] / max(len(a_words), 1)
        features["hedge_in_first_quarter"] = sum(1 for h in self.hedge_words if h in " ".join(a_words[:len(a_words)//4+1]))
        features["negation_count"] = sum(1 for n in self.negations if n in a_lower.split())
        features["has_negation"] = int(features["negation_count"] > 0)
        features["negation_ratio"] = features["negation_count"] / max(len(a_words), 1)
        features["answer_has_question"] = int("?" in answer)
        features["question_count_in_answer"] = answer.count("?")
        features["ends_with_question"] = int(answer.strip().endswith("?"))
        features["filler_count"] = sum(1 for f in self.filler_words if f in a_lower)
        features["filler_phrase_count"] = sum(1 for fp in self.filler_phrases if fp in a_lower)
        features["has_fillers"] = int(features["filler_count"] > 0 or features["filler_phrase_count"] > 0)
        features["filler_ratio"] = (features["filler_count"] + features["filler_phrase_count"]) / max(len(a_words), 1)
        features["pivot_phrase_count"] = sum(1 for p in self.pivot_phrases if p in a_lower)
        features["has_pivot_phrases"] = int(features["pivot_phrase_count"] > 0)
        features["thanks_starter"] = int(any(a_lower.startswith(t) for t in self.thanks_starters))
        features["deflection_count"] = features["pivot_phrase_count"] + features["thanks_starter"]
        features["has_deflection"] = int(features["deflection_count"] > 0)
        features["vague_word_count"] = sum(1 for v in self.vague_words if v in a_words)
        features["vague_phrase_count"] = sum(1 for vp in self.vague_phrases if vp in a_lower)
        features["vagueness_ratio"] = (features["vague_word_count"] + features["vague_phrase_count"]) / max(len(a_words), 1)
        features["modal_verb_count"] = sum(1 for m in self.modal_verbs if m in a_words)
        features["lexicon_modal_ratio"] = features["modal_verb_count"] / max(len(a_words), 1)
        q_content = {t.lemma_ for t in q_doc if not t.is_stop and not t.is_punct and t.is_alpha and len(t.text) > 2}
        a_content = {t.lemma_ for t in a_doc if not t.is_stop and not t.is_punct and t.is_alpha and len(t.text) > 2}
        overlap = q_content & a_content
        features["content_overlap_count"] = len(overlap)
        features["content_overlap_ratio"] = len(overlap) / max(len(q_content), 1)
        features["a_coverage_of_q"] = len(overlap) / max(len(q_content), 1)
        features["q_coverage_of_a"] = len(overlap) / max(len(a_content), 1)
        q_nouns = {t.lemma_ for t in q_doc if t.pos_ == "NOUN"}
        a_nouns = {t.lemma_ for t in a_doc if t.pos_ == "NOUN"}
        features["noun_overlap"] = len(q_nouns & a_nouns) / max(len(q_nouns), 1) if q_nouns else 0
        q_verbs = {t.lemma_ for t in q_doc if t.pos_ == "VERB"}
        a_verbs = {t.lemma_ for t in a_doc if t.pos_ == "VERB"}
        features["verb_overlap"] = len(q_verbs & a_verbs) / max(len(q_verbs), 1) if q_verbs else 0
        q_entities = {ent.text.lower() for ent in q_doc.ents}
        a_entities = {ent.text.lower() for ent in a_doc.ents}
        features["q_entity_count"] = len(q_entities)
        features["a_entity_count"] = len(a_entities)
        entity_overlap = q_entities & a_entities
        features["entity_overlap"] = len(entity_overlap) / max(len(q_entities), 1) if q_entities else 0
        features["entity_overlap_count"] = len(entity_overlap)
        features["all_q_entities_in_a"] = int(q_entities.issubset(a_entities)) if q_entities else 0
        features["new_entities_in_a"] = len(a_entities - q_entities)
        first_person = {"i", "me", "my", "mine", "myself"}
        third_person = {"he", "she", "they", "them", "his", "her", "their", "it"}
        features["first_person_count"] = sum(1 for w in a_words if w in first_person)
        features["third_person_count"] = sum(1 for w in a_words if w in third_person)
        features["first_person_ratio"] = features["first_person_count"] / max(len(a_words), 1)
        features["uses_we"] = int("we" in a_words or "our" in a_words)
        features["uses_you"] = int("you" in a_words or "your" in a_words)
        features["person_shift"] = int(features["uses_you"] and not features["first_person_count"])
        features["num_commas"] = answer.count(",")
        features["num_periods"] = answer.count(".")
        features["has_ellipsis"] = int("..." in answer)
        features["has_dash"] = int(" - " in answer or "—" in answer)
        features["has_parentheses"] = int("(" in answer and ")" in answer)
        features["exclamation_count"] = answer.count("!")
        features["verb_count"] = sum(1 for t in a_doc if t.pos_ == "VERB")
        features["modal_count"] = sum(1 for t in a_doc if t.tag_ == "MD")
        features["has_modal"] = int(features["modal_count"] > 0)
        features["modal_ratio"] = features["modal_count"] / max(len(a_words), 1)
        features["adj_count"] = sum(1 for t in a_doc if t.pos_ == "ADJ")
        features["adv_count"] = sum(1 for t in a_doc if t.pos_ == "ADV")
        features["modifier_ratio"] = (features["adj_count"] + features["adv_count"]) / max(len(a_words), 1)
        features["noun_count"] = sum(1 for t in a_doc if t.pos_ == "NOUN")
        features["propn_count"] = sum(1 for t in a_doc if t.pos_ == "PROPN")
        features["noun_verb_ratio"] = features["noun_count"] / max(features["verb_count"], 1)
        features["a_is_very_short"] = int(len(a_words) < 10)
        features["a_is_short"] = int(len(a_words) < 30)
        features["a_is_medium"] = int(30 <= len(a_words) <= 100)
        features["a_is_long"] = int(len(a_words) > 100)
        features["a_is_very_long"] = int(len(a_words) > 200)
        if len(answer) > 10:
            try:
                features["flesch_reading_ease"] = textstat.flesch_reading_ease(answer)
                features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(answer)
                features["gunning_fog"] = textstat.gunning_fog(answer)
                features["avg_syllables_per_word"] = textstat.avg_syllables_per_word(answer)
            except:
                features["flesch_reading_ease"] = 50
                features["flesch_kincaid_grade"] = 8
                features["gunning_fog"] = 10
                features["avg_syllables_per_word"] = 1.5
        else:
            features["flesch_reading_ease"] = 50
            features["flesch_kincaid_grade"] = 8
            features["gunning_fog"] = 10
            features["avg_syllables_per_word"] = 1.5
        features["nrc_positive_count"] = sum(1 for w in a_words if w in self.nrc_positive)
        features["nrc_negative_count"] = sum(1 for w in a_words if w in self.nrc_negative)
        features["nrc_sentiment_diff"] = features["nrc_positive_count"] - features["nrc_negative_count"]
        features["nrc_sentiment_ratio"] = features["nrc_positive_count"] / max(features["nrc_negative_count"], 1)
        
        emotion_counts = self.lexicons.get_emotion_counts(a_words)
        for emotion, count in emotion_counts.items():
            features[f"nrc_{emotion}"] = count
        features["nrc_emotion_total"] = sum(emotion_counts.values())
        features["nrc_anger_fear"] = emotion_counts.get("anger", 0) + emotion_counts.get("fear", 0)
        features["nrc_joy_trust"] = emotion_counts.get("joy", 0) + emotion_counts.get("trust", 0)
        
        features["afinn_score"] = self.lexicons.get_afinn_score(answer)
        features["afinn_normalized"] = features["afinn_score"] / max(len(a_words), 1)
        
        features["mpqa_strong_count"] = sum(1 for w in a_words if w in self.mpqa_strong)
        features["mpqa_weak_count"] = sum(1 for w in a_words if w in self.mpqa_weak)
        features["mpqa_subjectivity"] = (features["mpqa_strong_count"] + features["mpqa_weak_count"]) / max(len(a_words), 1)
        
        features["avg_concreteness"] = self.lexicons.get_avg_concreteness(a_words)
        features["unique_word_ratio"] = len(set(a_words)) / max(len(a_words), 1)
        features["lexical_diversity"] = len(set(a_words)) / max(len(a_words), 1)
        features["specific_answer_yes_no"] = int(features["is_yes_no_q"] and features["starts_yes_no"])
        features["evades_yes_no"] = int(features["is_yes_no_q"] and not features["starts_yes_no"] and features["a_len_words"] > 20)
        features["short_yes_no_response"] = int(features["is_yes_no_q"] and features["starts_yes_no"] and features["a_len_words"] < 20)
        features["answers_wh_directly"] = int(features["is_wh_q"] and features["content_overlap_ratio"] > 0.3)
        features["deflects_wh"] = int(features["is_wh_q"] and features["has_deflection"])
        features["long_winded_for_simple_q"] = int(features["is_yes_no_q"] and features["a_len_words"] > 50 and not features["starts_yes_no"])
        features["explicit_score"] = (features["has_direct_answer"] * 3 + features["content_overlap_ratio"] * 2 + features["entity_overlap"] * 2 + features["noun_overlap"] * 2 + features["specific_answer_yes_no"] * 2 + features["avg_concreteness"] - features["hedge_count"] * 0.5)
        features["dodging_score"] = (features["has_pivot_phrases"] * 3 + features["thanks_starter"] * 2 + features["answer_has_question"] * 2 + features["starts_evasion"] * 2 + features["evades_yes_no"] * 2 + features["person_shift"] * 2 + (1 - features["content_overlap_ratio"]) * 2 + features["long_winded_for_simple_q"])
        features["partial_score"] = (features["hedge_count"] * 2 + features["vagueness_ratio"] * 5 + features["modal_verb_count"] + int(0.1 < features["content_overlap_ratio"] < 0.4) * 2 + features["has_hedges"] - features["has_direct_answer"])
        features["general_score"] = ((1 - features["entity_overlap"]) * 2 + features["new_entities_in_a"] * 0.5 + int(features["content_overlap_ratio"] < 0.2) * 2 + features["a_is_long"] + features["uses_we"])
        features["evasion_score"] = (features["pivot_phrase_count"] * 2 + features["thanks_starter"] + features["vagueness_ratio"] * 3 + features["mpqa_subjectivity"] * 2)
        return features

    def _is_yes_no_question(self, q_lower: str) -> bool:
        yes_no_starters = ["do ", "does ", "did ", "is ", "are ", "was ", "were ", "will ", "would ", "could ", "should ", "can ", "has ", "have ", "had ", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "can't", "hasn't", "haven't", "hadn't"]
        return any(q_lower.startswith(s) for s in yes_no_starters)


def load_data():
    df = pd.read_parquet("dataset/train_with_features.parquet")
    df = df[df["label"].isin(FOUR_CLASSES)].reset_index(drop=True)
    return df


def extract_all_features(df):
    extractor = AdvancedFeatureExtractor()
    features_list = []
    print("Extracting features...")
    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"  {idx}/{len(df)}")
        features = extractor.extract_features(row["question"], row["interview_answer"])
        features_list.append(features)
    feature_df = pd.DataFrame(features_list)
    print(f"Extracted {len(feature_df.columns)} features")
    return feature_df


def add_tfidf_features(df, feature_df, n_components=30):
    print("Adding TF-IDF features...")
    texts = (df["question"].fillna("") + " " + df["interview_answer"].fillna("")).tolist()
    tfidf = TfidfVectorizer(max_features=n_components, ngram_range=(1, 2), min_df=5, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(texts).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    combined_df = pd.concat([feature_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return combined_df, tfidf


def get_feature_importance(model, model_name, X, y, feature_names):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        return np.abs(model.coef_).mean(axis=0)
    else:
        print(f"  Computing permutation importance for {model_name}...")
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        return result.importances_mean


def select_top_features(importances, feature_names, k):
    indices = np.argsort(importances)[::-1][:k]
    return indices, [feature_names[i] for i in indices]


def objective_with_feature_selection(trial, model_type, X_train, y_train, X_val, y_val, feature_names):
    n_features = trial.suggest_int("n_features", 15, min(80, len(feature_names)))
    
    if model_type == "lgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1
        }
        base_model = lgb.LGBMClassifier(**params)
    elif model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "eval_metric": "mlogloss"
        }
        base_model = xgb.XGBClassifier(**params)
    elif model_type == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1
        }
        base_model = RandomForestClassifier(**params)
    elif model_type == "et":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1
        }
        base_model = ExtraTreesClassifier(**params)
    elif model_type == "cat":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 400),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "random_state": 42, "verbose": 0
        }
        base_model = CatBoostClassifier(**params)
    
    temp_model = base_model.__class__(**{k: v for k, v in params.items() if k != "n_estimators" and k != "iterations"})
    if model_type == "cat":
        temp_model.set_params(iterations=50)
    else:
        temp_model.set_params(n_estimators=50)
    temp_model.fit(X_train, y_train)
    
    importances = get_feature_importance(temp_model, model_type, X_train, y_train, feature_names)
    selected_indices, _ = select_top_features(importances, feature_names, n_features)
    
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    
    base_model.fit(X_train_selected, y_train)
    y_pred = base_model.predict(X_val_selected)
    
    trial.set_user_attr("selected_features", selected_indices.tolist())
    trial.set_user_attr("params", params)
    
    return f1_score(y_val, y_pred, average="macro")


def train_with_per_model_feature_selection():
    df = load_data()
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
    feature_df = extract_all_features(df)
    feature_df, tfidf = add_tfidf_features(df, feature_df, n_components=30)
    feature_names = feature_df.columns.tolist()
    
    X = feature_df.values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    
    print(f"\nTotal features: {X.shape[1]}")
    print(f"Classes: {le.classes_}")
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    X_train_full_balanced, y_train_full_balanced = smote.fit_resample(X_train_full_scaled, y_train_full)
    print(f"After SMOTE: {np.bincount(y_train_balanced)}")
    
    model_configs = {
        "LightGBM": "lgb",
        "XGBoost": "xgb",
        "Random Forest": "rf",
        "Extra Trees": "et",
        "CatBoost": "cat"
    }
    
    print("\n" + "="*70)
    print("PER-MODEL FEATURE SELECTION + HYPERPARAMETER TUNING")
    print("="*70)
    
    best_configs = {}
    
    for model_name, model_type in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Tuning {model_name} (features + hyperparameters)...")
        print(f"{'='*50}")
        
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(
            lambda trial: objective_with_feature_selection(
                trial, model_type, X_train_balanced, y_train_balanced, X_val_scaled, y_val, feature_names
            ),
            n_trials=40,
            show_progress_bar=True
        )
        
        best_trial = study.best_trial
        best_features = best_trial.user_attrs["selected_features"]
        best_params = best_trial.user_attrs["params"]
        
        print(f"\nBest {model_name}:")
        print(f"  Val F1: {best_trial.value:.4f}")
        print(f"  N features: {len(best_features)}")
        print(f"  Top features: {[feature_names[i] for i in best_features[:10]]}")
        
        best_configs[model_name] = {
            "model_type": model_type,
            "params": best_params,
            "feature_indices": best_features,
            "feature_names": [feature_names[i] for i in best_features],
            "val_f1": best_trial.value
        }
    
    print("\n" + "="*70)
    print("TRAINING FINAL MODELS WITH SELECTED FEATURES")
    print("="*70)
    
    results = {}
    trained_models = {}
    
    for model_name, config in best_configs.items():
        print(f"\nTraining {model_name} with {len(config['feature_indices'])} features...")
        
        X_train_selected = X_train_full_balanced[:, config["feature_indices"]]
        X_test_selected = X_test_scaled[:, config["feature_indices"]]
        
        if config["model_type"] == "lgb":
            model = lgb.LGBMClassifier(**config["params"])
        elif config["model_type"] == "xgb":
            model = xgb.XGBClassifier(**config["params"])
        elif config["model_type"] == "rf":
            model = RandomForestClassifier(**config["params"])
        elif config["model_type"] == "et":
            model = ExtraTreesClassifier(**config["params"])
        elif config["model_type"] == "cat":
            model = CatBoostClassifier(**config["params"])
        
        model.fit(X_train_selected, y_train_full_balanced)
        y_pred = model.predict(X_test_selected)
        
        test_f1 = f1_score(y_test, y_pred, average="macro")
        test_acc = accuracy_score(y_test, y_pred)
        
        print(f"  Test F1: {test_f1:.4f} | Acc: {test_acc:.4f}")
        
        results[model_name] = {"f1": test_f1, "acc": test_acc, "n_features": len(config["feature_indices"])}
        trained_models[model_name] = {"model": model, "feature_indices": config["feature_indices"], "config": config}
    
    print("\n" + "="*70)
    print("ENSEMBLE WITH PER-MODEL FEATURES")
    print("="*70)
    
    top3 = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)[:3]
    print(f"\nTop 3 models for ensemble: {[m[0] for m in top3]}")
    
    class FeatureSelectingClassifier:
        def __init__(self, model, feature_indices):
            self.model = model
            self.feature_indices = feature_indices
        
        def fit(self, X, y):
            X_selected = X[:, self.feature_indices]
            self.model.fit(X_selected, y)
            return self
        
        def predict(self, X):
            X_selected = X[:, self.feature_indices]
            return self.model.predict(X_selected)
        
        def predict_proba(self, X):
            X_selected = X[:, self.feature_indices]
            return self.model.predict_proba(X_selected)
    
    ensemble_preds_proba = []
    for model_name, _ in top3:
        model_info = trained_models[model_name]
        X_test_selected = X_test_scaled[:, model_info["feature_indices"]]
        proba = model_info["model"].predict_proba(X_test_selected)
        ensemble_preds_proba.append(proba)
    
    avg_proba = np.mean(ensemble_preds_proba, axis=0)
    ensemble_pred = np.argmax(avg_proba, axis=1)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average="macro")
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"\nSoft Voting Ensemble (top 3): F1={ensemble_f1:.4f}, Acc={ensemble_acc:.4f}")
    results["Ensemble (Top 3)"] = {"f1": ensemble_f1, "acc": ensemble_acc, "n_features": "mixed"}
    
    weights = [results[name]["f1"] for name, _ in top3]
    weights = np.array(weights) / sum(weights)
    weighted_proba = sum(w * p for w, p in zip(weights, ensemble_preds_proba))
    weighted_pred = np.argmax(weighted_proba, axis=1)
    weighted_f1 = f1_score(y_test, weighted_pred, average="macro")
    weighted_acc = accuracy_score(y_test, weighted_pred)
    
    print(f"Weighted Voting Ensemble: F1={weighted_f1:.4f}, Acc={weighted_acc:.4f}")
    results["Weighted Ensemble"] = {"f1": weighted_f1, "acc": weighted_acc, "n_features": "mixed"}
    
    print("\n" + "="*70)
    print("FINAL RANKING")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, res) in enumerate(sorted_results):
        n_feat = res.get("n_features", "N/A")
        print(f"{i+1}. {name}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}, Features={n_feat}")
    
    best_name = sorted_results[0][0]
    
    print("\n" + "="*70)
    print(f"BEST: {best_name}")
    print("="*70)
    
    if "Ensemble" in best_name:
        if "Weighted" in best_name:
            y_pred_best = weighted_pred
        else:
            y_pred_best = ensemble_pred
    else:
        model_info = trained_models[best_name]
        X_test_selected = X_test_scaled[:, model_info["feature_indices"]]
        y_pred_best = model_info["model"].predict(X_test_selected)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE PER MODEL")
    print("="*70)
    
    for model_name, config in best_configs.items():
        print(f"\n{model_name} - Top 15 features (out of {len(config['feature_names'])}):")
        for i, feat in enumerate(config["feature_names"][:15]):
            print(f"  {i+1}. {feat}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "trained_models": trained_models,
        "best_configs": best_configs,
        "scaler": scaler,
        "tfidf": tfidf,
        "label_encoder": le,
        "feature_names": feature_names,
        "results": results
    }, "models/model9_per_model_features.joblib")
    print(f"\nModel saved to models/model9_per_model_features.joblib")
    
    return results, trained_models, best_configs


if __name__ == "__main__":
    results, trained_models, best_configs = train_with_per_model_feature_selection()
