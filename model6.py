import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_recall_fscore_support
)
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from datasets import Dataset

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed. Install with: pip install lightgbm")

class Config:
    DATA_PATH = "dataset/train/train/data-00000-of-00001.arrow"
    FEATURES_PATH = "dataset/train_features.parquet"
    MODEL_DIR = "models"
    LEXICON_DIR = "external_datasets"
    
    LABELS = ['Dodging', 'Explicit', 'General', 'Partial/half-answer']
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1


def load_lexicons():
    lexicons = {}
    hedge_path = f"{Config.LEXICON_DIR}/hedges.txt"
    if os.path.exists(hedge_path):
        with open(hedge_path, 'r', encoding='utf-8') as f:
            lexicons['hedges'] = {line.strip().lower() for line in f if line.strip()}
    else:
        lexicons['hedges'] = set()

    lexicons['modals'] = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'ought'}
    lexicons['negations'] = {'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
                             "n't", 'cannot', "can't", "won't", "don't", "doesn't", "didn't"}
    lexicons['fillers'] = {'uh', 'um', 'er', 'ah', 'like', 'well', 'so', 'basically', 'actually'}
    lexicons['filler_phrases'] = ['you know', 'i mean', 'kind of', 'sort of']
    lexicons['vague'] = {'some', 'many', 'few', 'several', 'various', 'certain', 'thing', 'things',
                         'stuff', 'something', 'anything', 'people', 'someone', 'often', 'sometimes'}
    lexicons['vague_phrases'] = ['a lot', 'a bit', 'a little']
    lexicons['pivots'] = ['what i want to say', 'let me be clear', 'the fact is', 'the truth is',
                          'the real question', 'what matters is', 'the important thing',
                          'the bottom line', 'at the end of the day']
    lexicons['thanks'] = ['thank you', 'thanks for', 'great question', 'good question', 'i appreciate']
    
    print(f"Loaded lexicons: {len(lexicons['hedges'])} hedges, {len(lexicons['modals'])} modals")
    return lexicons


class FeatureExtractor:
    def __init__(self, lexicons):
        self.lexicons = lexicons
        print("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_sm")
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.sia = SentimentIntensityAnalyzer()
        try:
            from afinn import Afinn
            self.afinn = Afinn()
        except:
            self.afinn = None
        
        print("NLP models loaded!")
    
    def extract_semantic_features(self, questions, answers):
        print("Computing semantic embeddings...")
        q_emb = self.sbert.encode(questions, show_progress_bar=True)
        a_emb = self.sbert.encode(answers, show_progress_bar=True)
        qa_sim = np.array([
            np.dot(q, a) / (np.linalg.norm(q) * np.linalg.norm(a) + 1e-8)
            for q, a in zip(q_emb, a_emb)
        ])
        print("Computing first sentence similarity...")
        first_sents = [sent_tokenize(a)[0] if sent_tokenize(a) else "" for a in answers]
        fs_emb = self.sbert.encode(first_sents, show_progress_bar=True)
        fs_sim = np.array([
            np.dot(q, f) / (np.linalg.norm(q) * np.linalg.norm(f) + 1e-8)
            for q, f in zip(q_emb, fs_emb)
        ])
        
        return {
            'qa_similarity': qa_sim,
            'topic_shift_score': 1 - qa_sim,
            'first_sentence_similarity': fs_sim
        }
    
    def extract_row_features(self, question, answer):
        features = {}
        answer_lower = answer.lower()
        tokens = word_tokenize(answer_lower)
        alpha_tokens = [t for t in tokens if t.isalpha()]
        
        q_doc = self.nlp(question)
        a_doc = self.nlp(answer)

        features['answer_length_tokens'] = len([t for t in a_doc if not t.is_space])
        features['answer_length_chars'] = len(answer)
        features['answer_to_question_len_ratio'] = len(answer) / max(len(question), 1)
        features['num_sentences'] = len(list(a_doc.sents))

        features['hedge_score'] = sum(1 for t in tokens if t in self.lexicons['hedges'])
        
        filler_count = sum(1 for t in tokens if t in self.lexicons['fillers'])
        for phrase in self.lexicons['filler_phrases']:
            filler_count += answer_lower.count(phrase)
        features['filler_score'] = filler_count
        
        vague_count = sum(1 for t in tokens if t in self.lexicons['vague'])
        for phrase in self.lexicons['vague_phrases']:
            vague_count += answer_lower.count(phrase)
        features['vague_word_count'] = vague_count
        
        features['modal_verb_count'] = sum(1 for t in tokens if t in self.lexicons['modals'])

        features['num_numbers'] = sum(1 for t in a_doc if t.like_num or t.pos_ == 'NUM')
        features['num_named_entities'] = len(a_doc.ents)
        
        total_alpha = len([t for t in a_doc if t.is_alpha])
        content_tokens = len([t for t in a_doc if t.is_alpha and not t.is_stop])
        features['specificity_score'] = content_tokens / max(total_alpha, 1)

        vader = self.sia.polarity_scores(answer)
        features['sentiment_compound'] = vader['compound']
        features['sentiment_positive'] = vader['pos']
        features['sentiment_negative'] = vader['neg']
        features['sentiment_neutral'] = vader['neu']
        
        if self.afinn:
            features['afinn_score'] = max(-1, min(1, self.afinn.score(answer) / 10))
        else:
            features['afinn_score'] = 0
        
        features['emotion_confidence'] = abs(vader['compound'])

        if total_alpha > 0:
            features['pos_ratio_verbs'] = len([t for t in a_doc if t.pos_ == 'VERB']) / total_alpha
            features['pos_ratio_nouns'] = len([t for t in a_doc if t.pos_ == 'NOUN']) / total_alpha
            features['pos_ratio_pronouns'] = len([t for t in a_doc if t.pos_ == 'PRON']) / total_alpha
        else:
            features['pos_ratio_verbs'] = 0
            features['pos_ratio_nouns'] = 0
            features['pos_ratio_pronouns'] = 0
        
        num_clauses = 1
        for token in a_doc:
            if token.dep_ in ('ccomp', 'advcl', 'relcl', 'acl'):
                num_clauses += 1
        features['num_clauses'] = num_clauses

        features['starts_with_thanks'] = int(any(answer_lower.strip().startswith(p) for p in self.lexicons['thanks']))
        features['pivot_score'] = sum(1 for p in self.lexicons['pivots'] if p in answer_lower)
        features['negation_count'] = sum(1 for t in tokens if t in self.lexicons['negations'] or "n't" in t)

        q_ents = {e.text.lower() for e in q_doc.ents}
        a_ents = {e.text.lower() for e in a_doc.ents}
        features['deflection_score'] = len(a_ents - q_ents) / max(len(a_ents), 1)

        if len(alpha_tokens) > 0:
            features['ttr'] = len(set(alpha_tokens)) / len(alpha_tokens)
            word_counts = Counter(alpha_tokens)
            probs = [c / len(alpha_tokens) for c in word_counts.values()]
            features['entropy_score'] = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            features['ttr'] = 0
            features['entropy_score'] = 0

        q_words = {t.lemma_ for t in q_doc if not t.is_stop and t.is_alpha}
        a_words = {t.lemma_ for t in a_doc if not t.is_stop and t.is_alpha}
        features['keyword_overlap'] = len(q_words & a_words) / max(len(q_words), 1)
        features['entity_overlap'] = len(q_ents & a_ents)
        
        return features
    
    def extract_all_features(self, df):
        questions = df['question'].tolist()
        answers = df['interview_answer'].tolist()

        semantic = self.extract_semantic_features(questions, answers)

        print("Extracting row-wise features...")
        row_features = []
        for q, a in tqdm(zip(questions, answers), total=len(questions)):
            row_features.append(self.extract_row_features(q, a))

        feature_df = pd.DataFrame(row_features)
        for col, values in semantic.items():
            feature_df[col] = values
        
        return feature_df


def get_model(model_type, **kwargs):
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 10)
    
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=Config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == 'xgboost' and HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=Config.RANDOM_STATE,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    elif model_type == 'lightgbm' and HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=Config.RANDOM_STATE,
            class_weight='balanced',
            verbose=-1
        )
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=Config.RANDOM_STATE
        )
    elif model_type == 'logistic':
        return LogisticRegression(
            max_iter=1000,
            random_state=Config.RANDOM_STATE,
            class_weight='balanced'
        )
    elif model_type == 'svm':
        return SVC(
            kernel='rbf',
            random_state=Config.RANDOM_STATE,
            class_weight='balanced',
            probability=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, 
                       model_type, label_encoder, **kwargs):
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*50}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = get_model(model_type, **kwargs)
    print(f"Training on {len(X_train)} samples...")
    model.fit(X_train_scaled, y_train)

    val_pred = model.predict(X_val_scaled)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    print(f"Validation Macro F1: {val_f1:.4f}")

    test_pred = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n{'='*50}")
    print("TEST RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")

    target_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=target_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    if hasattr(model, 'feature_importances_'):
        print("\nTop 15 Feature Importances:")
        feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
        print(feat_imp.sort_values(ascending=False).head(15))
    
    return model, scaler, {
        'val_f1': val_f1,
        'test_f1': test_f1,
        'test_acc': test_acc
    }


def main():
    parser = argparse.ArgumentParser(description='Train feature-based evasion detection model')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting', 'logistic', 'svm'],
                        help='Model type to train')
    parser.add_argument('--n_estimators', type=int, default=200, help='Number of estimators')
    parser.add_argument('--max_depth', type=int, default=10, help='Max tree depth')
    parser.add_argument('--use_saved_features', action='store_true', 
                        help='Use saved features instead of re-extracting')
    parser.add_argument('--save_features', action='store_true',
                        help='Save extracted features to parquet')
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL 6: Feature-based Evasion Detection")
    print("="*60)

    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print("\n[1/4] Loading data...")
    if args.use_saved_features and os.path.exists(Config.FEATURES_PATH):
        print(f"Loading saved features from {Config.FEATURES_PATH}")
        df = pd.read_parquet(Config.FEATURES_PATH)
        feature_cols = [c for c in df.columns if c not in ['question', 'interview_question', 
                                                            'interview_answer', 'label', 'url',
                                                            'inaudible', 'multiple_questions', 
                                                            'affirmative_questions']]
    else:
        ds = Dataset.from_file(Config.DATA_PATH)
        df = ds.to_pandas()
        print(f"Loaded {len(df)} samples")

        print("\n[2/4] Extracting features...")
        lexicons = load_lexicons()
        extractor = FeatureExtractor(lexicons)
        feature_df = extractor.extract_all_features(df)

        df = pd.concat([df, feature_df], axis=1)
        feature_cols = feature_df.columns.tolist()

        if args.save_features:
            df.to_parquet(Config.FEATURES_PATH, index=False)
            print(f"Saved features to {Config.FEATURES_PATH}")
    
    print(f"Features: {len(feature_cols)}")

    print("\n[3/4] Preparing train/val/test splits...")
    X = df[feature_cols]

    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    print(f"Classes: {le.classes_}")
    print(f"Distribution: {Counter(y)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=Config.TEST_SIZE + Config.VAL_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n[4/4] Training model...")
    model, scaler, metrics = train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test,
        args.model, le,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{Config.MODEL_DIR}/model6_{args.model}_{timestamp}.joblib"
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'metrics': metrics
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    best_path = f"{Config.MODEL_DIR}/model6_best.joblib"
    if not os.path.exists(best_path):
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'label_encoder': le,
            'feature_cols': feature_cols,
            'metrics': metrics
        }, best_path)
        print(f"Saved as best model: {best_path}")


if __name__ == "__main__":
    main()
