"""
Test script for Task 2: Evasion Classification (9 classes)
Evaluates the trained model on a test dataset
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger
from models.model2 import EvasionFeatureExtractor

logger = Logger()

NINE_CLASSES = [
    "Explicit", "Dodging", "Implicit", "General", "Deflection",
    "Declining to answer", "Claims ignorance", "Clarification", "Partial/half-answer"
]

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
ARTIFACTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


def test(test_file=None):
    """
    Test the trained Task 2 model on a test dataset.
    
    Args:
        test_file: Path to test parquet file. If None, uses dataset/task2_test.parquet
    """
    logger.log("TASK 2: EVASION CLASSIFICATION - TESTING", "announce")
    
    # Load model artifacts
    model_path = os.path.join(ARTIFACTS_PATH, "task2_model.joblib")
    if not os.path.exists(model_path):
        logger.log(f"Model not found at {model_path}", "error")
        logger.log("Please run train() first", "warning")
        return
    
    artifacts = joblib.load(model_path)
    logger.log("Model loaded successfully", "success")
    
    # Load test data
    if test_file is None:
        test_file = os.path.join(DATASET_PATH, "task2_test.parquet")
    
    if not os.path.exists(test_file):
        logger.log(f"Test file not found at {test_file}", "error")
        logger.log("Please provide a valid test file", "warning")
        return
    
    df = pd.read_parquet(test_file)
    logger.log(f"Loaded {len(df)} test samples", "success")
    
    # Handle different column names
    if "evasion_label" in df.columns:
        df["label"] = df["evasion_label"]
    
    if "question" not in df.columns and "interview_question" in df.columns:
        df["question"] = df["interview_question"]
    
    # Check if we have labels (for evaluation) or just predictions
    has_labels = "label" in df.columns and df["label"].notna().all()
    
    if has_labels:
        df = df[df["label"].isin(NINE_CLASSES)].reset_index(drop=True)
        logger.log("Class distribution:", "plain")
        print(df["label"].value_counts())
    
    # Extract features
    extractor = EvasionFeatureExtractor()
    
    logger.log("Extracting features...", "plain")
    features_list = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  {idx}/{len(df)}")
        feats = extractor.extract(row["question"], row["interview_answer"])
        features_list.append(feats)
    
    feature_df = pd.DataFrame(features_list)
    X = feature_df.values
    X_scaled = artifacts["scaler"].transform(X)
    
    logger.log("RUNNING PREDICTIONS", "announce")
    
    # Get predictions from different models
    results = {}
    
    # Soft voting
    y_pred_soft = artifacts["soft_vote"].predict(X_scaled)
    results["Soft Voting"] = y_pred_soft
    
    # Stacking
    y_pred_stack = artifacts["stack"].predict(X_scaled)
    results["Stacking"] = y_pred_stack
    
    # Individual models
    for name, model in artifacts["trained_models"].items():
        y_pred = model.predict(X_scaled)
        results[name] = y_pred
    
    if has_labels:
        # Evaluate all models
        y_true = artifacts["label_encoder"].transform(df["label"])
        
        logger.log("EVALUATION RESULTS", "announce")
        
        best_f1 = 0
        best_model = None
        
        for name, y_pred in results.items():
            f1 = f1_score(y_true, y_pred, average="macro")
            acc = accuracy_score(y_true, y_pred)
            logger.log(f"{name}: F1={f1:.4f} | Acc={acc:.4f}", "plain")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = name
        
        logger.log(f"BEST: {best_model} (F1={best_f1:.4f})", "success")
        
        # Detailed report for best model
        best_pred = results[best_model]
        print("\nClassification Report:")
        print(classification_report(y_true, best_pred, target_names=artifacts["label_encoder"].classes_))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, best_pred))
        
        # Return predictions with labels
        df["predicted"] = artifacts["label_encoder"].inverse_transform(best_pred)
        
    else:
        # No labels - just return predictions
        logger.log("No labels found - returning predictions only", "warning")
        df["predicted"] = artifacts["label_encoder"].inverse_transform(results["Soft Voting"])
    
    # Save predictions
    output_path = os.path.join(DATASET_PATH, "task2_predictions.csv")
    df.to_csv(output_path, index=False)
    logger.log(f"Predictions saved to {output_path}", "success")
    
    return df


def predict_single(question: str, answer: str) -> dict:
    """
    Predict evasion label for a single Q&A pair.
    Returns dict with label and confidence scores.
    """
    model_path = os.path.join(ARTIFACTS_PATH, "task2_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please run train() first.")
    
    artifacts = joblib.load(model_path)
    extractor = EvasionFeatureExtractor()
    
    features = extractor.extract(question, answer)
    X = np.array([list(features.values())])
    X_scaled = artifacts["scaler"].transform(X)
    
    # Get prediction and probabilities
    y_pred = artifacts["soft_vote"].predict(X_scaled)
    y_proba = artifacts["soft_vote"].predict_proba(X_scaled)[0]
    
    label = artifacts["label_encoder"].inverse_transform(y_pred)[0]
    classes = artifacts["label_encoder"].classes_
    
    return {
        "label": label,
        "confidence": {cls: float(prob) for cls, prob in zip(classes, y_proba)}
    }


if __name__ == "__main__":
    test()


