import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import Config
from model import ClarityModel
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Create label encoder with the classes from training
le_clarity = LabelEncoder()
le_clarity.classes_ = np.array(['Ambivalent', 'Clear Non-Reply', 'Clear Reply'])

# Load model
model = ClarityModel(num_labels=len(le_clarity.classes_))
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE), strict=False)
model = model.to(Config.DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def predict_clarity(question, answer, summary=""):
    """
    Predict whether a politician's answer is clear, ambiguous, or evasive
    
    Args:
        question: The interviewer's question
        answer: The politician's response
        summary: Optional GPT summary of context
    
    Returns:
        dict: prediction results with label and probabilities
    """
    sep = tokenizer.sep_token
    
    # Combine inputs same way as training
    if summary and len(summary) > 10:
        text = f"{question} {sep} {summary} {sep} {answer}"
    else:
        text = f"{question} {sep} {answer}"
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=Config.MAX_LEN,
        padding='max_length'
    )
    
    # Move to device
    inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Model returns (loss, logits) - we want the second element
        if isinstance(outputs, tuple):
            loss, logits = outputs  # loss will be None, logits is what we need
        elif isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
            
        probs = F.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = le_clarity.inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx].item()
    
    # Get all probabilities
    all_probs = {
        label: probs[0][i].item() 
        for i, label in enumerate(le_clarity.classes_)
    }
    
    return {
        'predicted_label': pred_label,
        'confidence': confidence,
        'probabilities': all_probs
    }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("POLITICIAN ANSWER CLARITY PREDICTOR")
    print("=" * 60)
    
    # Test example 1
    question1 = "Will you raise taxes if elected?"
    answer1 = "We need to look at all options to ensure fiscal responsibility while maintaining essential services."
    
    print("\n📝 Example 1:")
    print(f"Question: {question1}")
    print(f"Answer: {answer1}")
    
    result1 = predict_clarity(question1, answer1)
    print(f"\n🎯 Prediction: {result1['predicted_label']}")
    print(f"💯 Confidence: {result1['confidence']:.2%}")
    print("📊 All probabilities:")
    for label, prob in result1['probabilities'].items():
        print(f"   {label}: {prob:.2%}")
    
    # Test example 2
    question2 = "Do you support the proposed healthcare bill?"
    answer2 = "Yes, I fully support this bill because it will provide affordable healthcare to millions of Americans."
    
    print("\n" + "=" * 60)
    print("\n📝 Example 2:")
    print(f"Question: {question2}")
    print(f"Answer: {answer2}")
    
    result2 = predict_clarity(question2, answer2)
    print(f"\n🎯 Prediction: {result2['predicted_label']}")
    print(f"💯 Confidence: {result2['confidence']:.2%}")
    print("📊 All probabilities:")
    for label, prob in result2['probabilities'].items():
        print(f"   {label}: {prob:.2%}")
    
    # Test example 3 - ambiguous answer
    question3 = "Will you cut funding to education?"
    answer3 = "Education is very important to me and I've always valued it throughout my career."
    
    print("\n" + "=" * 60)
    print("\n📝 Example 3:")
    print(f"Question: {question3}")
    print(f"Answer: {answer3}")
    
    result3 = predict_clarity(question3, answer3)
    print(f"\n🎯 Prediction: {result3['predicted_label']}")
    print(f"💯 Confidence: {result3['confidence']:.2%}")
    print("📊 All probabilities:")
    for label, prob in result3['probabilities'].items():
        print(f"   {label}: {prob:.2%}")
    
    print("\n" + "=" * 60)
    print("\n✅ Testing complete! You can now use predict_clarity() with your own examples.")
    print("\nUsage:")
    print("  result = predict_clarity('your question', 'politician answer')")
    print("=" * 60)