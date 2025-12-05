import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import Config
from model import ClarityModel

model = ClarityModel(num_labels=len(le_clarity.classes_))

model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH), strict=False)

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
    ).to(Config.DEVICE)
    
    # Predict
    with torch.no_grad():
        _, logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = F.softmax(logits, dim=1)
    
    # Get results
    pred_idx = torch.argmax(probs).item()
    pred_label = le_clarity.inverse_transform([pred_idx])[0]
    confidence = probs[0][pred_idx].item()
    
    # Show all probabilities
    print("\n" + "="*60)
    print(" CLARITY ANALYSIS")
    print("="*60)
    print(f"\n❓ Question: {question}")
    print(f"\n Answer: {answer}")
    print("\n" + "-"*60)
    print(" Prediction Probabilities:")
    print("-"*60)
    
    for i, label in enumerate(le_clarity.classes_):
        prob = probs[0][i].item()
        bar = "||" * int(prob * 40)
        marker = "-->" if i == pred_idx else "  "
        print(f"{marker} {label:20s}: {prob:6.2%} {bar}")
    
    print("\n" + "="*60)
    print(f" FINAL PREDICTION: {pred_label} ({confidence:.1%} confidence)")
    print("="*60 + "\n")
    
    return pred_label, confidence

print("TESTING WITH EXAMPLE QUESTIONS\n")

# Example 1: Clear dodge
q1 = "Will you raise taxes on the middle class?"
a1 = "I grew up in a middle class home, and I know how hard it is."

predict_clarity(q1, a1)

# Example 2: Clear answer
q2 = "Do you support universal healthcare?"
a2 = "Yes, I support universal healthcare. Every American deserves access to quality medical care regardless of their income."

predict_clarity(q2, a2)

# Example 3: Ambiguous
q3 = "What is your position on gun control?"
a3 = "I believe we need to find a balanced approach that respects both Second Amendment rights and public safety concerns."

predict_clarity(q3, a3)


print("\n" + "="*60)
print(" INTERACTIVE MODE - Try your own examples!")
print("="*60)
print("(Type 'quit' to exit)\n")

while True:
    question = input(" Enter question: ")
    if question.lower() == 'quit':
        break
    
    answer = input(" Enter answer: ")
    if answer.lower() == 'quit':
        break
    
    predict_clarity(question, answer)
