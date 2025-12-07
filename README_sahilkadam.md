# SemEval Task 6 - Sahil Kadam's Improvements

## Overview
Improvements to the politician answer clarity classification model, focusing on data handling and testing.

## Key Changes
1. **Fixed Testing Script** - Corrected column name from `answer` to `interview_answer`
   - Accuracy improved from **27.6% → 66.23%**
2. **Data Augmentation Pipeline** - Balance minority classes through oversampling
3. **Comprehensive Testing Tools** - Better model evaluation

## Project Structure
```
.
├── config.py                 # Model configuration
├── model.py                  # ClarityModel architecture
├── data_loader.py            # Data loading (supports augmentation)
├── training.py               # Training script
├── testing.py                # Original testing script
├── testing_fixed.py          # Fixed testing script (use this!)
├── quick_augmentation.py     # Data augmentation tool
└── dataset/                  # Dataset directory
```

## Setup

### Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full list

## Usage

### 1. Basic Training
```bash
python3 training.py
```

### 2. Testing
```bash
# Use the fixed testing script
python3 testing_fixed.py
```

### 3. With Data Augmentation
```bash
# Step 1: Create balanced dataset
python3 quick_augmentation.py

# Step 2: Train (automatically uses augmented data if available)
python3 training.py

# Step 3: Test
python3 testing_fixed.py
```

## Results

### Current Performance
- **Accuracy**: 66.23%
- **Macro F1**: 0.41
- **Model**: distilroberta-base
- **Training**: 4 epochs

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Ambivalent | 0.69 | 0.96 | 0.80 |
| Clear Non-Reply | 0.43 | 0.39 | 0.41 |
| Clear Reply | 1.00 | 0.01 | 0.03 |

### Key Issue Identified
Model heavily biased toward "Ambivalent" class due to class imbalance:
- Ambivalent: 59% of training data
- Clear Reply: 30%
- Clear Non-Reply: 10%

## Critical Bug Fix
**Problem**: Original `testing.py` used wrong column name
```python
answer = example.get('answer', '')  # ❌ Column doesn't exist
```

**Solution**: Use correct column name
```python
answer = example.get('interview_answer', '')  # ✅ Correct
```

This single fix improved accuracy by **139%** (27.6% → 66.23%)

## Data Augmentation
The `quick_augmentation.py` script balances classes by oversampling:
- Original: Ambivalent (2040), Clear Reply (1052), Clear Non-Reply (356)
- Augmented: All classes balanced to ~1800 samples
- Total training samples: 3,448 → 5,640

## Future Improvements
1. Train for more epochs (10-15)
2. Use larger model (roberta-base)
3. Implement advanced augmentation (back-translation)
4. Fine-tune class weights
5. Add focal loss for imbalanced classification

## Model Architecture
- **Base Model**: distilroberta-base (82M parameters)
- **Classification Head**: Linear → GELU → Dropout → Linear
- **Loss**: Weighted CrossEntropyLoss
- **Optimizer**: AdamW with linear warmup
