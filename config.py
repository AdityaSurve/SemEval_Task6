import torch

class Config:
    
    MODEL_NAME = "distilroberta-base"  
    
    # Tokenization
    MAX_LEN = 256  
    
    # Training 
    BATCH_SIZE = 4  
    ACCUMULATION_STEPS = 8  
    EPOCHS = 4  
    LEARNING_RATE = 2e-5  
    WARMUP_RATIO = 0.1  
    WEIGHT_DECAY = 0.01
    
    # Regularization
    DROPOUT = 0.1  
    MAX_GRAD_NORM = 1.0  
    
    # Task-specific
    NUM_LABELS_CLARITY = 3
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    MODEL_SAVE_PATH = "best_clarity_model.pth"
    
    # Early stopping
    PATIENCE = 2  

