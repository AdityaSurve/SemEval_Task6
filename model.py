%%writefile model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config

class ClarityModel(nn.Module):
    """
    Single-task model focused on clarity classification
    Simpler architecture = better performance for single task
    """
    def __init__(self, num_labels, class_weights=None):
        super(ClarityModel, self).__init__()
        
        # Loading pre-trained transformer
        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size
        
        # using intermediate layer which is a simple and effective way
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),  
            nn.Dropout(Config.DROPOUT),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Weighted loss for class imbalance
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] - optional, for training
            
        Returns:
            loss (if labels provided), logits [batch_size, num_labels]
        """
        # Getting contextualized embeddings from the transformer
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Using [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classifier
        logits = self.classifier(pooled_output)
        
        # Calculating loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return loss, logits
    
    def predict(self, input_ids, attention_mask):
        """
        Get predictions without computing loss
        Used during inference
        """
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
        return preds
