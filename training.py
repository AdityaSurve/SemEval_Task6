import torch
import gc
import sys
import importlib
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from transformers import get_linear_schedule_with_warmup
import numpy as np

# Force reload modules
for module in ['config', 'data_loader', 'model']:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

from config import Config
from model import ClarityModel
from data_loader import get_dataloaders

# Cleaning memory
gc.collect()
torch.cuda.empty_cache()

# Loading the data

train_loader, val_loader, test_loader, le_clarity, class_weights = get_dataloaders()

num_labels = len(le_clarity.classes_)
print(f"\n** Training for {num_labels} classes: {le_clarity.classes_} **")

# Setup model
device = Config.DEVICE
print(f"\n ** Training on: {device} **")

# Moving class weights to device
class_weights = class_weights.to(device)

model = ClarityModel(num_labels=num_labels, class_weights=class_weights).to(device)

# Counting the parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"** Total params: {total_params:,} **")
print(f"** Trainable params: {trainable_params:,} **")

# Optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY
)

# Learning rate scheduler with warmup
num_training_steps = len(train_loader) * Config.EPOCHS
num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

print(f"\n** Training for {Config.EPOCHS} epochs **")
print(f"** Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps} **")

# Training functions
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    loop = tqdm(loader, desc="Training")
    
    for i, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels_clarity'].to(device)
        
        # Forward pass
        loss, logits = model(input_ids, attention_mask, labels)
        
        # Gradient accumulation
        loss = loss / Config.ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % Config.ACCUMULATION_STEPS == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * Config.ACCUMULATION_STEPS
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        
        del input_ids, attention_mask, labels, loss, logits
        
        
        if i % 50 == 0:
            torch.cuda.empty_cache()
        
        # Update progress bar
        loop.set_postfix(
            loss=total_loss / (i + 1),
            lr=scheduler.get_last_lr()[0]
        )
    
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, macro_f1

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels_clarity'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            
            del input_ids, attention_mask, labels, loss, logits
    
    
    torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, macro_f1, all_preds, all_labels

# Training loop with early stopping

print("** TRAINING START **")


best_val_f1 = 0
patience_counter = 0

for epoch in range(Config.EPOCHS):
   
    print(f"** Epoch {epoch + 1}/{Config.EPOCHS} **")
    
    
    # Train
    train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device)
    
    # Validate
    val_loss, val_f1, val_preds, val_labels = evaluate(model, val_loader, device)
    
    print(f"\n** Results: **")
    print(f"  Train Loss: {train_loss:.4f} | Train Macro-F1: {train_f1:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Macro-F1:   {val_f1:.4f}")
    
    # Saving the best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f" New best model saved! (F1: {best_val_f1:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{Config.PATIENCE})")
    
    # Early stopping
    if patience_counter >= Config.PATIENCE:
        print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
        break

# Final evaluation on test set

print("** TEST EVALUATION **")


# Load best model
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

test_loss, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)

print(f"\n ** Test Macro-F1: {test_f1:.4f} **")
print(f"** Test Loss: {test_loss:.4f} **")

# Detailed classification report
print("\n" + "=" * 50)
print("📋 DETAILED CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(
    test_labels, 
    test_preds, 
    target_names=le_clarity.classes_,
    digits=4
))

print("\n** Training complete!")
print(f"** Best model saved to: {Config.MODEL_SAVE_PATH}")