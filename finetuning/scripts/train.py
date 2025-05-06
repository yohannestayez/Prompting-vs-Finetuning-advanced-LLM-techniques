import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import numpy as np
from finetuning.scripts.log import get_file_logger

# Initialize the logger
logger = get_file_logger()

def evaluate(model, val_loader, device):
    """Evaluate the model on validation data and compute loss/accuracy"""
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0  # Count of correct predictions
    total = 0    # Total number of samples
    criterion = torch.nn.CrossEntropyLoss()  # Loss function

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in val_loader:
            # Move batch data to the specified device (GPU/CPU)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)  # Compute loss

            # Accumulate metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)  # Get predicted classes
            correct += (preds == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Increment total sample count

    # Calculate average validation loss and accuracy
    val_loss = total_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc

def train(model, train_dataset, val_dataset, device, epochs=10, batch_size=16, lr=5e-5, patience=3):
    """Train the model with early stopping based on validation loss"""
    model.to(device)  # Move model to specified device (GPU/CPU)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Variables for early stopping
    best_val_loss = float("inf")  # Initialize with very high value
    patience_counter = 0  # Counts epochs without improvement

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        correct = 0
        total = 0

        # Process batches with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Training steps
            optimizer.zero_grad()  # Clear existing gradients
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Accumulate training metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Calculate training metrics for the epoch
        train_acc = correct / total
        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Log training progress
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best validation loss
            patience_counter = 0     # Reset patience counter
        else:
            patience_counter += 1    # Increment patience counter
            logger.info(f"⏳ No improvement. Early stop patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("🛑 Early stopping triggered.")
                break  # Stop training if patience is exceeded