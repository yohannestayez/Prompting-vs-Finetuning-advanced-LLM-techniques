import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from finetuning.scripts.log import get_file_logger

def evaluate(model, test_dataset, device, batch_size=32):
    """Evaluate model performance on test dataset and log classification metrics"""
    
    # Get the logger instance for recording evaluation results
    logger = get_file_logger()
    
    # Move model to the specified device (GPU/CPU)
    model.to(device)
    # Set model to evaluation mode (disables dropout/batch norm)
    model.eval()

    # Create DataLoader for batching test samples
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Lists to accumulate all predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Process each batch in the test dataset
        for batch in test_loader:
            # Move batch tensors to the specified device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Get predicted class indices (argmax of logits)
            preds = torch.argmax(outputs.logits, dim=1)

            # Store predictions and labels (moving back to CPU if needed)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Generate comprehensive classification report from sklearn
    # digits=4 controls decimal places in the output
    report = classification_report(all_labels, all_preds, digits=4)
    
    # Log the complete evaluation report using the logger
    # Using %s formatter for proper string handling
    logger.info("Evaluation Results:\n%s", report)
    
    # Return the report in case calling code needs to process it
    return report