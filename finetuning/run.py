from sys import path
path.append('.')
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import random_split
from finetuning.scripts.dataset import SpamDataset
from finetuning.scripts.train import train
from finetuning.scripts.evaluate import evaluate

# Load cleaned dataset
df = pd.read_csv("finetuning/data/spam_cleaned.csv")
X = df["cleaned_message"].astype(str).tolist()
y = df["label"].tolist()

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize
encodings = tokenizer(X, truncation=True, padding=True, return_tensors="pt")
labels = torch.tensor(y)

# Create full dataset
full_dataset = SpamDataset(encodings, labels)   

# Train/Val split (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train
train(model, train_dataset, val_dataset, device)

# Save model + tokenizer
torch.save(model.state_dict(), "finetuning/models/distilbert_finetuned.pt")
tokenizer.save_pretrained("finetuning/models/distilbert_finetuned")

# Test set
X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt")
test_dataset = SpamDataset(test_encodings, torch.tensor(y_test))

# Evaluate
evaluate(model, test_dataset, device)
