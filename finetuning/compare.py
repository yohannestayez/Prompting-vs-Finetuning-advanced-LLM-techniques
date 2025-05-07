# spam_comparison.py

import torch
import json
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv
load_dotenv()

# setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename='./finetuning/compare.log',
    filemode='w'
)

FINETUNED_MODEL_DIR = "./finetuning/models"
DATASET_CSV         = "./finetuning/data/spam_cleaned.csv"
GEMINI_MODEL_NAME   = os.getenv("GEMINI_MODEL") 
NUM_FEWSHOT         = 5
TEST_SAMPLES        = 20 
API_KEY             = os.getenv("GEMINI_API")

# Label mapping
LABEL_MAP = {0: "not spam", 1: "spam"}

# 1. Load Fine-tuned BERT Model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load(f"{FINETUNED_MODEL_DIR}/distilbert_finetuned.pt"))
model.eval()
logging.info('Model loaded successfully')

# 2. Prediction with Fine-tuned BERT
def predict_finetuned(texts, model, tokenizer, max_length=512, device="cpu"):
    logging.info(f"Predicting {len(texts)} texts with fine-tuned BERT")
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits
        preds   = torch.argmax(logits, dim=-1).cpu().tolist()
    logging.info("BERT prediction completed")
    return preds

# 3. Gemini Configuration
def call_gemini(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Input prompt must be a non-empty string")
    
    try:
        genai.configure(api_key=API_KEY)
        logging.info("Calling Gemini API")
        response = genai.GenerativeModel(GEMINI_MODEL_NAME).generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1024,
                temperature=0.0,
                top_k=40,
                top_p=0.95
            )
        )
        logging.info("Gemini API call successful")
        return response.text
    except Exception as e:
        logging.error(f"Failed to generate text: {str(e)}")
        raise ValueError(f"Failed to generate text: {str(e)}")

# 4. Batch Prediction for Gemini
def parse_gemini_response(response: str) -> list[str]:
    try:
        logging.info("Parsing Gemini response")
        response = response.strip().replace("'", '"')
        if response.startswith("["):
            labels = json.loads(response)
        else:
            labels = re.findall(r'(spam|not spam)', response, flags=re.IGNORECASE)
        parsed_labels = [lbl.lower().strip() for lbl in labels]
        logging.info(f"Parsed {len(parsed_labels)} labels from Gemini response")
        return parsed_labels
    except Exception as e:
        logging.error(f"Failed to parse response: {str(e)}")
        return []

def build_batch_prompt(test_emails, examples):
    logging.info("Building batch prompt for Gemini")
    example_lines = []
    for idx, (email, label) in enumerate(examples, 1):
        example_lines.append(f"Example {idx}:\nEmail: {email}\nLabel: {label}\n")
    
    test_lines = []
    for idx, email in enumerate(test_emails, 1):
        test_lines.append(f"Email {idx}: {email}")
    
    return f"""
    Classify these emails as either 'spam' or 'not spam'. First some examples:
    
    {''.join(example_lines)}
    
    Now classify these {len(test_emails)} emails:
    {''.join(test_lines)}
    
    Return ONLY a Python list of labels using exactly this format:
    ["spam", "not spam", ..., "spam"]
    """

# 5. Evaluation Function
def evaluate_methods(true_labels, bert_preds, gemini_preds):
    gemini_binary = [1 if lbl == "spam" else 0 for lbl in gemini_preds]
    
    logging.info("=== Fine-tuned BERT Metrics ===")
    logging.info(f"Accuracy:  {accuracy_score(true_labels, bert_preds):.4f}")
    logging.info(f"Precision: {precision_score(true_labels, bert_preds):.4f}")
    logging.info(f"Recall:    {recall_score(true_labels, bert_preds):.4f}")
    logging.info(f"F1 Score:  {f1_score(true_labels, bert_preds):.4f}\n")
    
    logging.info("=== Gemini Metrics ===")
    logging.info(f"Accuracy:  {accuracy_score(true_labels, gemini_binary):.4f}")
    logging.info(f"Precision: {precision_score(true_labels, gemini_binary):.4f}")
    logging.info(f"Recall:    {recall_score(true_labels, gemini_binary):.4f}")
    logging.info(f"F1 Score:  {f1_score(true_labels, gemini_binary):.4f}")

# 6. Main Execution
def main():
    logging.info("Loading dataset")
    df = pd.read_csv(DATASET_CSV)
    
    # Split data
    test_df = df.sample(n=TEST_SAMPLES, random_state=42)
    examples_df = df.drop(test_df.index).sample(n=NUM_FEWSHOT, random_state=1)
    
    # Prepare data
    texts = test_df["message"].tolist()
    true_labels = test_df["label"].tolist()
    
    # Convert examples to text labels
    examples = [
        (row["message"], LABEL_MAP[row["label"]]) 
        for _, row in examples_df.iterrows()
    ]
    
    # BERT predictions
    bert_preds = predict_finetuned(texts, model, tokenizer)
    
    # Gemini batch prediction
    batch_prompt = build_batch_prompt(texts, examples)
    gemini_response = call_gemini(batch_prompt)
    gemini_labels = parse_gemini_response(gemini_response)
    
    # Handle potential parsing errors
    if len(gemini_labels) != TEST_SAMPLES:
        logging.warning(f"Warning: Expected {TEST_SAMPLES} labels, got {len(gemini_labels)}")
        gemini_labels = ["not spam"] * TEST_SAMPLES
    
    # Evaluate both methods
    evaluate_methods(true_labels, bert_preds, gemini_labels)
    logging.info("Comparison complete")

if __name__ == "__main__":
    main()
