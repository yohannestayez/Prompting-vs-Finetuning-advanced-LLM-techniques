# Spam Classification: Fine-tuning vs. Prompt Engineering

This project compares two natural language processing techniques for spam email classification:

1.  **Fine-tuning**: A DistilBERT model is fine-tuned on a custom dataset of spam and non-spam emails.
2.  **Prompt Engineering**: Google's Gemini model is used with a few-shot prompting technique to classify emails.

The primary goal is to evaluate and contrast the performance, setup, and potential use cases of these two distinct approaches for the given task.

## Project Structure

```
.
├── finetuning/
│   ├── models/                     # Saved fine-tuned models and tokenizers
│   │   └── distilbert_finetuned.pt # Example fine-tuned DistilBERT model
│   ├── notebooks/                  # Jupyter notebooks for experimentation (if any)
│   ├── scripts/
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   ├── evaluate.py             # Evaluation script for fine-tuned models
│   │   ├── log.py                  # Logging utility
│   │   └── train.py                # Training script for fine-tuning
│   ├── compare.log                 # Log file for the comparison script
│   ├── compare.py                  # Script to run and compare both methods
│   ├── run.py                      # Script to run the fine-tuning process
│   └── training.log                # Log file for the fine-tuning process
├── results.md                      # Detailed results and analysis of the comparison
└── README.md                       # This file
```

## Core Comparison: Fine-tuning vs. Prompt Engineering

The project focuses on highlighting the differences between:

*   **Fine-tuning (DistilBERT):**
    *   Involves taking a pre-trained model (DistilBERT) and further training it on a specific dataset (`spam_cleaned.csv`).
    *   The `finetuning/run.py` script manages this process, including data loading, model training, and saving the fine-tuned model.
    *   This approach typically requires a labeled dataset and computational resources for training.

*   **Prompt Engineering (Gemini):**
    *   Utilizes a large language model (Gemini) by crafting a specific input prompt.
    *   This project employs a **few-shot prompting** technique: the prompt includes a small number of examples (email text and its corresponding label: "spam" or "not spam") to guide the model.
    *   The `finetuning/compare.py` script implements this by:
        1.  Selecting a few examples from the dataset.
        2.  Constructing a prompt that presents these examples to Gemini.
        3.  Requesting classification for new, unseen emails based on these examples.
    *   This method can be effective with very little task-specific data but relies heavily on the quality of the prompt and the capabilities of the base LLM.

The `finetuning/compare.py` script is central to this comparison. It loads the fine-tuned DistilBERT model and uses the Gemini API to get predictions for a common test set, allowing for a direct comparison of their performance metrics.

## Results

The detailed quantitative and qualitative results of the comparison, including performance metrics (Accuracy, Precision, Recall, F1-score) for both methods and an analysis of their behavior, are documented in `results.md`.
