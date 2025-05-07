**Report: Comparative Assessment of Fine-Tuned BERT and Prompt-Engineered Gemini in Spam Classification**

### 1. Objective

Evaluate and compare the classification accuracy of

* **DistilBERT**, which is pre-trained and fine-tuned using a spam email dataset.
* **Gemini**, employing a few-shot prompt-engineered methodology with five examples.

All the assessments were performed on a held-out test set of 20 messages derived from the same cleaned dataset.

### 2. Experimental Setup

* **Dataset**

* Source: spam_cleaned

* Test set: 20 messages chosen at random (random state = 42)

* Few-shot samples: 5 random messages (state = 1)

**Models & Configuration**

1. **Fine-Tuned DistilBERT**

* Base: distilbert
* Classification head fine-tuned using full training split

2. **Gemini Prompt**


**Labels**

* Positive class ("spam") = 1

* Negative class ("not spam") = 0

### 3. Results

| Model                        | Accuracy | Precision | Recall | F1 Score |
| ---------------------------- | -------- | --------- | ------ | -------- |
| **Fine-Tuned DistilBERT**    | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| **Prompt-Engineered Gemini** | 0.7000   | 0.3333    | 1.0000 | 0.5000   |


### 4. Analysis

* **Fine-Tuned DistilBERT**, as its name implies, performed perfect classification on the 20-sample test set, proving it can identify spam from non-spam under existing circumstances.

* **Prompt-Engineered Gemini** exhibited:

- **Recall = 1.0000**: all spam messages were correctly identified.

- **Precision = 0.3333:** a very high ratio of false positives ("spam" to non-spam messages).

- **F1 Score = 0.5000**, indicating balance of perfect recall and moderate precision.

### 5. Recommendations

1. **Broader Assessment**

* Expand the test set size to at least 100 messages to better support model performance and evaluation metric stability.

2. **Prompt Refinement**

* Augment sparse example sets to represent richer spam patterns and edge conditions.

* Try using structured templates or chain-of-thought formats to direct the Gemini model.

3. **Hybrid Deployment Strategy**

* As main classifier, employ DistilBERT; selectively utilize Gemini prompts for uncertain or new email content.

4. **Error Analysis**

* Review misclassified non-spam messages to find common flaws and modify prompt wording or post-processing rules accordingly.
  Meaning

### 6. Conclusion

On the test set specified, the fine-tuned DistilBERT model surpasses the prompt-engineered Gemini method in overall accuracy, F1 score, and precision. The Gemini method, although attaining perfect recall, needs to have its false positives minimized through additional prompt tuning. Scaling tests and optimizing prompting techniques will be paramount in fine-tuning both solutions for deployment.
