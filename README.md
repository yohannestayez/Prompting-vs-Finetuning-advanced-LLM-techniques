# Project: Exploring Advanced Prompt Engineering Techniques

This project delves into the practical application and comparative analysis of prompt engineering across different AI tasks. It showcases how carefully crafted prompts can steer Large Language Models (LLMs) like Google's Gemini to perform complex operations and achieve specific goals. The project is organized into two main explorations:

## 1. Multi-Agent like structured Code Optimization (`prompting/`)

This module demonstrates an AI-powered Python code optimization system built using AutoGen. It employs a multi-agent framework where specialized AI agents collaborate to:
- Analyze Python code for inefficiencies.
- Classify types of inefficiencies.
- Devise optimization strategies.
- Implement optimized code.
- Generate insights and refinement suggestions.

A significant focus is placed on **sophisticated prompt engineering techniques** to guide each agent. These include:
- **Role Prompting**: Assigning personas (e.g., "senior performance optimization engineer").
- **Instruction Prompting**: Clear, direct task instructions.
- **Few-Shot Prompting**: Providing examples of desired input/output.
- **Structured Output Prompting**: Ensuring outputs in specific formats (e.g., JSON).
- **Chain-of-Thought (CoT) Prompting**: Guiding the model through reasoning steps.
- **Self-Consistency**: Generating multiple reasoning paths for robust solutions.
- **ReAct (Reason + Act) Prompting**: Iterative reasoning and action cycles.
- **Generated Knowledge Prompting**: Synthesizing new insights from context.

For a detailed breakdown of the agents and specific prompts, please refer to the `prompting/README.md` file.

## 2. Fine-tuning vs. Prompt Engineering for Classification (`finetuning/`)

This module provides a comparative study between traditional model fine-tuning and prompt engineering for a spam email classification task. It contrasts:
- **Fine-tuning**: Further training a pre-trained model (DistilBERT) on a specific dataset.
- **Prompt Engineering**: Using a large language model (Gemini) with **few-shot prompting**, where the model is guided by a few examples of spam/non-spam emails within the prompt itself.

The goal is to evaluate the performance, setup requirements, and practical implications of using prompt engineering as an alternative or complement to fine-tuning for classification tasks.

Detailed methodology and results of this comparison can be found in `finetuning/README.md` and `finetuning/results.md`.

## Overall Emphasis

Across both modules, this project highlights the power and versatility of **prompt engineering**. It explores how different prompting strategies can be used to elicit desired behaviors from LLMs, manage complex workflows, and achieve nuanced outcomes in tasks ranging from code generation and optimization to text classification. 
