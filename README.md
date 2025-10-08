# 🧠 Automated Peer Review Generation using LoRA Fine-Tuning of LLaMA 3.2B

This project explores the use of **Large Language Models (LLMs)** for **automating peer review generation**.  
Using a small, domain-specific dataset of research papers and reviews (from the [OpenReview dataset](https://github.com/Seafoodair/Openreview)), the goal is to train a lightweight, interpretable model capable of producing structured reviews that summarize the **strengths**, **weaknesses**, and **numeric evaluations** of scientific papers.

The approach combines **prompt-based generation**, **Low-Rank Adaptation (LoRA)** fine-tuning, and **quantized model loading** to achieve high efficiency on limited resources.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Key Features](#key-features)  
4. [Setup Instructions](#setup-instructions)  
5. [Data Preparation](#data-preparation)  
6. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)  
7. [Evaluation](#evaluation)  
8. [Results Summary](#results-summary)  
9. [Limitations and Future Work](#limitations-and-future-work)  
10. [Citation](#citation)  
11. [License](#license)

---

## Project Overview

Traditional peer review is essential to academic publishing but suffers from **delays, inconsistency, and reviewer fatigue**.  
Recent advances in **LLMs** make it possible to support or partially automate this process.

This project fine-tunes a **quantized LLaMA 3.2B-Instruct** model using **LoRA (Low-Rank Adaptation)** to:
- Generate structured reviews (Strengths/Weaknesses).
- Predict numeric **rating (1–10)** and **confidence (1–5)** scores.
- Evaluate the generated text against human-written reviews.

The pipeline is designed to be **reproducible, resource-efficient, and interpretable**, suitable for educational or research purposes.

---

## 📂 Repository Structure

