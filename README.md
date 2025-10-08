# Fine-Tuning a Small LLM with LoRA for Automated Paper Review Insights

This repository contains the implementation and experiments for the project "Fine-tuning a Small LLM with LoRA for Automated Paper Review Insights", which explores whether a lightweight fine-tuned language model can assist in the peer review process by generating structured evaluations of scientific papers.

The work focuses on automated generation of paper strengths and weaknesses, along with numeric rating and confidence predictions, using a LoRA fine-tuned LLaMA 3.2B model.

---

## Table of Contents


---

## Project Overview

Peer reviewing is a cornerstone of scientific communication but remains time-consuming and inconsistent.  
This project investigates how parameter-efficient fine-tuning (LoRA) can be used to adapt a small LLM to generate structured, interpretable peer reviews, even with a limited dataset.  

Specifically, it compares:
- **Zero-shot generation** (baseline performance of the pretrained model)
- **Fine-tuned generation** (LoRA-adapted model on human-written reviews)

Evaluation is conducted on both textual and numerical levels, using regression and semantic metrics.

---

## Repository Structure

├── data/
│ ├── tp_2017conference.xlsx # Original OpenReview dataset (2017)
│ ├── train.csv # Cleaned training split
│ ├── val.csv # Validation split
│ └── test.csv # Test split
│
├── notebooks/
│ ├── main.ipynb # Main training + evaluation pipeline
│ ├── dataset_stats.ipynb # Dataset exploration and token analysis
│ ├── finetuned-llama3/ # Model outputs and checkpoints
│ └── finetuned-llama3-lora/ # Final fine-tuned LoRA weights
│
├── train_structured.csv # Zero-shot distilled structured reviews (train set)
├── val_structured.csv # Zero-shot distilled structured reviews (validation set)
├── zero_shot_predictions.csv # Model predictions before fine-tuning
├── finetuned_predictions.csv # Model predictions after fine-tuning
└── README.md

---
## Setup Instructions

### 1. Connect to your VM
Access your instance via the Azure portal at [labs.azure.com/virtualmachines](https://labs.azure.com/virtualmachines).  
You can open **VS Code → Remote Explorer → Connect to Host**.

Ensure your VM includes:
- Ubuntu 20.04+ (or similar Linux)
- CUDA-enabled GPU
- Python 3.10 – 3.12 environment

### 2. Clone the repository
```bash
git clone https://github.com/camillabonomo02/ML_project.git
cd ML_project
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Authenticate with Hugging Face
In the section of the notebook dedicated to the Hugging Face log in insert your own or use the given one if not expired. This grants access to download meta-llama/Llama-3.2-3B-Instruct.
To generate your how token access at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and generate a fine-grained token for meta-llama\Llama 3.2-3B-Instruct and selct at least "Read acess to contents on selcted repos".
Then log in:

```python
from huggingface_hub import login
login("your_huggingface_token")
```
---

## Run the notebooks



