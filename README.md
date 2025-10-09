# Fine-Tuning a Small LLM with LoRA for Automated Paper Review Insights

This repository contains the implementation and experiments for the project "Fine-tuning a Small LLM with LoRA for Automated Paper Review Insights", which explores whether a lightweight fine-tuned language model can assist in the peer review process by generating structured evaluations of scientific papers.

The work focuses on automated generation of paper strengths and weaknesses, along with numeric rating and confidence predictions, using a LoRA fine-tuned LLaMA 3.2B model.

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
```
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
```
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
The dataset_stats.ipynb notebook contains a basic dataset exploration and main tokens statistics.

The main.ipynb notebook contains the entire development of the project:

### Dataset preparation
This cells:

- Load the tp_2017conference.xlsx dataset

- Clean text fields (title, abstract, review)

- Extract numeric scores (rating_num, confidence_num)

- Split into train.csv, val.csv, and test.csv

### Zero-Shot structured review generation
The base model (LLaMA 3.2B-Instruct) generates structured reviews following a fixed prompt template.
Outputs are saved as train_structured.csv and val_structured.csv.

### LoRA Fine-Tuning
Continue in the same notebook (Fine-Tuning section), the cells:

- Load LLaMA 3.2B in 4-bit quantization via bitsandbytes

- Prepare the model using PEFT (prepare_model_for_kbit_training)

- Fine-tunes via LoRA adapters

- Saves weights under notebooks/finetuned-llama3-lora/

### Generate predictions and apply to test set
After fine-tuning, run the model to generate predictions on the test set using both zero-shot and fine-tuned models.
These predictions will be saved in zero_shot_predictions.csv and finetuned_predictions.csv.

### Evaluation
The final cells evaluate:

- Regression metrics: MAE, RMSE, R², Pearson, Spearman
- Semantic alignment: BERTScore (F1)

## Discussion
The fine-tuned model successfully learns the review structure and improves textual alignment with human-written reviews.
However, confidence prediction remains inconsistent, suggesting that reviewer uncertainty cannot be reliably inferred from text alone.


#### Project developed by Camilla Bonomo

