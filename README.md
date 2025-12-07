# Fine-Tuning a Small LLM with LoRA for Automated Paper Review Insights

This repository provides an end-to-end, scriptable workflow for transforming raw OpenReview-style conference data into structured reviewer insights.  
The pipeline extracts cleaned examples, distills strengths/weaknesses with a zero-shot teacher, fine-tunes a lightweight LLaMA 3.2B model via LoRA, and evaluates both textual and numeric review quality.

---

## Project Overview

Peer review remains essential yet inconsistent. Here we explore whether a compact LoRA-adapted LLM can:

- Summarize papers into concise **strengths** and **weaknesses** bullet lists.
- Produce an approximate **overall rating** on the 1–10 OpenReview scale.
- Improve upon zero-shot behavior while staying cheap to train (4-bit quantization, LoRA adapters only).

The main script `run_experiment.py` orchestrates five sequential steps:
1. Prepare cleaned train/validation splits.
2. Distill structured S/W targets with a zero-shot “teacher”.
3. Fine-tune the student model on the teacher JSON outputs.
4. Run inference (structured JSON + rating logits) and rate-only calibration passes.
5. Evaluate numeric metrics plus S/W quality diagnostics.

---

## Repository Structure

```
├── data/                # Raw spreadsheets + derived CSV splits + teacher targets
├── model/               # Saved LoRA adapters (final_adapter_sw) and HF checkpoints
├── results/             # Inference outputs, calibration runs, evaluation metrics
├── requirements.txt     # Runtime dependencies for the pipeline
└── run_experiment.py    # Single entry point for every stage
```

`data/` should contain your original spreadsheet (e.g., `tp_2020conference.xlsx`).  
Every other artifact (`*_clean.csv`, `*_sw_targets.csv`, results CSVs, and metrics) is auto-generated inside these folders.

---

## Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/camillabonomo02/Automated_Paper_Review.git
   cd Automated_Paper_Review
   ```
2. **Create a Python environment (recommended 3.10+) and install deps**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Log in to Hugging Face**
- CLI alternative:
  ```bash
  huggingface-cli login
  ```
  or via Python:
  ```python
  from huggingface_hub import login
  login("hf_your-token")
  ```
   - The token must have read access to `meta-llama/Llama-3.2-3B-Instruct`.
   - Configure your environment for GPU + CUDA (4-bit inference/training relies on `bitsandbytes`).

---

## Data Requirements

`run_experiment.py` expects a CSV or Excel file containing at least the following columns (case-insensitive):

- `title`
- `abstract`
- `review` (free-form reviewer text)
- One column containing a numeric or textual `rate`.

Missing or malformed rows are discarded during cleaning. Ratings are normalized to the 1–10 OpenReview scale.

---

## Running the Pipeline

The entire workflow is driven by the `--step` argument. You can run each stage separately or execute everything with `--step all`.

```bash
python run_experiment.py --step <prepare|teacher|train|infer|eval|all> \
    --file data/tp_2020conference.xlsx \
    [--skip_full_infer] [--skip_rate_only]
```

### 1. Prepare (`--step prepare`)

- Reads the raw spreadsheet/CSV.
- Cleans text fields, parses ratings, and splits into 80/20 train/validation (`data/train_clean.csv`, `data/val_clean.csv`).

### 2. Teacher Distillation (`--step teacher`)

- Uses `meta-llama/Llama-3.2-3B-Instruct` in 4-bit mode to generate **strict JSON** strength/weakness targets.
- Produces `data/train_sw_targets.csv` and `data/val_sw_targets.csv`.
- Supports resume logic and configurable subsampling (see constants in `Config`).

### 3. LoRA Training (`--step train`)

- Fine-tunes the base model on the teacher JSON structures using PEFT LoRA adapters.
- Stores outputs under `model/final_adapter_sw/` (adapter weights + tokenizer).
- Training parameters (LR, batch size, grad acc steps, epochs) are defined in `Config`.

### 4. Inference (`--step infer`)

- **Full inference** on validation:
  - Runs both zero-shot (`mode=zeroshot`) and adapter (`mode=ft`) models to produce JSON S/W predictions plus rating logits (`results/val_<mode>_results.csv`).
- **Rate-only inference** on train (for calibration) unless `--skip_rate_only`:
  - Produces `results/train_<mode>_rateonly_results.csv`.
- Use `--skip_full_infer` to omit the S/W generation pass if you only need rate calibration.

### 5. Evaluation (`--step eval`)

- Fits a robust regression calibration (HuberRegressor) on rate-only predictions.
- Computes MAE, RMSE, and Pearson correlation for raw vs calibrated predictions (`results/final_metrics.json`).
- Runs the S/W quality diagnostics:
  - Fallback percentage (default text usage).
  - Lexical diversity of bullets.
  - Optional cosine similarity vs teacher outputs (if sentence-transformer embeddings are available).
  - Metrics saved to `results/sw_quality_metrics.json`.

### 6. All-In-One (`--step all`)

Runs every stage in sequence with default options. Ideal for fully reproducing the pipeline once requirements and data are prepared.

---

## Key Configuration Options

All knobs live inside the `Config` class in `run_experiment.py`. Highlights:

- `MODEL_ID`: change to another chat model if needed.
- `TEACHER_*`: token budget, batch size, retry counts, subsampling limits for distillation.
- `MIX_USE_REVIEW_PROB`: probability of feeding the human review during student training (default = always use it).
- `MAX_SEQ_LEN`: max tokens for HF datasets.
- `INFER_*`: generation parameters for evaluation runs.
- `CALIB_TRAIN_MAX_EXAMPLES`: cap on train samples used for calibration inference.

Modify these constants before running the corresponding steps.

---

## Outputs & Artifacts

| Stage      | Artifact(s)                                                       |
|------------|------------------------------------------------------------------|
| Prepare    | `data/train_clean.csv`, `data/val_clean.csv`                      |
| Teacher    | `data/train_sw_targets.csv`, `data/val_sw_targets.csv`            |
| Train      | `model/final_adapter_sw/` (LoRA adapter + tokenizer)              |
| Inference  | `results/val_zeroshot_results.csv`, `results/val_ft_results.csv`, `results/train_<mode>_rateonly_results.csv` |
| Evaluation | `results/final_metrics.json`, `results/sw_quality_metrics.json`   |

Each inference CSV contains raw model outputs, parsed JSON, and numeric ratings for downstream analysis.

---

## Tips & Troubleshooting

- **Accelerate + bitsandbytes**: ensure CUDA drivers and GPU memory are sufficient; 4-bit quantization keeps VRAM usage manageable (~8–10 GB).
- **Teacher resume**: delete `data/train_sw_targets.csv` or `data/val_sw_targets.csv` if you want to regenerate from scratch.
- **JSON parsing failures**: the teacher stage automatically retries with a stricter prompt and higher token budget; worst case it falls back to a deterministic S/W template so downstream steps never crash.
- **SentenceTransformer optional**: if you cannot download embeddings (no GPU or HF access), S/W similarity metrics gracefully skip the embedding step.

---

## License & Attribution

This work was developed by **Camilla Bonomo** as part of the “Fine-Tuning a Small LLM with LoRA for Automated Paper Review Insights” project.  
Please credit the project if you build upon this codebase or release derived adapters/datasets.

--- 

Happy experimenting! Feel free to adapt the prompts, training recipes, or evaluation routines to match your conference or domain. 
