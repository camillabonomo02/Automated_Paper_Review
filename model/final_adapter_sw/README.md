---
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.2-3B-Instruct
- lora
- transformers
---

# Model Card for the Automated Paper Review LoRA Adapter

This folder hosts a PEFT/LoRA adapter that fine-tunes `meta-llama/Llama-3.2-3B-Instruct` to turn an academic paper's title, abstract, and reviewer comments into a structured JSON object containing strengths, weaknesses, and an approximate 1–10 overall rating. The adapter is the final artifact of the “Automated Paper Review” workflow described in the project root README.

## Model Details

### Model Description

The adapter specializes a 3B-parameter Llama 3.2 chat model for single-turn text generation. At inference time the model receives a templated instruction containing the paper metadata and free-form review and must return JSON with two arrays of concise bullet points plus a numeric score. Training data is obtained by distilling zero-shot outputs from the same base model, ensuring consistent formatting while keeping costs low.

- **Developed by:** Camilla Bonomo (Automated Paper Review project)
- **Funded by [optional]:** Independent student research project
- **Shared by [optional]:** Automated Paper Review repository
- **Model type:** Decoder-only causal LLM with PEFT LoRA adapter (rank 8, alpha 16, dropout 0.05)
- **Language(s) (NLP):** English (academic CS-style prose)
- **License:** Inherits the Meta Llama 3.2 Community License; adapter distribution must comply with the base model’s terms
- **Finetuned from model [optional]:** `meta-llama/Llama-3.2-3B-Instruct`

### Model Sources [optional]

- **Repository:** https://github.com/camillabonomo02/Automated_Paper_Review

## Uses

### Direct Use

- Generate structured reviewer insights when given a paper title, abstract, and full textual review.
- Produce JSON with `strengths`, `weaknesses`, and `overall_rating` fields that downstream analytics scripts can parse.
- Compare zero-shot vs. fine-tuned behavior inside the evaluation pipeline shipped with this repo.

### Out-of-Scope Use

- Automated accept/reject decisions, reviewer replacement, or any process that requires verified factual accuracy.
- Multi-lingual review curation (training data is exclusively English).
- Open-ended chat or general-purpose knowledge tasks unrelated to academic peer review.

## Bias, Risks, and Limitations

- The teacher data is derived from a single OpenReview-style conference spreadsheet focused on computer vision/ML papers. Topics, writing style, and reviewer tone outside that domain may cause brittle outputs.
- Teacher signals originate from the same base model, so any hallucinations or stylistic quirks in the teacher responses are reinforced during fine-tuning.
- Ratings are heuristic (teacher-generated) and exhibit weak correlation with the original reviewer scores even after calibration, so they should never be treated as authoritative quality judgments.
- JSON formatting is generally stable, but long or poorly structured reviews can still trigger truncated generations.

### Recommendations

Inform downstream users that the adapter mirrors both the biases of the OpenReview source material and the Llama 3.2 teacher. Always keep a human reviewer in the loop, and run the provided calibration utilities before comparing numeric ratings to real submission scores.

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "model/final_adapter_sw"

tokenizer = AutoTokenizer.from_pretrained(base_model)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_path)

prompt = """<s>[INST]You are an assistant that creates JSON with strengths, weaknesses,
and an overall rating (1-10). Use the review to populate the fields.
Paper Title: {title}
Abstract: {abstract}
Review: {review}
Return valid JSON with arrays of short bullet strings.[/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=220,
    temperature=0.1,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

See `run_experiment.py` for the full prompt template and parsing helpers.

## Training Details

### Training Data

- Source: `data/train_clean.csv` and `data/val_clean.csv`, which are cleaned slices of an OpenReview-style conference export (`tp_2020conference.xlsx`). The cleaner removes URLs, section labels, and invalid ratings.
- Teacher Distillation: JSON targets are produced by zeroshot prompting `meta-llama/Llama-3.2-3B-Instruct` with deterministic sampling (`temperature=0.0`, `top_p=1.0`, `max_new_tokens=140`, up to two retries). This yields 500 train and 300 validation structured examples (`train_sw_targets.csv`, `val_sw_targets.csv`), each containing title, abstract, review, and JSON strings.
- Ratings: Numeric scores originate from the original spreadsheet when present; otherwise, the pipeline drops those rows for training purposes.

### Training Procedure

#### Preprocessing [optional]

- Text cleanup removes URLs, `Abstract:`/`Review:` prefixes, Markdown hashes, and collapses whitespace.
- Reviews lacking title/abstract pairs or valid numeric ratings are filtered out.
- During training the prompt always includes the human review (`MIX_USE_REVIEW_PROB = 1.0`) to reinforce extraction behavior.

#### Training Hyperparameters

- **Training regime:** bf16 compute, 4-bit quantized base model (bitsandbytes) with PEFT LoRA adapters
- **Epochs:** 1 full pass over the 500 teacher-labeled train rows
- **Effective batch size:** 8 examples per optimizer step (micro-batch 2 × gradient accumulation 4)
- **Learning rate:** 2e-4 with default cosine schedule from the Hugging Face `SFTTrainer`
- **LoRA config:** `r=8`, `alpha=16`, `dropout=0.05`, targeting all attention projection matrices
- **Sequence length:** inputs truncated/padded to 700 tokens; outputs capped at 220 tokens during supervised fine-tuning
- **Optimizer:** AdamW (default betas/eps from the trainer)

#### Speeds, Sizes, Times [optional]

- Training consumed <2 GPU-hours thanks to 4-bit loading of the base model.
- Adapter checkpoint size: ~8.8 MB (`adapter_model.safetensors`).
- Tokenizer files (~16 MB) are copied to simplify standalone use.

## Evaluation

Evaluation compares zero-shot behavior of the base model with the LoRA-adapted model on the 289 validation rows that contain both reviews and teacher JSON references.

### Testing Data, Factors & Metrics

#### Testing Data

- `data/val_sw_targets.csv` for strengths/weaknesses (301 rows; 289 valid generations during evaluation).
- `results/val_*` CSVs hold raw outputs, parsed JSON, and numeric rating predictions for both zeroshot and fine-tuned modes.

#### Factors

- Two generation modes (`zeroshot`, `ft`) and two scoring variants (raw logits vs. calibrated via Huber regression on train rate-only inference).
- Structured-output quality diagnostics: fallback usage, lexical diversity, cosine similarity with teacher JSON (requires embeddings).

#### Metrics

- Regression metrics on validation ratings: MAE, RMSE, Pearson correlation before/after calibration.
- Structured-text diagnostics: fallback rate (share of generations that fail to return JSON), type-token lexical diversity, cosine similarity with teacher JSON embeddings.

### Results

- Rating metrics (`results/final_metrics.json`):
  - Zeroshot raw MAE/RMSE/Pearson: 2.72 / 3.26 / 0.36
  - Zeroshot calibrated: 1.82 / 2.06 / 0.36
  - Fine-tuned raw: 3.05 / 3.69 / 0.07
  - Fine-tuned calibrated: 1.99 / 2.20 / -0.07
- Structured-output diagnostics (`results/sw_quality_metrics.json`, 289 samples):
  - Fallback rate: 0% (both modes produced valid JSON after retries)
  - Lexical diversity: 0.23 (zeroshot) vs. 0.20 (fine-tuned)
  - Teacher cosine similarity (with sentence-transformer embeddings): 0.54 (zeroshot) vs. 0.51 (fine-tuned)

#### Summary

The adapter successfully enforces consistent JSON formatting and focuses its generations on concise bullet lists, matching the teacher’s structure. However, the tiny fine-tuning set leads to modest or neutral gains on numeric rating prediction; calibrated zeroshot remains competitive. Use the adapter primarily for structured strengths/weaknesses extraction rather than for final scoring.

## Model Examination [optional]

No dedicated interpretability study has been run. The lexical diversity and cosine similarity diagnostics act as lightweight sanity checks for hallucination rates and adherence to teacher outputs.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Single consumer-grade CUDA GPU with ~10 GB of accessible VRAM (4-bit quantized loading)
- **Hours used:** <2 GPU-hours for the LoRA fine-tune
- **Cloud Provider:** None (local workstation)
- **Compute Region:** Not applicable / local
- **Carbon Emitted:** Not formally tracked; expected to be <0.5 kg CO2eq, but please recompute for your own training footprint

## Technical Specifications [optional]

### Model Architecture and Objective

- Base model: Llama 3.2 3B Instruct (decoder-only transformer, rotary embeddings, grouped-query attention).
- Objective: Supervised fine-tuning with log-likelihood loss on JSON-formatted targets describing strengths, weaknesses, and ratings.
- Adapter: LoRA applied to attention projection matrices (Q, K, V, O) and MLP gates using PEFT.

### Compute Infrastructure

#### Hardware

- Requires a GPU with ≥12 GB VRAM for training from scratch (4-bit quantization reduces memory usage to ~8–10 GB). Inference can run on a modern laptop GPU or CPU with reduced throughput.

#### Software

- Python 3.10+, PyTorch 2.2, Transformers 4.45, PEFT 0.12, bitsandbytes 0.43, Accelerate 0.34 (see `requirements.txt` for the exact pins used when the model was trained).

## Citation [optional]

Please cite both the project and the base model if you build on this work.

**BibTeX:**

```
@misc{bonomo2024automatedpaperreview,
  title        = {Fine-Tuning a Small LLM with LoRA for Automated Paper Review Insights},
  author       = {Camilla Bonomo},
  year         = {2024},
  howpublished = {\url{https://github.com/camillabonomo02/Automated_Paper_Review}}
}
```

```
@misc{meta2024llama3dot2,
  title        = {Llama 3.2 Technical Report},
  author       = {Meta AI},
  year         = {2024},
  howpublished = {\url{https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct}}
}
```

**APA:**

Camilla Bonomo. (2024). *Fine-Tuning a Small LLM with LoRA for Automated Paper Review* [Computer software]. https://github.com/camillabonomo02/Automated_Paper_Review  
Meta AI. (2024). *Llama 3.2 Technical Report* [Computer software]. https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

## Glossary [optional]

- **LoRA:** Low-Rank Adaptation, a parameter-efficient fine-tuning method that injects trainable rank-decomposed matrices into frozen transformer layers.
- **Teacher Distillation:** Using a stronger or zeroshot model to create pseudo-labels that supervise a smaller or specialized student.
- **Lexical Diversity:** Type-token ratio computed over generated text; higher values indicate less repetition.
- **Calibration:** Regression-based adjustment of raw model scores to better align with ground-truth ratings.

## More Information [optional]

For questions, open an issue in the repository or email the maintainer listed in `README.md`. Contributions (new datasets, prompts, evaluation ideas) are welcome via pull requests.

## Model Card Authors [optional]

Camilla Bonomo (project maintainer) with assistance from the Codex CLI automated documentation agent.
