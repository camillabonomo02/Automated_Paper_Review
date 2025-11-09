"""
Paper Review Pipeline (rate + strengths/weaknesses) — concise, proposal-aligned

What this script does:
1) PREPARE: Read your Excel/CSV (Seafoodair/OpenReview 2020) and make train/val splits.
   Required columns in the file: title, abstract, review, rate
2) DISTILL: Ask a small LLM for STRICT JSON with {strengths[3], weaknesses[3], rate:float}.
3) DATASETS + LORA: Build HF datasets and fine-tune with LoRA (small, fast).
4) NUMERIC EVAL: Baselines + zero-shot regression metrics for 'rate' (MAE/RMSE/R2/Pearson + CI).
5) HUMAN REFS: Extract human Strengths/Weaknesses from the review text (deterministic heuristic).
6) S/W EVAL: Compare model S/W (zero-shot and distilled) to human S/W with BERTScore (± CI).
7) PLOTS: Merge metrics and render a simple MAE plot (rate) + BERTScore plot (S/W).

CLI (common):
  python paper_review_pipeline.py --step prepare --source excel --xlsx data/tp_2020conference.xlsx
  python paper_review_pipeline.py --step distill --limit 200
  python paper_review_pipeline.py --step buildds
  python paper_review_pipeline.py --step finetune
  python paper_review_pipeline.py --step baselines
  python paper_review_pipeline.py --step zseval
  python paper_review_pipeline.py --step refs
  python paper_review_pipeline.py --step zssw
  python paper_review_pipeline.py --step evalsw
  python paper_review_pipeline.py --step plot
"""

# =========================
# Imports & Config
# =========================
import os, json, pathlib, platform, random, re, argparse, math
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import numpy as np
import pandas as pd

SEED = 42
ROOT = pathlib.Path(".").resolve()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "model"
for d in (DATA_DIR, RESULTS_DIR, MODEL_DIR): d.mkdir(exist_ok=True, parents=True)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # small instruct model as requested
MAX_LENGTH = 512
STRUCT_SAMPLES = 20  # quick sanity-size for S/W sampling when needed

# =========================
# Repro + Env
# =========================
def set_seed(seed: int = SEED):
    """Deterministic-ish runs."""
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def env_report():
    """Tiny run fingerprint."""
    import torch
    rpt = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / "env_report.json").write_text(json.dumps(rpt, indent=2), encoding="utf-8")

# =========================
# Strict JSON schema (+ parser)
# =========================
# We now ONLY predict 'rate' numerically; schema reflects that.
# Keep SCHEMA_EXAMPLE as-is (with normal { } braces)
SCHEMA_EXAMPLE = (
    '{"strengths":["clear contribution","solid methodology","useful ablation"],'
    '"weaknesses":["limited dataset","unclear baseline","insufficient error analysis"],'
    '"rate":6.0}'
)

SYSTEM_PROMPT = (
    "You are a peer-review assistant. Return ONLY a valid JSON object with EXACTLY these keys:\n"
    "strengths: list of exactly 3 short strings\n"
    "weaknesses: list of exactly 3 short strings\n"
    "rate: number (float)\n"
    "Use concise, paper-agnostic language. Output ONLY the JSON object."
)

def build_prompts(title: str, abstract: str):
    # We avoid str.format() on a string that contains JSON braces.
    user = (
        "Paper title: " + str(title).strip() + "\n"
        "Abstract:\n" + str(abstract).strip() + "\n\n"
        "Return STRICT JSON with this exact schema:\n" + SCHEMA_EXAMPLE
    )
    return SYSTEM_PROMPT, user

def to_chat_template(system: str, user: str) -> str:
    """Simple chat format that many instruct models accept."""
    return f"<|SYSTEM|>\n{system}\n<|USER|>\n{user}\n<|ASSISTANT|>\n"

def _to_float(val) -> Optional[float]:
    """Coerce messy strings (e.g., '6: Weak Accept') into floats, if possible."""
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val.strip())
        if m:
            try: return float(m.group(0))
            except: return None
    return None

def strict_json_rate(text: str) -> Dict[str, Any]:
    """
    Extract first JSON object and force the schema:
      - strengths/weaknesses: 3 strings each (pad/truncate)
      - rate: float or NaN if not recoverable
    """
    text = text.strip()
    m = re.search(r'\{.*\}', text, flags=re.S)
    obj = {"strengths": ["", "", ""], "weaknesses": ["", "", ""], "rate": np.nan}
    if not m: return obj
    try:
        raw = json.loads(m.group(0))
        st = raw.get("strengths", []); wk = raw.get("weaknesses", [])
        st = [str(x).strip() for x in (st if isinstance(st, list) else [st])]
        wk = [str(x).strip() for x in (wk if isinstance(wk, list) else [wk])]
        obj["strengths"] = (st + [""]*3)[:3]
        obj["weaknesses"] = (wk + [""]*3)[:3]
        r = _to_float(raw.get("rate", None))
        obj["rate"] = r if r is not None else np.nan
    except: pass
    return obj

# =========================
# Data: PREPARE
# =========================
def step_prepare(x_path: pathlib.Path):
    """
    Build train/val from Excel/CSV. We require *exactly* these columns:
      title, abstract, review, rate
    We group by title (paper), concatenate reviews, and compute mean(rate) as 'rate_mean'.
    """
    # Load
    if x_path.suffix.lower() == ".csv":
        df = pd.read_csv(x_path)
    else:
        df = pd.read_excel(x_path)

    # Validate columns
    needed = ["title", "abstract", "review", "rate"]
    miss = [c for c in needed if c not in df.columns]
    assert not miss, f"Missing columns: {miss}"

    # Basic cleaning
    df = df.dropna(subset=["title", "abstract", "review"]).copy()
    df["abstract"] = df["abstract"].astype(str).str.replace("Abstract:###", "", regex=False).str.strip()

    # Group by paper:
    # - abstract: take first
    # - review:   concatenate all reviews for that paper
    # - rate:     numeric mean across the group's reviews
    grp = df.groupby("title", as_index=False).agg(
        abstract=("abstract", "first"),
        review=("review", lambda s: "\n\n".join(map(str, s))),
        rate_mean=("rate", lambda s: pd.to_numeric(s, errors="coerce").mean())
    )

    # Keep only rows with a valid mean rate
    usable = grp.dropna(subset=["rate_mean"]).copy()

    # Train/val split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(usable, test_size=0.2, random_state=SEED, shuffle=True)

    # Save
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(DATA_DIR / "train_clean.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_clean.csv", index=False)
    grp.to_csv(DATA_DIR / "all_clean.csv", index=False)
    print(f"[prepare] train={len(train_df)} val={len(val_df)} saved")

# =========================
# Model: load + generate STRICT JSON
# =========================
def load_model_and_tokenizer():
    """
    Load the base model + tokenizer.
    - If bitsandbytes is available (Linux), use 4-bit NF4 quantization.
    - Otherwise, fall back to a regular load with an appropriate dtype on CUDA/MPS/CPU.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Detect backends
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Try 4-bit on Linux if bitsandbytes is available
    use_4bit = False
    BitsAndBytesConfig = None
    try:
        from transformers import BitsAndBytesConfig  # may fail on macOS
        # also ensure the wheels exist; importlib.metadata will be called internally by HF
        import importlib.metadata as im
        _ = im.version("bitsandbytes")
        use_4bit = True
    except Exception:
        use_4bit = False
        BitsAndBytesConfig = None

    if use_4bit:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if has_cuda else torch.float16
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=False,
        )
        return tok, mdl

    # Fallback path (macOS MPS / CPU / plain CUDA)
    # Choose dtype conservatively for stability across backends
    if has_cuda:
        dtype = torch.float16
        device_map = "auto"
    elif has_mps:
        # MPS prefers float16 but some models still run better in float32; start with fp16
        dtype = torch.float16
        device_map = {"": "mps"}
    else:
        dtype = torch.float32
        device_map = "cpu"

    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    return tok, mdl


def generate_structured_json(tokenizer, model, title: str, abstract: str,
                             max_new_tokens=192, temperature=0.1, top_p=0.9):
    """Ask the LLM for STRICT JSON; parse safely into our schema ({S,W,rate})."""
    import torch
    sys, usr = build_prompts(title, abstract)
    prompt = to_chat_template(sys, usr)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p,
                             pad_token_id=tokenizer.eos_token_id, use_cache=True)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return strict_json_rate(text)

# =========================
# Distillation → Datasets → LoRA
# =========================
def step_distill_structured(limit: Optional[int] = None):
    """Create STRICT JSON targets for train & val (teacher → {S/W, rate})."""
    set_seed(); tok, mdl = load_model_and_tokenizer()
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    if limit: train_df = train_df.head(limit); val_df = val_df.head(max(1, limit // 5))

    def _run(df_in: pd.DataFrame, name: str):
        rows = []
        for _, r in df_in.iterrows():
            obj = generate_structured_json(tok, mdl, r["title"], r["abstract"])
            rows.append({
                "title": r["title"],
                "abstract": r["abstract"],
                "review_structured": json.dumps(obj, ensure_ascii=False),
                "rate_gen": obj["rate"],
                "rate_mean": r.get("rate_mean", np.nan),
            })
        pd.DataFrame(rows).to_csv(DATA_DIR / f"{name}_structured.csv", index=False)
    _run(train_df, "train"); _run(val_df, "val")
    print("[distill] train_structured.csv / val_structured.csv")

def build_hf_datasets():
    """Turn (prompt + STRICT JSON) into tokenized datasets for LoRA fine-tuning."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def _to_hf(df: pd.DataFrame):
        prompts, responses = [], []
        for _, r in df.iterrows():
            sys, usr = build_prompts(r["title"], r["abstract"])
            prompts.append(to_chat_template(sys, usr))
            responses.append(r["review_structured"])  # STRICT JSON string
        return Dataset.from_dict({"prompt": prompts, "response": responses})

    tr = _to_hf(pd.read_csv(DATA_DIR / "train_structured.csv"))
    va = _to_hf(pd.read_csv(DATA_DIR / "val_structured.csv"))

    def tokenize(example):
        text = example["prompt"] + example["response"]
        toks = tok(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
        toks["labels"] = toks["input_ids"].copy()
        return toks

    tr_tok = tr.map(tokenize, remove_columns=tr.column_names)
    va_tok = va.map(tokenize, remove_columns=va.column_names)
    tr_tok.save_to_disk(str(DATA_DIR / "train_tokenized"))
    va_tok.save_to_disk(str(DATA_DIR / "val_tokenized"))
    print("[buildds] tokenized datasets saved")

def step_finetune_lora():
    """
    LoRA fine-tuning that works with and without bitsandbytes.
    - If bitsandbytes is present: k-bit (4-bit) path.
    - Else: full-precision-ish path on CUDA/MPS/CPU (may need a smaller BASE_MODEL).
    """
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoTokenizer, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling, AutoModelForCausalLM
    )
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

    set_seed()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Detect backends
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    # Try bitsandbytes
    use_4bit = False
    try:
        from transformers import BitsAndBytesConfig
        import importlib.metadata as im
        _ = im.version("bitsandbytes")
        use_4bit = True
    except Exception:
        use_4bit = False

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if has_cuda else torch.float16
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=False
        )
        model = prepare_model_for_kbit_training(base)
    else:
        # Fallback: regular load
        if has_cuda:
            dtype = torch.float16; device_map = "auto"
        elif has_mps:
            dtype = torch.float16; device_map = {"": "mps"}
        else:
            dtype = torch.float32; device_map = "cpu"

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True, trust_remote_code=False
        )
        model = base  # no k-bit prep needed

    # Attach LoRA adapters
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # Data
    tr = load_from_disk(str(DATA_DIR / "train_tokenized"))
    va = load_from_disk(str(DATA_DIR / "val_tokenized"))
    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # TrainingArgs: avoid bf16/fp16 on MPS/CPU
    use_bf16 = has_cuda  # bf16 only meaningful on recent NVIDIA
    use_fp16 = has_cuda  # avoid fp16 flags on MPS/CPU

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "finetuned-llama3"),
        remove_unused_columns=False,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=25,
        eval_steps=100,
        save_steps=100,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    from transformers import EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=coll,
        train_dataset=tr,
        eval_dataset=va,
        tokenizer=tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    out = trainer.train()
    (RESULTS_DIR / "trainer_log.json").write_text(
        json.dumps({"train_metrics": out.metrics}, indent=2), encoding="utf-8"
    )
    print("[finetune] done; best checkpoint under model/finetuned-llama3")

# =========================
# Numeric baselines + zero-shot numeric (rate)
# =========================
def _rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _bootstrap_ci(metric_fn, y_true, y_pred, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(SEED)
    yt, yp = np.array(y_true), np.array(y_pred); n = len(yt)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(yt[idx], yp[idx]))
    lo, hi = np.quantile(vals, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def _eval_reg(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": _rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(pearsonr(y_true, y_pred)[0]),
    }

def step_baselines_and_ci():
    """Baselines for 'rate': mean-predictor and linear(length(abstract))."""
    from sklearn.linear_model import LinearRegression
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")

    y_rate = val_df["rate_mean"].tolist()
    mean_pred = [float(np.mean(train_df["rate_mean"]))] * len(y_rate)

    Xtr = train_df["abstract"].str.len().to_numpy().reshape(-1,1)
    Xva = val_df["abstract"].str.len().to_numpy().reshape(-1,1)
    reg = LinearRegression().fit(Xtr, train_df["rate_mean"])
    lin_pred = reg.predict(Xva)

    def with_ci(y, p):
        m = _eval_reg(y, p); ci = {}
        for key, fn in [("MAE", lambda a,b: _eval_reg(a,b)["MAE"]),
                        ("RMSE", lambda a,b: _eval_reg(a,b)["RMSE"])]:
            lo, hi = _bootstrap_ci(lambda a,b: fn(a,b), np.array(y), np.array(p))
            ci[f"{key}_CI"] = [lo, hi]
        return (m, ci)

    metrics = {
        "rate_mean_baseline": with_ci(y_rate, mean_pred),
        "rate_len_linear":    with_ci(y_rate, lin_pred),
    }
    (RESULTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[baselines] baseline_metrics.json saved")

def step_zero_shot_numeric_eval():
    """Zero-shot regression on 'rate' using STRICT JSON outputs."""
    set_seed(); tok, mdl = load_model_and_tokenizer()
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    y_rate = val_df["rate_mean"].tolist()

    zs_rate = []
    for _, r in val_df.iterrows():
        o = generate_structured_json(tok, mdl, r["title"], r["abstract"])
        zs_rate.append(o["rate"] if o["rate"] == o["rate"] else None)

    yt, yp = [], []
    for t, p in zip(y_rate, zs_rate):
        if p is not None and not np.isnan(t):
            yt.append(t); yp.append(p)

    out = {}
    if len(yt) > 5: out["rate_zero_shot"] = _eval_reg(yt, yp)
    (RESULTS_DIR / "zs_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[zseval] zs_metrics.json saved")

# =========================
# HUMAN S/W refs + zero-shot S/W + S/W eval
# =========================
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
def _split_sents(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()][:50]

def extract_sw_from_review(review_text: str) -> Dict[str, List[str]]:
    """
    Build HUMAN S/W references deterministically from the real review text:
    1) Read sections after 'Strengths:' / 'Weaknesses:' if present.
    2) Otherwise, rank sentences by simple positive/negative cue keywords.
    Always returns exactly 3 strengths and 3 weaknesses.
    """
    text = (review_text or "").strip()
    strengths, weaknesses = [], []

    # Try explicit headers first
    lines = [l.strip() for l in text.splitlines()]
    current = None
    for ln in lines:
        if re.match(r'^\s*strengths?\s*[:\-–]', ln, re.I): current = "S"; continue
        if re.match(r'^\s*(weaknesses?|limitations?)\s*[:\-–]', ln, re.I): current = "W"; continue
        if not ln.strip(): current = None; continue
        if current == "S": strengths.append(ln)
        elif current == "W": weaknesses.append(ln)

    # Fallback: lightweight keyword scoring
    def top_k(sents, pos=True, k=3):
        pos_kw = ["strength", "novel", "clear", "sound", "rigor", "significant", "thorough", "well-written", "solid"]
        neg_kw = ["weak", "limitation", "concern", "issue", "unclear", "missing", "insufficient", "confusing", "error", "bias"]
        out = []
        for s in sents:
            toks = s.lower(); kws = pos_kw if pos else neg_kw
            sc = sum(toks.count(w) for w in kws)
            sc += 1 if (("+" in s or "✓" in s) if pos else ("-" in s or "✗" in s)) else 0
            out.append((sc, s))
        out.sort(key=lambda x: (-x[0], len(x[1])))
        return [s for sc, s in out if sc > 0][:k]

    if len(strengths) < 3 or len(weaknesses) < 3:
        sents = _split_sents(text)
        if len(strengths) < 3: strengths = (strengths + top_k(sents, pos=True,  k=3))[:3]
        if len(weaknesses) < 3: weaknesses = (weaknesses + top_k(sents, pos=False, k=3))[:3]

    strengths = (strengths + [""]*3)[:3]
    weaknesses = (weaknesses + [""]*3)[:3]
    return {"strengths": strengths, "weaknesses": weaknesses}

def step_build_sw_references(split_csv: pathlib.Path, out_json: pathlib.Path):
    """Create HUMAN S/W refs (JSONL) from the real 'review' column in the chosen split."""
    df = pd.read_csv(split_csv)
    with out_json.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            sw = extract_sw_from_review(str(r["review"]))
            f.write(json.dumps({"title": r["title"],
                                "strengths_ref": sw["strengths"],
                                "weaknesses_ref": sw["weaknesses"]}, ensure_ascii=False) + "\n")
    print(f"[refs] {out_json} saved")

def step_zero_shot_structured_sw():
    """Generate zero-shot STRICT-JSON ({S/W, rate}) for the *validation* set."""
    set_seed(); tok, mdl = load_model_and_tokenizer()
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    rows = []
    for _, r in val_df.iterrows():
        obj = generate_structured_json(tok, mdl, r["title"], r["abstract"])
        rows.append({"title": r["title"], "review_structured": json.dumps(obj, ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(DATA_DIR / "val_zero_shot_structured.csv", index=False)
    print("[zssw] val_zero_shot_structured.csv saved")

def _flatten_sw(obj_json: str, key: str) -> List[str]:
    try:
        o = json.loads(obj_json); arr = o.get(key, [])
        return [str(x).strip() for x in (arr if isinstance(arr, list) else [arr])]
    except: return ["", "", ""]

def _pairwise(cands: List[str], refs: List[str]) -> Tuple[List[str], List[str]]:
    cands = (cands + [""]*3)[:3]; refs = (refs + [""]*3)[:3]; return cands, refs

def _bootstrap_ci_scalar(values: List[float], alpha=0.05, seed=SEED) -> Tuple[float, float]:
    rng = np.random.default_rng(seed); arr = np.array(values, dtype=float)
    if len(arr) == 0: return (math.nan, math.nan)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(np.mean(arr[idx]))
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def step_eval_sw(model_csv: pathlib.Path, ref_jsonl: pathlib.Path,
                 distilled_csv: Optional[pathlib.Path] = DATA_DIR / "val_structured.csv"):
    """
    Evaluate S/W text quality with BERTScore:
      - ZERO-SHOT vs HUMAN refs
      - (If available) DISTILLED vs HUMAN refs
    Save sw_metrics.json and merge into metrics_flat.json.
    """
    from bert_score import score as bertscore

    # Load human refs
    refs = {}
    with ref_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line); refs[j["title"]] = j

    def _eval_one(csv_path: pathlib.Path, tag: str) -> Dict[str, Any]:
        df = pd.read_csv(csv_path)
        s_scores, w_scores = [], []
        for _, r in df.iterrows():
            ref = refs.get(r["title"]); 
            if not ref: continue
            cs = _flatten_sw(r["review_structured"], "strengths")
            cw = _flatten_sw(r["review_structured"], "weaknesses")
            rs = ref["strengths_ref"]; rw = ref["weaknesses_ref"]
            cs, rs = _pairwise(cs, rs); cw, rw = _pairwise(cw, rw)
            _, _, F1s = bertscore(cs, rs, lang="en", rescale_with_baseline=True)
            _, _, F1w = bertscore(cw, rw, lang="en", rescale_with_baseline=True)
            s_scores.append(float(F1s.mean())); w_scores.append(float(F1w.mean()))
        out = {
            f"{tag}.strengths.bertscore.F1.mean": float(np.mean(s_scores)) if s_scores else math.nan,
            f"{tag}.weaknesses.bertscore.F1.mean": float(np.mean(w_scores)) if w_scores else math.nan,
            f"{tag}.strengths.bertscore.F1.CI": list(_bootstrap_ci_scalar(s_scores)) if s_scores else [math.nan, math.nan],
            f"{tag}.weaknesses.bertscore.F1.CI": list(_bootstrap_ci_scalar(w_scores)) if w_scores else [math.nan, math.nan],
        }
        return out

    merged = {}
    merged.update(_eval_one(model_csv, "zero_shot"))
    if distilled_csv and pathlib.Path(distilled_csv).exists():
        merged.update(_eval_one(distilled_csv, "distilled"))

    metrics_path = RESULTS_DIR / "metrics_flat.json"
    if metrics_path.exists():
        base = json.loads(metrics_path.read_text(encoding="utf-8")); base.update(merged)
        metrics_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
    else:
        metrics_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (RESULTS_DIR / "sw_metrics.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("[evalsw] sw_metrics.json saved (merged into metrics_flat.json)")

# =========================
# Merge + Plots
# =========================
def step_plot_and_merge():
    """Gather metrics and make two small plots: MAE (rate) and BERTScore (S/W)."""
    import matplotlib.pyplot as plt

    merged = {}
    # baselines
    bp = RESULTS_DIR / "baseline_metrics.json"
    if bp.exists():
        base = json.loads(bp.read_text(encoding="utf-8"))
        for k, (main, ci) in base.items():
            for m, v in main.items(): merged[f"baseline.{k}.{m}"] = v
            for m, v in ci.items():   merged[f"baseline.{k}.{m}"] = v
    # zero-shot numeric
    zp = RESULTS_DIR / "zs_metrics.json"
    if zp.exists():
        zs = json.loads(zp.read_text(encoding="utf-8"))
        for k, main in zs.items():
            for m, v in main.items(): merged[f"zero_shot.{k}.{m}"] = v
    # S/W
    sp = RESULTS_DIR / "sw_metrics.json"
    if sp.exists():
        sw = json.loads(sp.read_text(encoding="utf-8")); merged.update(sw)

    pd.Series(merged).to_csv(RESULTS_DIR / "metrics_flat.csv")
    (RESULTS_DIR / "metrics_flat.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("[merge] metrics_flat.{json,csv} saved")

    # MAE plot for 'rate'
    pairs = [
        ("baseline.rate_mean_baseline.MAE", "baseline.rate_mean_baseline.MAE_CI", "Mean BL"),
        ("baseline.rate_len_linear.MAE",    "baseline.rate_len_linear.MAE_CI",    "Len Linear"),
        ("zero_shot.rate_zero_shot.MAE",    None,                                 "Zero-shot"),
    ]
    xs, means, lows, highs, labels = [], [], [], [], []
    for i, (m, ci, lab) in enumerate(pairs):
        mean = merged.get(m, None)
        if mean is None: continue
        xs.append(i); means.append(mean); labels.append(lab)
        if ci and merged.get(ci): lo, hi = merged[ci]
        else: lo, hi = mean, mean
        lows.append(lo); highs.append(hi)

    if xs:
        fig, ax = plt.subplots(figsize=(8,5))
        for i, mean in enumerate(means):
            ax.bar(i, mean, width=0.5)
            lo, hi = lows[i], highs[i]
            ax.errorbar(i, mean, yerr=[[mean-lo],[hi-mean]], fmt="none", capsize=5)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("MAE (±CI)"); ax.set_title("Rate — Baselines vs Zero-shot")
        plt.tight_layout(); fig.savefig(RESULTS_DIR / "rate_mae_ci_plot.png", dpi=160)
        print("[plot] rate_mae_ci_plot.png saved")

    # S/W BERTScore plot
    sw_keys = [
        ("zero_shot.strengths.bertscore.F1.mean", "ZS Strengths"),
        ("zero_shot.weaknesses.bertscore.F1.mean", "ZS Weaknesses"),
        ("distilled.strengths.bertscore.F1.mean",  "LoRA Strengths"),
        ("distilled.weaknesses.bertscore.F1.mean", "LoRA Weaknesses"),
    ]
    vals, lbls = [], []
    for k, lab in sw_keys:
        v = merged.get(k); 
        if v is not None: vals.append(v); lbls.append(lab)
    if vals:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(lbls, rotation=15, ha="right")
        ax.set_ylim(0, 1.0); ax.set_ylabel("BERTScore F1")
        ax.set_title("Strengths/Weaknesses — BERTScore")
        plt.tight_layout(); fig.savefig(RESULTS_DIR / "sw_bertscore_plot.png", dpi=160)
        print("[plot] sw_bertscore_plot.png saved")

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
                        help=("prepare | distill | buildds | finetune | "
                              "baselines | zseval | refs | zssw | evalsw | plot | all"))
    parser.add_argument("--source", type=str, default="excel", choices=["excel","csv"],
                        help="Use your local Seafoodair/OpenReview file (excel or csv)")
    parser.add_argument("--xlsx", type=str, default=str(DATA_DIR / "tp_2020conference.xlsx"),
                        help="Path to Excel/CSV with columns: title, abstract, review, rate")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick runs in distill")
    args = parser.parse_args()

    set_seed(); env_report()

    # 1) Prepare
    if args.step in ("prepare", "all"):
        step_prepare(pathlib.Path(args.xlsx))

    # 2–4) Distill → HF → LoRA
    if args.step in ("distill", "all"):  step_distill_structured(limit=args.limit)
    if args.step in ("buildds", "all"):  build_hf_datasets()
    if args.step in ("finetune", "all"): step_finetune_lora()

    # 5) Baselines + 6) Zero-shot numeric (rate)
    if args.step in ("baselines", "all"): step_baselines_and_ci()
    if args.step in ("zseval", "all"):    step_zero_shot_numeric_eval()

    # 7) HUMAN refs + zero-shot S/W + S/W eval
    if args.step in ("refs", "all"):
        step_build_sw_references(DATA_DIR / "val_clean.csv", DATA_DIR / "val_sw_ref.jsonl")
    if args.step in ("zssw", "all"):
        step_zero_shot_structured_sw()
    if args.step in ("evalsw", "all"):
        step_eval_sw(DATA_DIR / "val_zero_shot_structured.csv", DATA_DIR / "val_sw_ref.jsonl",
                     distilled_csv=DATA_DIR / "val_structured.csv")

    # 8) Merge + plots
    if args.step in ("plot", "all"): step_plot_and_merge()

if __name__ == "__main__":
    main()
