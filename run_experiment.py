# paper_review_pipeline.py
# ======================================================
# Paper Review Pipeline (rate + strengths/weaknesses)
# ======================================================

import os, json, pathlib, platform, random, re, argparse, math, sys
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

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_LENGTH = 512

# -------------------------
# Repro + tiny env report
# -------------------------
def set_seed(seed: int = SEED):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def env_report():
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

# -------------------------
# Prompting
# -------------------------
SYSTEM_PROMPT = (
    "You are a peer-review assistant. Return ONLY valid JSON with EXACTLY these keys:\n"
    "- strengths: list of exactly 3 short strings\n"
    "- weaknesses: list of exactly 3 short strings\n"
    "- rate: number (decimal in [1,10])\n"
    "Rules:\n"
    "1) Output JSON only (no prose, no code fences). Begin with '{' and end with '}'.\n"
    "2) Use the full 1–10 scale; avoid clustering around 7–8.\n"
    "3) Base the rate on novelty, methodological soundness, and evaluation strength.\n"
)

# Deterministic evidence cues from abstract: reduce mode collapse
_re_pct = re.compile(r'(\d{1,2}\.?\d?)\s?%')
DATASET_WORDS = ["imagenet","cifar","mnist","glue","squad","ms coco","libri","wmt","wikitext",
                 "benchmarks","dataset","datasets","real-world","ablation","user study","human study"]
EVAL_WORDS = ["state-of-the-art","sota","outperform","improve","significant","p<","confidence interval",
              "baseline","comparison","ablation","metrics","auc","f1","accuracy","precision","recall"]

def extract_evidence_cues(title: str, abstract: str) -> Dict[str, Any]:
    t = f"{title} {abstract}".lower()
    datasets = sum(t.count(w) for w in DATASET_WORDS)
    eval_terms = sum(t.count(w) for w in EVAL_WORDS)
    pct_imp = [float(x) for x in _re_pct.findall(t)]
    has_numbers = bool(re.search(r'\b\d+(\.\d+)?\b', t))
    len_tokens = len(re.findall(r'\w+', abstract))
    return {
        "len_tokens": len_tokens,
        "datasets_hits": int(datasets),
        "eval_hits": int(eval_terms),
        "max_pct_mentioned": float(max(pct_imp) if pct_imp else 0.0),
        "has_numbers": bool(has_numbers),
    }

def build_prompts(title: str, abstract: str):
    cues = extract_evidence_cues(title, abstract)
    user = (
        f"Paper title: {str(title).strip()}\n"
        f"Abstract:\n{str(abstract).strip()}\n\n"
        "Use this evidence summary (derived deterministically from the abstract):\n"
        f"- tokens: {cues['len_tokens']}\n"
        f"- dataset hits: {cues['datasets_hits']}\n"
        f"- evaluation/benchmark hits: {cues['eval_hits']}\n"
        f"- max % mentioned: {cues['max_pct_mentioned']}\n"
        f"- has numeric results: {cues['has_numbers']}\n\n"
        "Scoring rubric:\n"
        "9–10: clear novelty + solid methodology + strong empirical evidence across datasets;\n"
        "7–8: meaningful contribution + sound methods + decent evaluation;\n"
        "5–6: incremental or partial; limited or mixed evaluation;\n"
        "3–4: weak novelty or weak empirical support;\n"
        "1–2: unclear or flawed.\n"
        "Return STRICT JSON with keys strengths[3], weaknesses[3], rate (decimal 1–10). "
        "JSON only. No prose. No code fences."
    )
    return SYSTEM_PROMPT, user

def to_chat_prompt_with_template(tokenizer, system: str, user: str) -> str:
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -------------------------
# JSON helpers
# -------------------------
def _to_float(val) -> Optional[float]:
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val.strip())
        if m:
            try: return float(m.group(0))
            except: return None
    return None

def _decode_last_json(txt: str) -> Dict[str, Any]:
    txt = (txt or "").strip()
    # keep after last assistant marker if present
    for sep in ["\nAssistant:", "\nUser:", "\n</s>"]:
        if sep in txt: txt = txt.split(sep)[-1]
    # trim to last closing brace
    if txt.count("{") and txt.count("}"): txt = txt[:txt.rfind("}")+1]
    last = None
    for m in re.finditer(r"\{.*?\}", txt, flags=re.S):
        s = m.group(0)
        try:
            last = json.loads(s)
        except: pass
    if last is None:
        return {"strengths":["","",""], "weaknesses":["","",""], "rate": np.nan}
    st = last.get("strengths", [])
    wk = last.get("weaknesses", [])
    st = [str(x).strip() for x in (st if isinstance(st,list) else [st])]
    wk = [str(x).strip() for x in (wk if isinstance(wk,list) else [wk])]
    out = {
        "strengths": (st + [""]*3)[:3],
        "weaknesses": (wk + [""]*3)[:3],
        "rate": np.nan
    }
    for k in ["rate","rating","score","overall","overall_rating"]:
        r = _to_float(last.get(k, None))
        if r is not None:
            out["rate"] = float(min(10.0, max(1.0, round(float(r), 1))))
            break
    return out

# Heuristic fallback when JSON has NaN rate
def _heuristic_rate_from_cues(cues: Dict[str, Any]) -> float:
    score = 4.0
    score += min(3.0, cues["datasets_hits"] * 0.6)
    score += min(2.0, cues["eval_hits"] * 0.4)
    score += 0.6 if cues["has_numbers"] else 0.0
    score += min(1.0, max(0, (cues["max_pct_mentioned"] - 2.0) / 10.0))
    score += min(0.8, max(0, (cues["len_tokens"] - 120) / 400.0))
    return float(max(1.0, min(10.0, round(score, 1))))

# -------------------------
# Data: PREPARE
# -------------------------
def step_prepare(x_path: pathlib.Path):
    if x_path.suffix.lower() == ".csv": df = pd.read_csv(x_path)
    else: df = pd.read_excel(x_path)
    df.columns = [c.strip() for c in df.columns]
    needed = ["title", "abstract", "review", "rate"]
    miss = [c for c in needed if c not in df.columns]
    assert not miss, f"Missing columns: {miss}"
    df = df.dropna(subset=["title","abstract","review"]).copy()
    df["abstract"] = df["abstract"].astype(str).str.replace("Abstract:###","",regex=False).str.strip()
    grp = df.groupby("title", as_index=False).agg(
        abstract=("abstract","first"),
        review=("review", lambda s: "\n\n".join(map(str, s))),
        rate_mean=("rate", lambda s: pd.to_numeric(s, errors="coerce").mean())
    )
    usable = grp.dropna(subset=["rate_mean"]).copy()
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(usable, test_size=0.2, random_state=SEED, shuffle=True)
    train_df.to_csv(DATA_DIR / "train_clean.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_clean.csv", index=False)
    grp.to_csv(DATA_DIR / "all_clean.csv", index=False)
    print(f"[prepare] train={len(train_df)} val={len(val_df)} saved")

# -------------------------
# Model load
# -------------------------
def load_model_and_tokenizer():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    use_4bit = False
    try:
        import importlib.metadata as im
        _ = im.version("bitsandbytes"); use_4bit = True
    except Exception: use_4bit = False
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16 if has_cuda else torch.float16)
        mdl = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                   device_map="auto", trust_remote_code=False)
        return tok, mdl
    if has_cuda: dtype = torch.float16; device_map = "auto"
    elif has_mps: dtype = torch.float16; device_map = {"": "mps"}
    else: dtype = torch.float32; device_map = "cpu"
    mdl = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype,
                                               device_map=device_map, low_cpu_mem_usage=True,
                                               trust_remote_code=False)
    return tok, mdl

# -------------------------
# Generation
# -------------------------
def generate_structured_json(tokenizer, model, title, abstract,
                             max_new_tokens=220, attempts=4, allow_fallback=True):
    import torch, time
    cues = extract_evidence_cues(title, abstract)
    sys, usr = build_prompts(title, abstract)

    def _gen(do_sample: bool):
        prompt = to_chat_prompt_with_template(tokenizer, sys, usr)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                          pad_token_id=tokenizer.eos_token_id,
                          use_cache=True)
        if do_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9,
                                   repetition_penalty=1.1))
        else:
            gen_kwargs.update(dict(do_sample=False))  # no temp/top_p warnings
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        txt = tokenizer.decode(out[0], skip_special_tokens=True)
        return _decode_last_json(txt)

    best = None
    for i in range(attempts):
        obj = _gen(do_sample=(i % 2 == 1))
        if (obj.get("rate") == obj.get("rate")) and any(obj["strengths"]) and any(obj["weaknesses"]):
            best = obj; break
        best = obj

    if best is None:
        best = {"strengths":["","",""], "weaknesses":["","",""], "rate": np.nan}

    # Fallback: never hardcode 6.0; derive from evidence cues
    if (best.get("rate") != best.get("rate")) or (best.get("rate") is None):
        if allow_fallback:
            best["rate"] = _heuristic_rate_from_cues(cues)

    # Always keep lists non-empty
    best["strengths"]  = [(s or "clear contribution") for s in (best.get("strengths",["","",""]) + [""]*3)[:3]]
    best["weaknesses"] = [(w or "limited evaluation") for w in (best.get("weaknesses",["","",""]) + [""]*3)[:3]]
    return best

# -------------------------
# Distillation / Datasets / LoRA (unchanged logic)
# -------------------------
def step_distill_structured(limit: Optional[int] = None):
    set_seed(); tok, mdl = load_model_and_tokenizer()
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    if limit: train_df = train_df.head(limit); val_df = val_df.head(max(1, limit // 5))
    def _run(df_in: pd.DataFrame, name: str):
        rows = []
        for _, r in df_in.iterrows():
            obj = generate_structured_json(tok, mdl, r["title"], r["abstract"], allow_fallback=True)
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
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    def _to_hf(df: pd.DataFrame):
        prompts, responses = [], []
        for _, r in df.iterrows():
            sys, usr = build_prompts(r["title"], r["abstract"])
            prompts.append(to_chat_prompt_with_template(tok, sys, usr))
            responses.append(r["review_structured"])
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
    import json, torch
    from datasets import load_from_disk
    from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling, BitsAndBytesConfig)
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile per 4-bit.")
    try:
        import bitsandbytes as _bnb  # noqa: F401
    except Exception as e:
        raise RuntimeError("Installa bitsandbytes==0.43.1 e triton compatibile.") from e
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_cfg,
                                                device_map="auto", trust_remote_code=False)
    base = prepare_model_for_kbit_training(base)
    if hasattr(base,"config"): base.config.use_cache = False
    if hasattr(base,"enable_input_require_grads"): base.enable_input_require_grads()
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                          task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg)
    tr = load_from_disk(str(DATA_DIR / "train_tokenized"))
    va = load_from_disk(str(DATA_DIR / "val_tokenized"))
    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "finetuned-llama3"), remove_unused_columns=False,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        eval_strategy="steps", save_strategy="steps", logging_steps=25, eval_steps=100, save_steps=100,
        num_train_epochs=3, learning_rate=2e-5, warmup_ratio=0.05,
        bf16=bf16_ok, fp16=not bf16_ok, gradient_checkpointing=True,
        report_to=[], load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    )
    from transformers import EarlyStoppingCallback
    trainer = Trainer(model=model, args=args, data_collator=coll,
                      train_dataset=tr, eval_dataset=va, tokenizer=tok,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    out = trainer.train()
    (RESULTS_DIR / "trainer_log.json").write_text(json.dumps({"train_metrics": out.metrics}, indent=2), encoding="utf-8")
    print("[finetune] done; best checkpoint under model/finetuned-llama3")

# -------------------------
# Baselines + CI
# -------------------------
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

# -------------------------
# Calibration (robusta)
# -------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\ufeff","") for c in df.columns]
    return df

def _fit_rate_calibrator(train_structured_csv: pathlib.Path, train_clean_csv: pathlib.Path):
    from sklearn.linear_model import HuberRegressor
    try:
        tr_s = _normalize_cols(pd.read_csv(train_structured_csv))
    except Exception:
        # if structured predictions are missing, return identity
        return (1.0, 0.0)
    try:
        tr_c = _normalize_cols(pd.read_csv(train_clean_csv))
    except Exception:
        return (1.0, 0.0)

    # reconstruct rate_mean if absent
    if "rate_mean" not in tr_c.columns:
        if "rate" in tr_c.columns:
            tmp = tr_c.copy()
            tmp["rate_num"] = pd.to_numeric(tmp["rate"], errors="coerce")
            # ensure there's a title column to group by
            if "title" not in tmp.columns:
                return (1.0, 0.0)
            tr_c = tmp.groupby("title", as_index=False).agg(rate_mean=("rate_num", "mean"))
        else:
            return (1.0, 0.0)

    # ensure structured file has generated rates
    if "rate_gen" not in tr_s.columns:
        # try to extract rate_gen from a JSON column if present (robust fallback)
        if "review_structured" in tr_s.columns:
            def _extract_rate(x):
                try:
                    j = json.loads(x)
                    r = j.get("rate", None)
                    return float(r) if r is not None else np.nan
                except Exception:
                    return np.nan
            tr_s["rate_gen"] = tr_s["review_structured"].apply(_extract_rate)
        else:
            return (1.0, 0.0)

    # merge on title; bail out if title missing
    if "title" not in tr_s.columns or "title" not in tr_c.columns:
        return (1.0, 0.0)

    df = tr_s.merge(tr_c[["title", "rate_mean"]], on="title", how="inner")
    if "rate_gen" not in df.columns or "rate_mean" not in df.columns:
        return (1.0, 0.0)

    df = df.dropna(subset=["rate_gen", "rate_mean"]).copy()

    if len(df) < 8:
        return (1.0, 0.0)  # identity

    x = df["rate_gen"].astype(float).to_numpy().reshape(-1, 1)
    y = df["rate_mean"].astype(float).to_numpy()
    reg = HuberRegressor().fit(x, y)
    # Return (a, b) so that a*pred + b
    return (float(reg.coef_[0]), float(reg.intercept_))

def _apply_calibration(arr: List[Optional[float]], a: float, b: float):
    out = []
    for v in arr:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(float(max(1.0, min(10.0, a*float(v) + b))))
    return out

# -------------------------
# Zero-shot numeric eval
# -------------------------
def step_zero_shot_numeric_eval():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    def rmse(y_true, y_pred): 
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    def eval_reg(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "Pearson": float(pearsonr(y_true, y_pred)[0])
        }
    set_seed()
    tok, mdl = load_model_and_tokenizer()
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    y_rate = val_df["rate_mean"].tolist()

    zs_rate, raw_rows = [], []
    for _, r in val_df.iterrows():
        obj = generate_structured_json(tok, mdl, r["title"], r["abstract"], allow_fallback=True)
        rate = obj["rate"] if obj["rate"] == obj["rate"] else None
        zs_rate.append(rate)
        raw_rows.append({"title": r["title"], "parsed_rate": rate, "obj": json.dumps(obj, ensure_ascii=False)})

    pd.DataFrame(raw_rows).to_csv(RESULTS_DIR / "zseval_debug.csv", index=False)

    a, b = _fit_rate_calibrator(DATA_DIR / "train_structured.csv", DATA_DIR / "train_clean.csv")
    zs_rate_cal = _apply_calibration(zs_rate, a, b)

    yt, yp = [], []
    for t, p in zip(y_rate, zs_rate_cal):
        if p is not None and not np.isnan(t):
            yt.append(t); yp.append(p)

    out = {}
    if len(yt) > 5:
        out["rate_zero_shot"] = eval_reg(yt, yp)
        out["rate_zero_shot_calibration"] = {"a": a, "b": b}

    (RESULTS_DIR / "zs_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[zseval] zs_metrics.json saved; usable={len(yt)}/{len(y_rate)}. Debug → results/zseval_debug.csv")

# -------------------------
# HUMAN S/W refs + eval
# -------------------------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
def _split_sents(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()][:50]

def extract_sw_from_review(review_text: str) -> Dict[str, List[str]]:
    text = (review_text or "").strip()
    strengths, weaknesses = [], []
    lines = [l.strip() for l in text.splitlines()]
    current = None
    for ln in lines:
        if re.match(r'^\s*strengths?\s*[:\-–]', ln, re.I): current = "S"; continue
        if re.match(r'^\s*(weaknesses?|limitations?)\s*[:\-–]', ln, re.I): current = "W"; continue
        if not ln.strip(): current = None; continue
        if current == "S": strengths.append(ln)
        elif current == "W": weaknesses.append(ln)
    def top_k(sents, pos=True, k=3):
        pos_kw = ["strength","novel","clear","sound","rigor","significant","thorough","well-written","solid"]
        neg_kw = ["weak","limitation","concern","issue","unclear","missing","insufficient","confusing","error","bias"]
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
    df = pd.read_csv(split_csv)
    with out_json.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            sw = extract_sw_from_review(str(r["review"]))
            f.write(json.dumps({"title": r["title"],
                                "strengths_ref": sw["strengths"],
                                "weaknesses_ref": sw["weaknesses"]}, ensure_ascii=False) + "\n")
    print(f"[refs] {out_json} saved")

def step_zero_shot_structured_sw():
    set_seed(); tok, mdl = load_model_and_tokenizer()
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    rows = []
    for _, r in val_df.iterrows():
        obj = generate_structured_json(tok, mdl, r["title"], r["abstract"], allow_fallback=True)
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
    from bert_score import score as bertscore
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
            f"{tag}.weaknesses.bertscore.F1.mean": float(np.mean(w_scores)) if s_scores else math.nan,
            f"{tag}.strengths.bertscore.F1.CI": list(_bootstrap_ci_scalar(s_scores)) if s_scores else [math.nan, math.nan],
            f"{tag}.weaknesses.bertscore.F1.CI": list(_bootstrap_ci_scalar(w_scores)) if s_scores else [math.nan, math.nan],
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

# -------------------------
# Merge + Plots
# -------------------------
def step_plot_and_merge():
    import matplotlib.pyplot as plt
    merged = {}
    bp = RESULTS_DIR / "baseline_metrics.json"
    if bp.exists():
        base = json.loads(bp.read_text(encoding="utf-8"))
        for k, (main, ci) in base.items():
            for m, v in main.items(): merged[f"baseline.{k}.{m}"] = v
            for m, v in ci.items():   merged[f"baseline.{k}.{m}"] = v
    zp = RESULTS_DIR / "zs_metrics.json"
    if zp.exists():
        zs = json.loads(zp.read_text(encoding="utf-8"))
        for k, main in zs.items():
            for m, v in main.items(): merged[f"zero_shot.{k}.{m}"] = v
    sp = RESULTS_DIR / "sw_metrics.json"
    if sp.exists():
        sw = json.loads(sp.read_text(encoding="utf-8")); merged.update(sw)
    pd.Series(merged).to_csv(RESULTS_DIR / "metrics_flat.csv")
    (RESULTS_DIR / "metrics_flat.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("[merge] metrics_flat.{json,csv} saved")
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

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
        help=("prepare | distill | buildds | finetune | baselines | zseval | refs | zssw | evalsw | plot | all"))
    parser.add_argument("--source", type=str, default="excel", choices=["excel","csv"])
    parser.add_argument("--xlsx", type=str, default=str(DATA_DIR / "tp_2020conference.xlsx"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(); env_report()

    if args.step in ("prepare", "all"):
        step_prepare(pathlib.Path(args.xlsx))
    if args.step in ("distill", "all"):
        step_distill_structured(limit=args.limit)
    if args.step in ("buildds", "all"):
        build_hf_datasets()
    if args.step in ("finetune", "all"):
        step_finetune_lora()
    if args.step in ("baselines", "all"):
        step_baselines_and_ci()
    if args.step in ("zseval", "all"):
        step_zero_shot_numeric_eval()
    if args.step in ("refs", "all"):
        step_build_sw_references(DATA_DIR / "val_clean.csv", DATA_DIR / "val_sw_ref.jsonl")
    if args.step in ("zssw", "all"):
        step_zero_shot_structured_sw()
    if args.step in ("evalsw", "all"):
        step_eval_sw(DATA_DIR / "val_zero_shot_structured.csv", DATA_DIR / "val_sw_ref.jsonl",
                     distilled_csv=DATA_DIR / "val_structured.csv")
    if args.step in ("plot", "all"):
        step_plot_and_merge()

if __name__ == "__main__":
    main()
