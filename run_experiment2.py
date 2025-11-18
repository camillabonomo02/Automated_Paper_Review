# run_experiment.py
# ======================================================
# Proposal-2: S/W + Rate from Title+Abstract
# Model: LLaMA 3.2 3B Instruct with LoRA (4-bit)
#
# Pipeline steps:
#   prepare   -> clean & split (train/val) with rate_mean per title
#   sup       -> build supervised targets from human reviews (+rate_mean)
#   buildds   -> tokenize masked chat datasets (JSON target)
#   finetune  -> LoRA SFT on JSON {strengths, weaknesses, rate}
#   zstrain   -> zero-shot JSON on TRAIN (for calibration)
#   zseval    -> zero-shot JSON on VAL
#   infer_ft  -> finetuned JSON on VAL
#   refs      -> human S/W references for VAL (for BERTScore)
#   evalsw    -> BERTScore best-match (ZS vs FT)
#   evalrate  -> regression metrics (Baselines, ZS raw/cal, FT raw/cal)
#   plot      -> simple comparison plots
#   all       -> run everything
# ======================================================

import os, re, json, math, random, platform, argparse, pathlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------- Paths & Globals ----------
SEED = 42
ROOT = pathlib.Path(".").resolve()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "model"
for d in (DATA_DIR, RESULTS_DIR, MODEL_DIR): d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_LEN = 1024

# ---------- Repro & Env ----------
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

# ---------- Prompting ----------
SYSTEM_PROMPT = (
    "You are a peer-review assistant. Return ONLY valid JSON with EXACTLY these keys:\n"
    "- strengths: list of up to 3 short strings\n"
    "- weaknesses: list of up to 3 short strings\n"
    "- rate: number (decimal in [1,10])\n"
    "Rules:\n"
    "1) Output JSON only (no prose, no code fences).\n"
    "2) Be specific; avoid vague adjectives.\n"
)

def build_prompts(title: str, abstract: str) -> Tuple[str, str]:
    user = (
        f"Paper title: {str(title).strip()}\n"
        f"Abstract:\n{str(abstract).strip()}\n\n"
        "Task: From title+abstract only, predict likely Strengths, Weaknesses, and a numeric rate [1,10]\n"
        "that a human reviewer would give. Return STRICT JSON with keys 'strengths' (<=3), 'weaknesses' (<=3), 'rate' (decimal)."
    )
    return SYSTEM_PROMPT, user

def to_chat_prompt_with_template(tokenizer, system: str, user: str) -> str:
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ---------- Cleaning & Target Extraction ----------
_WS = re.compile(r"\s+")
_SEC_HDR = re.compile(r"^\s*(review|summary|overview|strengths?|pros|advantages?|weakness(es)?|cons|limitations?|comments?|questions?)\s*[:\-–#]*\s*$", re.I)
_MD_BULLET = re.compile(r"^\s*([\-\*\•]|\d{1,2}\.)\s+")
_URL = re.compile(r"https?://\S+")
_BRACKETS = re.compile(r"\[[^\]]{0,120}\]|\([^)]{0,120}\)")
_CONTRAST = re.compile(r"\b(however|but|though|nevertheless|yet|nonetheless|albeit)\b", re.I)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

_POS = [
    r"\bnovel(ty)?\b", r"\bstrong\b", r"\bclear(?!ly\s*not)\b", r"\bsound\b", r"\bthorough\b",
    r"\brigor(ous)?\b", r"\binsight(ful)?\b", r"\buseful\b", r"\bsignificant\b", r"\bwell[-\s]?written\b"
]
_NEG = [
    r"\bweak(ness)?\b", r"\blimitation(s)?\b", r"\bconcern(s)?\b", r"\bissue(s)?\b", r"\bflaw(s)?\b",
    r"\bunclear\b", r"\bnot\s+clear\b", r"\bmissing\b", r"\binsufficient\b", r"\berror(s)?\b", r"\bbias(ed)?\b",
    r"\bconfound(ing)?\b", r"\black(s)?\b", r"\bno(t)?\s+evidence\b"
]
_POS_RE = re.compile("|".join(_POS), re.I)
_NEG_RE = re.compile("|".join(_NEG), re.I)
_GENERIC_STR = re.compile(r"^(the paper (is|was)|well[-\s]?written|interesting|good|nice)\b", re.I)

def _clean_abstract(a: str) -> str:
    """Stricter: remove 'Abstract:###' once, any leftover '###', URLs, collapse whitespace."""
    a = str(a or "")
    if a.startswith("Abstract:###"):
        a = a[len("Abstract:###"):]
    a = a.replace("Abstract:###", " ")
    a = a.replace("###", " ")
    a = _URL.sub(" ", a)
    a = _WS.sub(" ", a).strip()
    return a

def _strip_review_prefix(t: str) -> str:
    """Only remove 'Review:###' prefix/markers; KEEP section headers for S/W parsing."""
    s = str(t or "")
    if s.startswith("Review:###"):
        s = s[len("Review:###"):]
    s = s.replace("Review:###", " ")
    s = s.replace("###", " ")
    return _WS.sub(" ", s).strip()

def _preclean_review(text: str) -> str:
    """Aggressive cleaner for polarity fallback; used *inside* extraction, not in prepare."""
    t = str(text or "")
    if t.startswith("Review:###"):
        t = t[len("Review:###"):]
    t = t.replace("Review:###", " ").replace("###", " ")
    t = _URL.sub(" ", t)
    out = []
    for ln in t.splitlines():
        ln = _MD_BULLET.sub("", ln).strip()
        if _SEC_HDR.match(ln):
            continue
        out.append(ln)
    t = " ".join(out)
    t = _BRACKETS.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _split_sentences_and_clauses(t: str) -> List[str]:
    sents = []
    for s in _SENT_SPLIT.split(t):
        s = s.strip()
        if not s: continue
        if _CONTRAST.search(s):
            parts = re.split(_CONTRAST, s)
            for p in parts:
                p = p.strip(" ,;:—-")
                if 10 <= len(p) <= 280: sents.append(p)
        else:
            if 10 <= len(s) <= 280: sents.append(s)
    return sents[:120]

def _score_polarity(s: str) -> Tuple[int,int]:
    pos = len(_POS_RE.findall(s))
    neg = len(_NEG_RE.findall(s))
    if "?" in s: neg += 1
    if pos > 0 and neg > 0: neg += 1  # mixed → lean negative
    return pos, neg

def _diverse_topk(cands: List[str], k=3) -> List[str]:
    if not cands: return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = vec.fit_transform(cands)
        keep = []
        for i in range(len(cands)):
            if not keep: keep.append(i); continue
            sim = cosine_similarity(X[i], X[keep]).max()
            if sim < 0.75: keep.append(i)
            if len(keep) >= k: break
        return [cands[i] for i in keep][:k]
    except Exception:
        out, seen = [], set()
        for s in cands:
            if s not in seen:
                out.append(s); seen.add(s)
            if len(out) >= k: break
        return out

def _section_parse(review_text: str) -> Dict[str, List[str]]:
    """Prefer explicit Strengths/Weaknesses sections if present."""
    lines = [l.strip() for l in (review_text or "").splitlines()]
    sections = {}
    current = None
    for ln in lines:
        if re.match(r"^\s*(strengths?|pros|advantages?)\s*[:\--#]*\s*$", ln, re.I): current = "S"; continue
        if re.match(r"^\s*(weakness(es)?|cons|limitations?)\s*[:\--#]*\s*$", ln, re.I): current = "W"; continue
        if re.match(r"^\s*(summary|overview|review|comments?|questions?)\s*[:\--#]*\s*$", ln, re.I): current = None; continue
        if not ln: continue
        if current:
            ln = _MD_BULLET.sub("", ln)
            ln = _BRACKETS.sub(" ", ln)
            ln = _URL.sub(" ", ln)
            ln = _WS.sub(" ", ln).strip()
            sections.setdefault(current, []).append(ln)
    out = {"S": sections.get("S", []), "W": sections.get("W", [])}
    return out

def extract_sw_from_review(review_text: str, k=3) -> Dict[str, List[str]]:
    """Deterministic, section-first extractor; falls back to polarity scoring."""
    # Section parse first (on original, only prefix-stripped review stored in dataset)
    sec = _section_parse(review_text)
    S_sec = [s for s in sec["S"] if 10 <= len(s) <= 280 and not GENERIC_STR.match(s)]
    W_sec = [s for s in sec["W"] if 10 <= len(s) <= 280]
    if S_sec or W_sec:
        S = _diverse_topk(S_sec, k)
        W = _diverse_topk(W_sec, k)
        return {"strengths": S, "weaknesses": W}

    # Fallback: polarity-based scorer on aggressively cleaned review
    t = _preclean_review(review_text)
    sents = _split_sentences_and_clauses(t)
    pos_cand, neg_cand = [], []
    for s in sents:
        if GENERIC_STR.match(s):  # drop generic praise
            continue
        pos, neg = _score_polarity(s)
        if pos==0 and neg==0: continue
        if neg > pos or "?" in s:
            neg_cand.append((neg - pos + (1 if "?" in s else 0), s))
        else:
            pos_cand.append((pos - neg, s))
    pos_cand.sort(key=lambda x: (-x[0], len(x[1])))
    neg_cand.sort(key=lambda x: (-x[0], len(x[1])))
    S = _diverse_topk([s for _,s in pos_cand][:10], k)
    W = _diverse_topk([s for _,s in neg_cand][:10], k)
    return {"strengths": S, "weaknesses": W}

# --- prepare: keep headers, only strip prefixes; use numeric rate column strictly
def step_prepare(x_path: pathlib.Path):
    # load
    if x_path.suffix.lower() == ".csv":
        df = pd.read_csv(x_path)
    else:
        df = pd.read_excel(x_path)
    df.columns = [c.strip() for c in df.columns]

    # must have these
    needed = ["title", "abstract", "review"]
    miss = [c for c in needed if c not in df.columns]
    assert not miss, f"Missing columns: {miss}"

    # clean text fields
    df = df.dropna(subset=["title","abstract","review"]).copy()
    df["abstract"] = df["abstract"].map(_clean_abstract)
    # IMPORTANT: keep section headers; only strip the Review:### prefix markers
    df["review"]   = df["review"].map(_strip_review_prefix)

    # choose the numeric rate column from the sheet
    rate_cols = [c for c in df.columns if c.lower() in {"rate","rating","score","overall","overall_rating"}]
    assert rate_cols, "No numeric rate column found (expected one of: rate, rating, score, overall, overall_rating)."
    rc = rate_cols[0]
    df["rate_num"] = pd.to_numeric(df[rc], errors="coerce")  # keep only true numbers

    # group by title
    grp = df.groupby("title", as_index=False).agg(
        abstract=("abstract","first"),
        review=("review", lambda s: "\n\n".join(map(str, s))),
        rate_mean=("rate_num", "mean")  # mean over only numeric rows
    )

    # split reproducibly
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(grp, test_size=0.2, random_state=SEED, shuffle=True)

    # save
    train_df.to_csv(DATA_DIR / "train_clean.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_clean.csv", index=False)

    # quick sanity report
    cov_train = (~train_df["rate_mean"].isna()).mean()
    cov_val   = (~val_df["rate_mean"].isna()).mean()
    print(f"[prepare] train={len(train_df)} val={len(val_df)} | rate coverage: "
          f"train={cov_train:.1%} val={cov_val:.1%}")

# ---------- Supervised Targets (TRAIN & VAL) ----------
def step_make_supervised_targets(src_csv: pathlib.Path, out_csv: pathlib.Path, k=3, is_train=False):
    df = pd.read_csv(src_csv)
    rows = []
    for _, r in df.iterrows():
        sw = extract_sw_from_review(str(r["review"]), k=k)
        rate = r.get("rate_mean", np.nan)
        rate = float(rate) if pd.notna(rate) else np.nan

        # For TRAIN: require numeric rate so model truly learns rate
        if is_train and np.isnan(rate):
            continue

        tgt = {
            "strengths": (sw["strengths"] + [""]*k)[:k],
            "weaknesses": (sw["weaknesses"] + [""]*k)[:k],
            "rate": None if np.isnan(rate) else float(round(min(10.0, max(1.0, rate)), 2))
        }
        rows.append({
            "title": r["title"],
            "abstract": r["abstract"],
            "rate_mean": None if np.isnan(rate) else float(rate),
            "target_json": json.dumps(tgt, ensure_ascii=False)
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[sup] {out_csv} saved (rows kept: {len(out)})")

# ---------- Datasets & LoRA ----------
def build_hf_datasets():
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    train_sup = pd.read_csv(DATA_DIR / "train_supervised.csv")
    val_sup   = pd.read_csv(DATA_DIR / "val_supervised.csv")

    def _to_hf(df):
        prompts, responses = [], []
        for _, r in df.iterrows():
            sys, usr = build_prompts(r["title"], r["abstract"])
            prompts.append(to_chat_prompt_with_template(tok, sys, usr))
            responses.append(r["target_json"])
        return Dataset.from_dict({"prompt": prompts, "response": responses})

    tr = _to_hf(train_sup); va = _to_hf(val_sup)

    def tokenize(example):
        prompt_ids = tok(example["prompt"], add_special_tokens=False).input_ids
        resp_ids   = tok(example["response"], add_special_tokens=False).input_ids
        input_ids = (prompt_ids + resp_ids)[:MAX_LEN]
        labels    = ([-100]*len(prompt_ids) + resp_ids)[:MAX_LEN]
        attn      = [1]*len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    tr_tok = tr.map(tokenize, remove_columns=tr.column_names)
    va_tok = va.map(tokenize, remove_columns=va.column_names)
    tr_tok.save_to_disk(str(DATA_DIR / "train_tokenized"))
    va_tok.save_to_disk(str(DATA_DIR / "val_tokenized"))
    print("[buildds] tokenized datasets saved")

def finetune_lora():
    import torch
    from datasets import load_from_disk
    from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling, BitsAndBytesConfig)
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for 4-bit LoRA training.")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_cfg,
                                                device_map="auto", trust_remote_code=False)
    base = prepare_model_for_kbit_training(base)
    if hasattr(base, "config"): base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"): base.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora_cfg)

    tr = load_from_disk(str(DATA_DIR / "train_tokenized"))
    va = load_from_disk(str(DATA_DIR / "val_tokenized"))
    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    from transformers import EarlyStoppingCallback
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "finetuned-llama3"),
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="steps", save_strategy="steps",
        logging_steps=25, eval_steps=100, save_steps=100,
        num_train_epochs=3, learning_rate=2e-5, warmup_ratio=0.05,
        bf16=bf16_ok, fp16=not bf16_ok, gradient_checkpointing=True,
        report_to=[], load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
    )
    trainer = Trainer(model=model, args=args, data_collator=coll,
                      train_dataset=tr, eval_dataset=va, tokenizer=tok,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    model.save_pretrained(str(MODEL_DIR / "finetuned-llama3"))
    print("[finetune] done; adapters under model/finetuned-llama3")

# ---------- Generation (ZS & FT) ----------
GEN_KW = dict(max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9)
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _to_float(val) -> Optional[float]:
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        m = _NUM_RE.search(val.strip())
        if m:
            try: return float(m.group(0))
            except: return None
    return None

def _decode_last_json(text: str) -> Dict[str, Any]:
    txt = (text or "").strip()
    if txt.count("{") and txt.count("}"): txt = txt[:txt.rfind("}")+1]
    last = None
    for m in re.finditer(r"\{.*?\}", txt, flags=re.S):
        s = m.group(0)
        try: last = json.loads(s)
        except: pass
    if last is None: return {"strengths": [], "weaknesses": [], "rate": None}
    S = last.get("strengths", []); W = last.get("weaknesses", []); R = last.get("rate", None)
    S = [str(x).strip() for x in (S if isinstance(S, list) else [S])]
    W = [str(x).strip() for x in (W if isinstance(W, list) else [W])]
    r = _to_float(R)
    if r is not None: r = float(min(10.0, max(1.0, round(r, 2))))
    return {"strengths": [s for s in S if s][:3], "weaknesses": [w for w in W if w][:3], "rate": r}

def _gen_rows(df_in: pd.DataFrame, tok, mdl) -> List[Dict[str,Any]]:
    rows = []
    device = mdl.device if hasattr(mdl, "device") else "cpu"
    for _, r in df_in.iterrows():
        sys, usr = build_prompts(r["title"], r["abstract"])
        prompt = to_chat_prompt_with_template(tok, sys, usr)
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = mdl.generate(**inputs, **GEN_KW, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        obj = _decode_last_json(text)
        rows.append({
            "title": r["title"],
            "review_structured": json.dumps({"strengths":obj["strengths"], "weaknesses":obj["weaknesses"]}, ensure_ascii=False),
            "parsed_rate": obj["rate"]
        })
    return rows

def load_zeroshot_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype,
                                               device_map="auto", trust_remote_code=False)
    return tok, mdl

def load_finetuned_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    dtype = torch.bfloat16 if bf16_ok and torch.cuda.is_available() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype,
                                                device_map="auto", trust_remote_code=False)
    model = PeftModel.from_pretrained(base, str(MODEL_DIR / "finetuned-llama3"))
    return tok, model

# ---------- Steps: ZS/FT inference ----------
def step_zero_shot_train_for_calibration():
    set_seed()
    tok, mdl = load_zeroshot_model()
    tr = pd.read_csv(DATA_DIR / "train_clean.csv")
    rows = _gen_rows(tr, tok, mdl)
    pd.DataFrame(rows).to_csv(DATA_DIR / "train_structured.csv", index=False)
    print("[zstrain] data/train_structured.csv")

def step_zero_shot_val():
    set_seed()
    tok, mdl = load_zeroshot_model()
    va = pd.read_csv(DATA_DIR / "val_clean.csv")
    rows = _gen_rows(va, tok, mdl)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "val_zeroshot_structured.csv", index=False)
    print("[zseval] results/val_zeroshot_structured.csv")

def step_finetuned_val():
    set_seed()
    tok, mdl = load_finetuned_model()
    va = pd.read_csv(DATA_DIR / "val_clean.csv")
    rows = _gen_rows(va, tok, mdl)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "val_ft_structured.csv", index=False)
    print("[infer_ft] results/val_ft_structured.csv")

# ---------- S/W Evaluation ----------
def _flatten_sw(obj_json: str, key: str) -> List[str]:
    try:
        o = json.loads(obj_json); arr = o.get(key, [])
        return [str(x).strip() for x in (arr if isinstance(arr, list) else [arr]) if str(x).strip()]
    except: return []

def _bootstrap_ci_scalar(values: List[float], alpha=0.05, seed=SEED) -> Tuple[float, float]:
    rng = np.random.default_rng(seed); arr = np.array(values, dtype=float)
    if len(arr) == 0: return (math.nan, math.nan)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(np.mean(arr[idx]))
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def _bertscore_bestmatch(cands: List[str], refs: List[str], lang="en"):
    from bert_score import score as bertscore
    cands = [c for c in cands if c.strip()]; refs = [r for r in refs if r.strip()]
    if not cands or not refs: return float("nan")
    _, _, F1 = bertscore(cands, refs, lang=lang, rescale_with_baseline=True)
    F = F1.numpy() if hasattr(F1, "numpy") else np.array(F1)
    used_r = set(); scores = []
    # greedy best matching
    if F.ndim == 1:  # equal length fallback
        return float(np.mean(F))
    for i in range(len(cands)):
        best_j, best_val = None, -1.0
        for j in range(len(refs)):
            if j in used_r: continue
            v = float(F[i, j])
            if v > best_val: best_val, best_j = v, j
        if best_j is not None:
            used_r.add(best_j); scores.append(best_val)
    return float(np.mean(scores)) if scores else float("nan")

def step_build_sw_references(split_csv: pathlib.Path, out_json: pathlib.Path):
    df = pd.read_csv(split_csv)
    with out_json.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            sw = extract_sw_from_review(str(r["review"]))
            f.write(json.dumps({"title": r["title"],
                                "strengths_ref": sw["strengths"],
                                "weaknesses_ref": sw["weaknesses"]}, ensure_ascii=False) + "\n")
    print(f"[refs] {out_json} saved")

def step_eval_sw():
    ref_path = DATA_DIR / "val_sw_ref.jsonl"
    refs = {}
    with ref_path.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line); refs[j["title"]] = j

    def _eval(csv_path: pathlib.Path, tag: str):
        if not csv_path.exists(): return {}
        df = pd.read_csv(csv_path)
        s_scores, w_scores = [], []
        for _, r in df.iterrows():
            ref = refs.get(r["title"])
            if not ref: continue
            cs = _flatten_sw(r["review_structured"], "strengths")
            cw = _flatten_sw(r["review_structured"], "weaknesses")
            rs = ref["strengths_ref"]; rw = ref["weaknesses_ref"]
            s = _bertscore_bestmatch(cs, rs); w = _bertscore_bestmatch(cw, rw)
            if not (np.isnan(s) or np.isinf(s)): s_scores.append(float(s))
            if not (np.isnan(w) or np.isinf(w)): w_scores.append(float(w))
        out = {}
        if s_scores:
            out[f"{tag}.strengths.bertscore.F1.mean"] = float(np.mean(s_scores))
            out[f"{tag}.strengths.bertscore.F1.CI"] = list(_bootstrap_ci_scalar(s_scores))
        if w_scores:
            out[f"{tag}.weaknesses.bertscore.F1.mean"] = float(np.mean(w_scores))
            out[f"{tag}.weaknesses.bertscore.F1.CI"] = list(_bootstrap_ci_scalar(w_scores))
        return out

    merged = {}
    merged.update(_eval(RESULTS_DIR / "val_zeroshot_structured.csv", "zeroshot"))
    merged.update(_eval(RESULTS_DIR / "val_ft_structured.csv", "finetuned"))
    (RESULTS_DIR / "sw_metrics.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    flat_p = RESULTS_DIR / "metrics_flat.json"
    base = json.loads(flat_p.read_text(encoding="utf-8")) if flat_p.exists() else {}
    base.update(merged)
    flat_p.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print("[evalsw] sw_metrics.json saved & merged")

# ---------- Rate Evaluation ----------
def _rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _eval_reg(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": _rmse(yt, yp),
        "R2": float(r2_score(yt, yp)),
        "Pearson": float(pearsonr(yt, yp)[0]) if len(yt) > 1 else float("nan"),
        "N": int(len(yt)),
    }

def _bootstrap_ci(metric_fn, y_true, y_pred, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(SEED)
    yt, yp = np.array(y_true), np.array(y_pred); n = len(yt)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(yt[idx], yp[idx]))
    lo, hi = np.quantile(vals, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def _fit_rate_calibrator(train_structured_csv: pathlib.Path, train_clean_csv: pathlib.Path) -> Tuple[float,float]:
    from sklearn.linear_model import HuberRegressor
    if not (train_structured_csv.exists() and train_clean_csv.exists()):
        return (1.0, 0.0)
    tr_s = pd.read_csv(train_structured_csv)  # has parsed_rate (ZS on train)
    tr_c = pd.read_csv(train_clean_csv)       # has rate_mean
    if "parsed_rate" not in tr_s.columns or "rate_mean" not in tr_c.columns: return (1.0, 0.0)
    df = tr_s.merge(tr_c[["title","rate_mean"]], on="title", how="inner").dropna(subset=["parsed_rate","rate_mean"])
    if len(df) < 8: return (1.0, 0.0)
    x = df["parsed_rate"].astype(float).to_numpy().reshape(-1,1)
    y = df["rate_mean"].astype(float).to_numpy()
    reg = HuberRegressor().fit(x, y)
    return (float(reg.coef_[0]), float(reg.intercept_))

def _apply_calibration(arr: List[Optional[float]], a: float, b: float):
    out = []
    for v in arr:
        if v is None or (isinstance(v,float) and np.isnan(v)): out.append(None)
        else: out.append(float(max(1.0, min(10.0, a*float(v)+b))))
    return out

def step_eval_rate():
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv").dropna(subset=["rate_mean"])
    y_true_map = dict(zip(val_df["title"], val_df["rate_mean"]))

    def _load_pred(csv_path: pathlib.Path):
        if not csv_path.exists(): return {}
        df = pd.read_csv(csv_path)
        return {r["title"]: (None if pd.isna(r["parsed_rate"]) else float(r["parsed_rate"])) for _,r in df.iterrows()}

    zs = _load_pred(RESULTS_DIR / "val_zeroshot_structured.csv")
    ft = _load_pred(RESULTS_DIR / "val_ft_structured.csv")

    # align helper
    def _align(pred_map):
        yt, yp = [], []
        for t, y in y_true_map.items():
            p = pred_map.get(t, None)
            if p is not None:
                yt.append(float(y)); yp.append(float(p))
        return yt, yp

    yt_zs, yp_zs = _align(zs)
    yt_ft, yp_ft = _align(ft)

    # baselines
    tr = pd.read_csv(DATA_DIR / "train_clean.csv").dropna(subset=["rate_mean"])
    mean_pred = float(np.nanmean(tr["rate_mean"]))
    yt_base = list(y_true_map.values())
    yp_mean = [mean_pred]*len(yt_base)

    from sklearn.linear_model import LinearRegression
    Xtr = tr["abstract"].str.len().to_numpy().reshape(-1,1)
    ytr = tr["rate_mean"].astype(float).to_numpy()
    Xva = val_df["abstract"].str.len().to_numpy().reshape(-1,1)
    reg = LinearRegression().fit(Xtr, ytr)
    yp_lin_full = reg.predict(Xva)
    yp_lin = [float(p) for p in yp_lin_full]
    yt_lin = yt_base

    # calibration
    a, b = _fit_rate_calibrator(DATA_DIR / "train_structured.csv", DATA_DIR / "train_clean.csv")
    yp_zs_cal = _apply_calibration(yp_zs, a, b) if yp_zs else []
    yp_ft_cal = _apply_calibration(yp_ft, a, b) if yp_ft else []

    def pack_metrics(tag, yt, yp):
        if not yt or not yp: return {}
        m = _eval_reg(yt, yp)
        ci = {
            "MAE_CI": list(_bootstrap_ci(lambda a,b: _eval_reg(a,b)["MAE"], np.array(yt), np.array(yp))),
            "RMSE_CI": list(_bootstrap_ci(lambda a,b: _eval_reg(a,b)["RMSE"], np.array(yt), np.array(yp))),
        }
        return {tag: {**m, **ci}}

    metrics = {}
    metrics.update(pack_metrics("baseline.mean", yt_base, yp_mean))
    metrics.update(pack_metrics("baseline.len_linear", yt_lin, yp_lin))
    metrics.update(pack_metrics("zeroshot.raw", yt_zs, yp_zs))
    metrics.update(pack_metrics("zeroshot.cal", yt_zs, yp_zs_cal))
    metrics.update(pack_metrics("finetuned.raw", yt_ft, yp_ft))
    metrics.update(pack_metrics("finetuned.cal", yt_ft, yp_ft_cal))
    (RESULTS_DIR / "rate_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # merge flat
    flat_p = RESULTS_DIR / "metrics_flat.json"
    flat = json.loads(flat_p.read_text(encoding="utf-8")) if flat_p.exists() else {}
    for k, v in metrics.items():
        for kk, vv in v.items():
            flat[f"rate.{k}.{kk}"] = vv
    flat_p.write_text(json.dumps(flat, indent=2), encoding="utf-8")
    print("[evalrate] rate_metrics.json saved & merged")

# ---------- Plots ----------
def step_plot_and_merge():
    import matplotlib.pyplot as plt
    merged = {}

    sp = RESULTS_DIR / "sw_metrics.json"
    if sp.exists(): merged.update(json.loads(sp.read_text(encoding="utf-8")))
    rp = RESULTS_DIR / "rate_metrics.json"
    if rp.exists():
        rate = json.loads(rp.read_text(encoding="utf-8"))
        for tag, vals in rate.items():
            for k, v in vals.items():
                merged[f"rate.{tag}.{k}"] = v

    pd.Series(merged).to_csv(RESULTS_DIR / "metrics_flat.csv")
    (RESULTS_DIR / "metrics_flat.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("[merge] metrics_flat.{json,csv} saved")

    # SW bar
    keys_sw = [
        ("zeroshot.strengths.bertscore.F1.mean", "ZS Strengths"),
        ("zeroshot.weaknesses.bertscore.F1.mean", "ZS Weaknesses"),
        ("finetuned.strengths.bertscore.F1.mean", "FT Strengths"),
        ("finetuned.weaknesses.bertscore.F1.mean", "FT Weaknesses"),
    ]
    vals, lbls = [], []
    for k, lab in keys_sw:
        v = merged.get(k)
        if v is not None: vals.append(v); lbls.append(lab)
    if vals:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(lbls, rotation=15, ha="right")
        ax.set_ylim(0, 1.0); ax.set_ylabel("BERTScore F1 (↑)")
        ax.set_title("S/W — Zero-shot vs Fine-tuned")
        plt.tight_layout(); fig.savefig(RESULTS_DIR / "sw_bertscore_plot.png", dpi=160)
        print("[plot] sw_bertscore_plot.png saved")

    # Rate MAE bar
    mae_keys = [
        ("rate.baseline.mean.MAE", "BL Mean"),
        ("rate.baseline.len_linear.MAE", "BL LenLin"),
        ("rate.zeroshot.raw.MAE", "ZS raw"),
        ("rate.zeroshot.cal.MAE", "ZS cal"),
        ("rate.finetuned.raw.MAE", "FT raw"),
        ("rate.finetuned.cal.MAE", "FT cal"),
    ]
    rvals, rlbls = [], []
    for k, lab in mae_keys:
        v = merged.get(k)
        if v is not None: rvals.append(v); rlbls.append(lab)
    if rvals:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(len(rvals)), rvals)
        ax.set_xticks(range(len(rvals))); ax.set_xticklabels(rlbls, rotation=15, ha="right")
        ax.set_ylabel("MAE (↓)")
        ax.set_title("Rate — Baselines vs Zero-shot vs Fine-tuned")
        plt.tight_layout(); fig.savefig(RESULTS_DIR / "rate_mae_plot.png", dpi=160)
        print("[plot] rate_mae_plot.png saved")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=str, default="all",
                   help="prepare | sup | buildds | finetune | zstrain | zseval | refs | infer_ft | evalsw | evalrate | plot | all")
    p.add_argument("--xlsx", type=str, default=str(DATA_DIR / "tp_2020conference.xlsx"))
    args = p.parse_args()

    set_seed(); env_report()

    if args.step in ("prepare", "all"):
        step_prepare(pathlib.Path(args.xlsx))
    if args.step in ("sup", "all"):
        step_make_supervised_targets(DATA_DIR / "train_clean.csv", DATA_DIR / "train_supervised.csv", is_train=True)
        step_make_supervised_targets(DATA_DIR / "val_clean.csv",   DATA_DIR / "val_supervised.csv",   is_train=False)
    if args.step in ("buildds", "all"):
        build_hf_datasets()
    if args.step in ("finetune", "all"):
        finetune_lora()
    if args.step in ("zstrain", "all"):
        step_zero_shot_train_for_calibration()
    if args.step in ("zseval", "all"):
        step_zero_shot_val()
    if args.step in ("refs", "all"):
        step_build_sw_references(DATA_DIR / "val_clean.csv", DATA_DIR / "val_sw_ref.jsonl")
    if args.step in ("infer_ft", "all"):
        step_finetuned_val()
    if args.step in ("evalsw", "all"):
        step_eval_sw()
    if args.step in ("evalrate", "all"):
        step_eval_rate()
    if args.step in ("plot", "all"):
        step_plot_and_merge()

if __name__ == "__main__":
    main()