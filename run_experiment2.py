# run_experiment2.py
import os
import re
import json
import random
import pathlib
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ML imports
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# HF imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset


# ---------------------------
# CONFIG
# ---------------------------
class Config:
    SEED = 42
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_SEQ_LEN = 700  # include anche review in alcuni esempi per LoRA

    ROOT = pathlib.Path(".").resolve()
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results"
    MODEL_DIR = ROOT / "model"

    # LoRA params
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

    # Training params
    LR = 2e-4
    EPOCHS = 1
    BATCH_SIZE = 2
    GRAD_ACC_STEPS = 4

    # Teacher extraction (zeroshot)
    # più bassi = più veloci; se JSON tronchi alzi solo di poco
    TEACHER_MAX_NEW_TOKENS = 140
    TEACHER_BATCH_SIZE = 12
    TEACHER_TEMPERATURE = 0.0     # deterministico
    TEACHER_TOP_P = 1.0

    # Fai teacher su VAL solo su un subset per risparmiare tempo
    TEACHER_VAL_MAX_EXAMPLES = 300  # metti None per farli tutti (lento)

    # Resume teacher: se file esiste continua
    TEACHER_RESUME = True

    # Inference full
    MAX_NEW_TOKENS_FULL = 220
    INFER_BATCH_SIZE = 8
    INFER_TEMPERATURE = 0.1

    # Mix prompt training
    # Prob che un esempio includa anche la review in input
    MIX_USE_REVIEW_PROB = 0.5

    # Retry policy JSON teacher
    TEACHER_RETRY_MAX = 2
    TEACHER_RETRY_TOKENS_BOOST = 80  # se fallisce, aumenta tokens così

    def __init__(self):
        for d in (self.DATA_DIR, self.RESULTS_DIR, self.MODEL_DIR):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()


# ---------------------------
# UTILS
# ---------------------------
def set_seed(seed: int = CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield i, iterable[i:i+n]


# ---------------------------
# 1. DATA PROCESSOR
# ---------------------------
class DataProcessor:
    """
    Pipeline:
      1) load raw, clean, split train/val
      2) create S/W teacher targets (zeroshot) on train + small val subset
      3) save train_sw_targets.csv / val_sw_targets.csv
    """

    @staticmethod
    def clean_text(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"https?://\S+", "", text)
        text = text.replace("Abstract:", "").replace("Review:", "")
        text = text.replace("###", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def parse_rate(val: Any) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None

        s = str(val).lower().strip().replace(",", ".")

        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", s)
        if match:
            num, den = float(match.group(1)), float(match.group(2))
            if den == 100:
                return num / 10.0
            if den == 10:
                return num
            if den > 0:
                return (num / den) * 10.0

        try:
            floats = re.findall(r"\d+\.?\d*", s)
            valid = [float(f) for f in floats if 1.0 <= float(f) <= 10.0]
            if valid:
                return valid[-1]
        except ValueError:
            pass

        labels = {
            "strong accept": 9.5,
            "accept": 8.0,
            "weak accept": 7.0,
            "borderline": 5.5,
            "weak reject": 4.0,
            "reject": 2.5,
            "strong reject": 1.0,
        }
        for k, v in labels.items():
            if k in s:
                return v

        return None

    def prepare_data(self, file_path: str):
        logging.info(f"Loading data from {file_path}...")

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df.columns = [c.strip().lower() for c in df.columns]
        rate_cols = [c for c in df.columns if any(x in c for x in ["rate", "score", "overall"])]
        logging.info(f"Rating columns found: {rate_cols}")

        processed = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
            rating = None
            for c in rate_cols:
                parsed = self.parse_rate(row.get(c))
                if parsed is not None:
                    rating = parsed
                    break

            processed.append(
                {
                    "title": self.clean_text(row.get("title", "")),
                    "abstract": self.clean_text(row.get("abstract", "")),
                    "review": self.clean_text(row.get("review", "")),
                    "rate": rating,
                }
            )

        df_clean = pd.DataFrame(processed).dropna(subset=["title", "abstract"])
        train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=CFG.SEED)

        train_df.to_csv(CFG.DATA_DIR / "train_clean.csv", index=False)
        val_df.to_csv(CFG.DATA_DIR / "val_clean.csv", index=False)
        logging.info(f"Split saved. Train: {len(train_df)}, Val: {len(val_df)}")

    def create_sw_targets_from_teacher(self):
        """
        Teacher vede SOLO title+abstract.
        Usa resume e val-subset per velocità.
        """
        logging.info("Creating S/W targets using ZERO-SHOT teacher (title+abstract only)...")

        tok = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        teacher = AutoModelForCausalLM.from_pretrained(
            CFG.MODEL_ID, quantization_config=bnb, device_map="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=teacher,
            tokenizer=tok,
            max_new_tokens=CFG.TEACHER_MAX_NEW_TOKENS,
            temperature=CFG.TEACHER_TEMPERATURE,
            top_p=CFG.TEACHER_TOP_P,
            do_sample=False,
            return_full_text=False,
            batch_size=CFG.TEACHER_BATCH_SIZE,
        )

        mm = ModelManager()

        for split in ["train", "val"]:
            df = pd.read_csv(CFG.DATA_DIR / f"{split}_clean.csv")

            # per val prendi solo subset (veloce)
            if split == "val" and CFG.TEACHER_VAL_MAX_EXAMPLES is not None:
                df = df.sample(CFG.TEACHER_VAL_MAX_EXAMPLES, random_state=CFG.SEED)

            out_path = CFG.DATA_DIR / f"{split}_sw_targets.csv"

            rows = []
            start_idx = 0

            # resume
            if CFG.TEACHER_RESUME and out_path.exists():
                old = pd.read_csv(out_path)
                rows = old.to_dict("records")
                start_idx = len(old)
                logging.info(f"Resuming teacher for {split} at idx {start_idx}.")

            for i, batch in batched(df.iloc[start_idx:].reset_index(drop=True), CFG.TEACHER_BATCH_SIZE):
                prompts = []
                for _, r in batch.iterrows():
                    msgs = mm._get_sw_prompt(r["title"], r["abstract"])
                    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

                outs = pipe(prompts)

                for r_row, o in zip(batch.itertuples(index=False), outs):
                    out_text = o[0]["generated_text"]
                    parsed_json = self._teacher_parse_with_retry(
                        mm, pipe, tok, r_row.title, r_row.abstract, out_text
                    )

                    rows.append({
                        "title": r_row.title,
                        "abstract": r_row.abstract,
                        "review": r_row.review,
                        "target_json": json.dumps(parsed_json, ensure_ascii=False),
                    })

                # salva incrementalmente (così non perdi progresso)
                pd.DataFrame(rows).to_csv(out_path, index=False)

            logging.info(f"S/W targets created for {split}: {len(rows)} samples")

    def _teacher_parse_with_retry(
        self,
        mm,
        pipe,
        tok,
        title: str,
        abstract: str,
        out_text: str
    ) -> Dict[str, Any]:
        """
        1) parse robusto
        2) retry se JSON rotto o liste vuote
        3) fallback soft se ancora vuoto
        """
        parsed = mm._extract_json(out_text)
        parsed = mm._ensure_non_empty_sw(parsed)

        if mm._is_good_sw(parsed):
            return parsed

        # retry mirato
        for k in range(CFG.TEACHER_RETRY_MAX):
            msgs = mm._get_sw_prompt_strict(title, abstract)
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            outs = pipe(
                [prompt],
                max_new_tokens=CFG.TEACHER_MAX_NEW_TOKENS + (k+1)*CFG.TEACHER_RETRY_TOKENS_BOOST,
                temperature=0.0,
                do_sample=False,
                return_full_text=False
            )
            retry_text = outs[0][0]["generated_text"]
            parsed = mm._extract_json(retry_text)
            parsed = mm._ensure_non_empty_sw(parsed)
            if mm._is_good_sw(parsed):
                return parsed

        # fallback: non vuoi mai vuoto
        return mm._fallback_sw(title, abstract)


# ---------------------------
# 2. MODEL MANAGER
# ---------------------------
class ModelManager:

    # ---- PROMPTS ----
    def _scale_hint(self) -> str:
        return (
            "Use the full 1-10 scale. "
            "1-3 = reject / major flaws, "
            "4-6 = borderline / mixed, "
            "7-8 = accept, "
            "9-10 = strong accept."
        )

    # base prompt S/W (zeroshot)
    def _get_sw_prompt(self, title: str, abstract: str) -> List[Dict]:
        sys_msg = (
            "You are an expert reviewer. Analyze the title and abstract. "
            "Return ONLY a JSON with keys: "
            "'strengths' (list of 1-3 short bullet points), "
            "'weaknesses' (list of 1-3 short bullet points). "
            "No rating, no extra text. End after the JSON."
        )
        user_msg = f"Title: {title}\nAbstract: {abstract}"
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}]

    # stricter prompt for retries
    def _get_sw_prompt_strict(self, title: str, abstract: str) -> List[Dict]:
        sys_msg = (
            "You are an expert reviewer. "
            "Return ONLY a VALID JSON object, nothing else. "
            "Keys must be exactly: strengths, weaknesses. "
            "strengths = list with at least 1 item. "
            "weaknesses = list with at least 1 item. "
            "No markdown, no commentary, no rating."
        )
        user_msg = f"Title: {title}\nAbstract: {abstract}"
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}]

    # Prompt TRAIN S/W con review in input (mix)
    def _get_sw_train_prompt(
        self, title: str, abstract: str, review: str, use_review: bool
    ) -> List[Dict]:
        sys_msg = (
            "You are an expert reviewer. "
            "Extract ONLY a VALID JSON with keys: strengths, weaknesses. "
            "Each is a list of 1-3 concise points. "
            "No rating, no extra text."
        )
        if use_review and isinstance(review, str) and review.strip():
            user_msg = (
                f"Title: {title}\n"
                f"Abstract: {abstract}\n\n"
                f"Full review:\n{review}"
            )
        else:
            user_msg = f"Title: {title}\nAbstract: {abstract}"

        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}]

    # Prompt rate-only per logits
    def _get_rate_only_prompt(self, title: str, abstract: str) -> List[Dict]:
        sys_msg = (
            "You are an expert reviewer. "
            + self._scale_hint() +
            " Return ONLY a JSON: {'rate': number between 1 and 10}. No other text."
        )
        user_msg = f"Title: {title}\nAbstract: {abstract}"
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}]

    # ---- TRAIN LORA ON SW STRUCTURE ----
    def train_lora_sw_only(self):
        """
        LoRA per imparare SOLO struttura S/W teacher-style.
        """
        logging.info("--- Starting LoRA Training on S/W JSON targets ---")

        tok = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
        tok.pad_token = tok.eos_token

        train_df = pd.read_csv(CFG.DATA_DIR / "train_sw_targets.csv")
        val_df = pd.read_csv(CFG.DATA_DIR / "val_sw_targets.csv")

        def format_fn(x):
            use_review = (random.random() < CFG.MIX_USE_REVIEW_PROB)
            msgs = self._get_sw_train_prompt(
                x["title"], x["abstract"], x.get("review", ""), use_review=use_review
            )
            msgs.append({"role": "assistant", "content": x["target_json"]})
            return {"text": tok.apply_chat_template(msgs, tokenize=False)}

        ds_train = Dataset.from_pandas(train_df).map(format_fn)
        ds_val = Dataset.from_pandas(val_df).map(format_fn)

        def tokenize_fn(ex):
            return tok(ex["text"], truncation=True, max_length=CFG.MAX_SEQ_LEN)

        ds_train = ds_train.map(tokenize_fn, remove_columns=ds_train.column_names)
        ds_val = ds_val.map(tokenize_fn, remove_columns=ds_val.column_names)

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.MODEL_ID, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        peft_cfg = LoraConfig(
            r=CFG.LORA_R,
            lora_alpha=CFG.LORA_ALPHA,
            lora_dropout=CFG.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_cfg)

        args = TrainingArguments(
            output_dir=str(CFG.MODEL_DIR / "ckpt_sw"),
            per_device_train_batch_size=CFG.BATCH_SIZE,
            gradient_accumulation_steps=CFG.GRAD_ACC_STEPS,
            num_train_epochs=CFG.EPOCHS,
            learning_rate=CFG.LR,
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            group_by_length=True,
            optim="paged_adamw_8bit",
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
        )
        trainer.train()

        model.save_pretrained(CFG.MODEL_DIR / "final_adapter_sw")
        tok.save_pretrained(CFG.MODEL_DIR / "final_adapter_sw")
        logging.info("LoRA S/W-only training done.")

    # ---- RATE VIA LOGITS ----
    def predict_rate_logits(self, model, tok, title: str, abstract: str) -> float:
        msgs = self._get_rate_only_prompt(title, abstract)
        prompt_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = tok(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]

        candidates = [str(i) for i in range(1, 11)]
        logscores = []

        with torch.no_grad():
            for c in candidates:
                cand_ids = tok.encode(c, add_special_tokens=False)
                cand_ids_t = torch.tensor([cand_ids], device=model.device)

                input_ids = torch.cat([prompt_ids, cand_ids_t], dim=1)
                out = model(input_ids=input_ids)
                logits = out.logits

                cand_logits = logits[0, prompt_len - 1: prompt_len - 1 + len(cand_ids), :]
                log_probs = torch.log_softmax(cand_logits, dim=-1)

                score = 0.0
                for j, tid in enumerate(cand_ids):
                    score += log_probs[j, tid].item()
                logscores.append(score)

        logscores = np.asarray(logscores, dtype=np.float32)
        probs = np.exp(logscores - logscores.max())
        probs = probs / probs.sum()
        rates = np.arange(1, 11, dtype=np.float32)
        return float((probs * rates).sum())

    # ---- FULL INFERENCE (VAL): S/W JSON + logits rate ----
    def run_inference_full(self, use_adapter: bool, split: str = "val"):
        mode = "ft" if use_adapter else "zeroshot"
        logging.info(f"--- Inference: {mode.upper()} on {split.upper()} ---")

        tok = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.MODEL_ID, quantization_config=bnb, device_map="auto"
        )
        if use_adapter:
            adapter_path = CFG.MODEL_DIR / "final_adapter_sw"
            if not adapter_path.exists():
                raise FileNotFoundError("S/W Adapter not found. Train first or skip ft inference.")
            model = PeftModel.from_pretrained(model, str(adapter_path))

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=CFG.MAX_NEW_TOKENS_FULL,
            temperature=CFG.INFER_TEMPERATURE,
            return_full_text=False,
            batch_size=CFG.INFER_BATCH_SIZE,
        )

        df = pd.read_csv(CFG.DATA_DIR / f"{split}_clean.csv")
        results = []

        for _, batch in batched(df.reset_index(drop=True), CFG.INFER_BATCH_SIZE):
            prompts = []
            for _, r in batch.iterrows():
                msgs = self._get_sw_prompt(r["title"], r["abstract"])
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

            outs = pipe(prompts)

            for r_row, o in zip(batch.itertuples(index=False), outs):
                out_text = o[0]["generated_text"]
                parsed_json = self._extract_json(out_text)
                parsed_json = self._ensure_non_empty_sw(parsed_json)

                rate_logits = self.predict_rate_logits(model, tok, r_row.title, r_row.abstract)

                results.append(
                    {
                        "title": r_row.title,
                        "raw_output": out_text,
                        "parsed_json": json.dumps(parsed_json, ensure_ascii=False),
                        "parsed_rate": rate_logits,
                    }
                )

        out_path = CFG.RESULTS_DIR / f"{split}_{mode}_results.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        logging.info(f"Saved {out_path}")

    # ---- RATE-ONLY INFERENCE (TRAIN) for calibration ----
    def run_inference_rate_only(self, use_adapter: bool, split: str = "train"):
        mode = "ft" if use_adapter else "zeroshot"
        logging.info(f"--- Rate-only logits: {mode.upper()} on {split.upper()} ---")

        tok = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
        tok.pad_token = tok.eos_token

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.MODEL_ID, quantization_config=bnb, device_map="auto"
        )
        if use_adapter:
            adapter_path = CFG.MODEL_DIR / "final_adapter_sw"
            if not adapter_path.exists():
                raise FileNotFoundError("Adapter not found. Train first or skip ft rate-only.")
            model = PeftModel.from_pretrained(model, str(adapter_path))

        df = pd.read_csv(CFG.DATA_DIR / f"{split}_clean.csv")
        results = []

        for _, r in tqdm(df.iterrows(), total=len(df)):
            pred_rate = self.predict_rate_logits(model, tok, r["title"], r["abstract"])
            results.append({"title": r["title"], "parsed_rate": pred_rate})

        out_path = CFG.RESULTS_DIR / f"{split}_{mode}_rateonly_results.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        logging.info(f"Saved {out_path}")

    # -----------------
    # JSON helpers
    # -----------------
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Estrae il primo oggetto JSON ben formato nel testo.
        """
        if not isinstance(text, str):
            return {}

        # prova match con bilanciamento grezzo
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}

        blob = m.group(0).strip()

        # ripulisci trailing junk dopo ultimo }
        last = blob.rfind("}")
        blob = blob[:last+1]

        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
        return {}

    @staticmethod
    def _ensure_non_empty_sw(obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assicura che strengths/weaknesses siano liste non vuote.
        """
        if not isinstance(obj, dict):
            obj = {}

        s = obj.get("strengths", [])
        w = obj.get("weaknesses", [])

        if not isinstance(s, list):
            s = []
        if not isinstance(w, list):
            w = []

        if len(s) == 0:
            s = ["Clear motivation and coherent problem setup."]
        if len(w) == 0:
            w = ["The abstract does not fully clarify limitations or edge cases."]

        obj["strengths"] = s[:3]
        obj["weaknesses"] = w[:3]
        return obj

    @staticmethod
    def _is_good_sw(obj: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        s = obj.get("strengths", [])
        w = obj.get("weaknesses", [])
        return isinstance(s, list) and isinstance(w, list) and len(s) > 0 and len(w) > 0

    @staticmethod
    def _fallback_sw(title: str, abstract: str) -> Dict[str, Any]:
        """
        Fallback rapidissimo e safe (mai vuoto).
        """
        strengths = [
            "The paper addresses a relevant problem and is clearly motivated."
        ]
        weaknesses = [
            "The abstract leaves some methodological or evaluation details unclear."
        ]
        return {"strengths": strengths, "weaknesses": weaknesses}


# ---------------------------
# 3. EVALUATOR
# ---------------------------
class Evaluator:
    def __init__(self):
        self.calibrators = {}

    def fit_calibration(self, mode: str):
        """
        Fit calibration ONLY on TRAIN rate-only predictions (no leakage).
        """
        train_gt = pd.read_csv(CFG.DATA_DIR / "train_clean.csv")
        pred_path = CFG.RESULTS_DIR / f"train_{mode}_rateonly_results.csv"

        if not pred_path.exists():
            logging.warning(f"No train rate-only predictions for {mode}. Skipping calibration.")
            return

        train_pred = pd.read_csv(pred_path)
        merged = train_gt.merge(train_pred, on="title").dropna(subset=["rate", "parsed_rate"])

        X = merged["parsed_rate"].astype(float).values.reshape(-1, 1)
        y = merged["rate"].astype(float).values

        if len(X) > 5:
            reg = HuberRegressor().fit(X, y)
            self.calibrators[mode] = reg
            logging.info(f"Calibration fitted for {mode}.")

    def evaluate(self):
        logging.info("--- Evaluation on VALIDATION ---")
        val_df = pd.read_csv(CFG.DATA_DIR / "val_clean.csv").set_index("title")
        metrics = {}

        for mode in ["zeroshot", "ft"]:
            self.fit_calibration(mode)

            path = CFG.RESULTS_DIR / f"val_{mode}_results.csv"
            if not path.exists():
                logging.warning(f"Missing {path}, skipping.")
                continue

            pred_df = pd.read_csv(path).set_index("title")
            df = val_df.join(pred_df, lsuffix="_gt").dropna(subset=["rate", "parsed_rate"])

            y_true = df["rate"].astype(float).values
            y_raw = df["parsed_rate"].astype(float).values

            metrics[f"{mode}.raw"] = self._compute(y_true, y_raw)

            if mode in self.calibrators:
                y_cal = self.calibrators[mode].predict(y_raw.reshape(-1, 1))
                y_cal = np.clip(y_cal, 1.0, 10.0)
                metrics[f"{mode}.calibrated"] = self._compute(y_true, y_cal)

        print(json.dumps(metrics, indent=2))
        with open(CFG.RESULTS_DIR / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    @staticmethod
    def _compute(yt, yp):
        return {
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "Pearson": float(pearsonr(yt, yp)[0]) if len(yt) > 1 else 0.0,
        }


# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        default="all",
        choices=["prepare", "teacher", "train", "infer", "eval", "all"],
        help="""
prepare = clean + split
teacher = build S/W targets by zeroshot teacher (FAST + resume)
train = LoRA on S/W only
infer = full inference (val) + rate-only (train)
eval = compute metrics
all = everything
"""
    )
    parser.add_argument("--file", default="data/tp_2020conference.xlsx")
    parser.add_argument("--skip_full_infer", action="store_true")
    parser.add_argument("--skip_rate_only", action="store_true")

    args = parser.parse_args()

    setup_logging()
    set_seed()

    dp = DataProcessor()
    mm = ModelManager()
    ev = Evaluator()

    if args.step in ["prepare", "all"]:
        dp.prepare_data(args.file)

    if args.step in ["teacher", "all"]:
        dp.create_sw_targets_from_teacher()

    if args.step in ["train", "all"]:
        mm.train_lora_sw_only()

    if args.step in ["infer", "all"]:
        if not args.skip_full_infer:
            mm.run_inference_full(use_adapter=False, split="val")
            mm.run_inference_full(use_adapter=True, split="val")
        else:
            logging.info("Skipping full inference.")

        if not args.skip_rate_only:
            mm.run_inference_rate_only(use_adapter=False, split="train")
            mm.run_inference_rate_only(use_adapter=True, split="train")
        else:
            logging.info("Skipping rate-only inference.")

    if args.step in ["eval", "all"]:
        ev.evaluate()


if __name__ == "__main__":
    main()
