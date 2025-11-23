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

# ML Imports
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# HF Imports
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


# --- CONFIGURATION ---
class Config:
    SEED = 42
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_SEQ_LEN = 512

    ROOT = pathlib.Path(".").resolve()
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results"
    MODEL_DIR = ROOT / "model"

    # LoRA Params
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

    # Training Params (fixed, no early stopping)
    LR = 2e-4
    EPOCHS = 1
    BATCH_SIZE = 2
    GRAD_ACC_STEPS = 4

    # Inference Params
    MAX_NEW_TOKENS_FULL = 300     # full JSON (val only)
    MAX_NEW_TOKENS_RATE = 60      # rate-only (train for calibration)
    INFER_BATCH_SIZE = 8          # batching during inference

    def __init__(self):
        for d in (self.DATA_DIR, self.RESULTS_DIR, self.MODEL_DIR):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()


# --- UTILS ---
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


# --- 1. DATA PROCESSOR ---
class DataProcessor:
    """Handles parsing of Excel/CSV and creating training targets."""

    @staticmethod
    def clean_text(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"https?://\S+", "", text)  # remove URLs
        text = text.replace("Abstract:", "").strip()
        text = text.replace("Review:", "").replace("###", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def parse_rate(val: Any) -> Optional[float]:
        """Robust parsing of numeric scores from mixed Excel formats."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None

        s = str(val).lower().strip().replace(",", ".")

        # 1) Fraction "8/10"
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", s)
        if match:
            num, den = float(match.group(1)), float(match.group(2))
            if den == 100:
                return num / 10.0
            if den == 10:
                return num
            if den > 0:
                return (num / den) * 10.0

        # 2) Simple number inside text
        try:
            floats = re.findall(r"\d+\.?\d*", s)
            valid_nums = [float(f) for f in floats if 1.0 <= float(f) <= 10.0]
            if valid_nums:
                return valid_nums[-1]
        except ValueError:
            pass

        # 3) Text Labels
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

    def _extract_sw_from_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract Strengths/Weaknesses from unstructured review text.
        Handles OpenReview-ish markdown, headers, bullets, and fallback cues.
        """
        text = str(text)

        # 0) OpenReview/Markdown cleanup
        text = re.sub(r"^review\s*:\s*", "", text, flags=re.I)
        text = text.replace("###", "\n")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # 1) normalize common headers
        header_map = {
            r"\bstrong points?\b": "strengths",
            r"\bpositives?\b": "strengths",
            r"\bpros\b": "strengths",
            r"\bweak points?\b": "weaknesses",
            r"\bnegatives?\b": "weaknesses",
            r"\bcons\b": "weaknesses",
            r"\blimitations?\b": "weaknesses",
        }
        for pat, rep in header_map.items():
            text = re.sub(pat, rep, text, flags=re.I)

        # 2) find header blocks
        p_str = r"strengths\s*[:\-]?\s*(.*?)(?=\n\s*weaknesses\s*[:\-]|\Z)"
        p_weak = r"weaknesses\s*[:\-]?\s*(.*?)(?=\n\s*strengths\s*[:\-]|\Z)"

        m_str = re.search(p_str, text, flags=re.I | re.S)
        m_weak = re.search(p_weak, text, flags=re.I | re.S)

        def split_items(block: str) -> List[str]:
            block = block.strip()
            items = re.split(r"(?:\n\s*[\-\*•]\s*|\n\s*\d+[.)]\s*)", block)
            items = [re.sub(r"\s+", " ", i).strip() for i in items]
            items = [i for i in items if len(i) > 15]
            return items[:3]

        strengths = split_items(m_str.group(1)) if m_str else []
        weaknesses = split_items(m_weak.group(1)) if m_weak else []

        # 3) fallback cues if no headers found
        if not strengths and not weaknesses:
            sents = re.split(r"(?<=[.!?])\s+", text)
            pos_cues = ("novel", "interesting", "strong", "clear", "well-written", "effective")
            neg_cues = ("weak", "unclear", "lacking", "missing", "problem", "limitation", "concern")

            for sent in sents:
                low = sent.lower()
                if any(c in low for c in pos_cues) and len(strengths) < 3:
                    strengths.append(sent.strip()[:250])
                elif any(c in low for c in neg_cues) and len(weaknesses) < 3:
                    weaknesses.append(sent.strip()[:250])

            if not strengths:
                strengths = [sents[0].strip()[:250]] if sents else []
            weaknesses = weaknesses[:3]

        return strengths, weaknesses

    def prepare_data(self, file_path: str):
        logging.info(f"Loading data from {file_path}...")
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)  # needs openpyxl

        df.columns = [c.strip().lower() for c in df.columns]

        rate_cols = [c for c in df.columns if any(x in c for x in ["rate", "score", "overall"])]
        logging.info(f"Rating columns found: {rate_cols}")

        processed = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
            rating = None
            for c in rate_cols:
                parsed = self.parse_rate(row[c])
                if parsed is not None:
                    rating = parsed
                    break

            rev_text = self.clean_text(row.get("review", ""))

            processed.append(
                {
                    "title": self.clean_text(row.get("title", "")),
                    "abstract": self.clean_text(row.get("abstract", "")),
                    "review": rev_text,
                    "rate": rating,
                }
            )

        df_clean = pd.DataFrame(processed).dropna(subset=["title", "abstract"])

        from sklearn.model_selection import train_test_split

        train, val = train_test_split(df_clean, test_size=0.2, random_state=CFG.SEED)

        train.to_csv(CFG.DATA_DIR / "train_clean.csv", index=False)
        val.to_csv(CFG.DATA_DIR / "val_clean.csv", index=False)
        logging.info(f"Split saved. Train: {len(train)}, Val: {len(val)}")

    def create_supervised_json(self):
        logging.info("Generating supervised JSON targets...")
        for split in ["train", "val"]:
            path = CFG.DATA_DIR / f"{split}_clean.csv"
            df = pd.read_csv(path)
            rows = []

            for _, r in df.iterrows():
                if pd.isna(r["rate"]) or pd.isna(r["review"]) or str(r["review"]).strip() == "":
                    continue

                s, w = self._extract_sw_from_text(r["review"])

                target = {"strengths": s, "weaknesses": w, "rate": float(r["rate"])}

                rows.append(
                    {
                        "title": r["title"],
                        "abstract": r["abstract"],
                        "target_json": json.dumps(target, ensure_ascii=False),
                    }
                )

            out_path = CFG.DATA_DIR / f"{split}_supervised.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)
            logging.info(f"Targets created for {split}: {len(rows)} samples")


# --- 2. MODEL MANAGER ---
class ModelManager:
    def __init__(self):
        self.tokenizer = None

    # full task prompt (S/W + rate)
    def _get_chat_prompt(self, title: str, abstract: str) -> List[Dict]:
        sys_msg = (
            "You are an expert reviewer. Analyze the title and abstract. "
            "Return ONLY a JSON with keys: 'strengths' (list), 'weaknesses' (list), "
            "'rate' (number 1-10). No intro text."
        )
        user_msg = f"Title: {title}\nAbstract: {abstract}"
        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

    # cheap prompt (only rate) for calibration
    def _get_rate_only_prompt(self, title: str, abstract: str) -> List[Dict]:
        sys_msg = (
            "You are an expert reviewer. Given title and abstract, "
            "return ONLY a JSON: {'rate': number between 1 and 10}. "
            "No other keys, no text."
        )
        user_msg = f"Title: {title}\nAbstract: {abstract}"
        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

    def train_lora(self):
        logging.info("--- Starting LoRA Training (fixed 1 epoch) ---")
        tok = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
        tok.pad_token = tok.eos_token

        train_df = pd.read_csv(CFG.DATA_DIR / "train_supervised.csv")
        val_df = pd.read_csv(CFG.DATA_DIR / "val_supervised.csv")

        def format_fn(x):
            msgs = self._get_chat_prompt(x["title"], x["abstract"])
            msgs.append({"role": "assistant", "content": x["target_json"]})
            return {"text": tok.apply_chat_template(msgs, tokenize=False)}

        ds_train = Dataset.from_pandas(train_df).map(format_fn)
        ds_val = Dataset.from_pandas(val_df).map(format_fn)

        def tokenize_fn(ex):
            out = tok(
                ex["text"],
                truncation=True,
                max_length=CFG.MAX_SEQ_LEN,
                # padding dinamico via collator
            )
            # Labels are created by the data collator from padded input_ids
            return out

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
            output_dir=str(CFG.MODEL_DIR / "ckpt"),
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

        model.save_pretrained(CFG.MODEL_DIR / "final_adapter")
        tok.save_pretrained(CFG.MODEL_DIR / "final_adapter")
        logging.info("Training done.")

    # -------- FULL INFERENCE (JSON) --------
    def run_inference_full(self, use_adapter: bool, split: str = "val"):
        """
        Full inference returning strengths/weaknesses/rate.
        Use ONLY on validation for qualitative + final eval.
        """
        mode = "ft" if use_adapter else "zeroshot"
        logging.info(f"--- Full Inference: {mode.upper()} on {split.upper()} ---")

        tok = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # decoder-only models expect left padding at generation time

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.MODEL_ID, quantization_config=bnb, device_map="auto"
        )
        if use_adapter:
            model = PeftModel.from_pretrained(model, str(CFG.MODEL_DIR / "final_adapter"))

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=CFG.MAX_NEW_TOKENS_FULL,
            temperature=0.1,
            return_full_text=False,
            batch_size=CFG.INFER_BATCH_SIZE,
        )

        df = pd.read_csv(CFG.DATA_DIR / f"{split}_clean.csv")
        results = []

        for i in tqdm(range(0, len(df), CFG.INFER_BATCH_SIZE), total=(len(df)//CFG.INFER_BATCH_SIZE + 1)):
            batch = df.iloc[i:i+CFG.INFER_BATCH_SIZE]
            prompts = []
            for _, r in batch.iterrows():
                msgs = self._get_chat_prompt(r["title"], r["abstract"])
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

            outs = pipe(prompts)
            for r_row, o in zip(batch.itertuples(index=False), outs):
                out_text = o[0]["generated_text"]
                parsed = self._extract_json(out_text)
                results.append(
                    {
                        "title": r_row.title,
                        "raw_output": out_text,
                        "parsed_rate": parsed.get("rate"),
                        "parsed_json": json.dumps(parsed),
                    }
                )

        pd.DataFrame(results).to_csv(
            CFG.RESULTS_DIR / f"{split}_{mode}_results.csv", index=False
        )

    # -------- RATE-ONLY INFERENCE (FOR CALIBRATION) --------
    def run_inference_rate_only(self, use_adapter: bool, split: str = "train"):
        """
        Cheap inference returning ONLY rate.
        Use on TRAIN for calibration (fast).
        """
        mode = "ft" if use_adapter else "zeroshot"
        logging.info(f"--- Rate-only Inference: {mode.upper()} on {split.upper()} ---")

        tok = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.MODEL_ID, quantization_config=bnb, device_map="auto"
        )
        if use_adapter:
            model = PeftModel.from_pretrained(model, str(CFG.MODEL_DIR / "final_adapter"))

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=CFG.MAX_NEW_TOKENS_RATE,
            do_sample=False,  # greedy decoding (avoids temperature==0 errors)
            return_full_text=False,
            batch_size=CFG.INFER_BATCH_SIZE,
        )

        df = pd.read_csv(CFG.DATA_DIR / f"{split}_clean.csv")
        results = []

        for i in tqdm(range(0, len(df), CFG.INFER_BATCH_SIZE), total=(len(df)//CFG.INFER_BATCH_SIZE + 1)):
            batch = df.iloc[i:i+CFG.INFER_BATCH_SIZE]
            prompts = []
            for _, r in batch.iterrows():
                msgs = self._get_rate_only_prompt(r["title"], r["abstract"])
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

            outs = pipe(prompts)
            for r_row, o in zip(batch.itertuples(index=False), outs):
                out_text = o[0]["generated_text"]
                parsed = self._extract_json(out_text)
                results.append(
                    {
                        "title": r_row.title,
                        "parsed_rate": parsed.get("rate"),
                        "raw_output": out_text,
                    }
                )

        pd.DataFrame(results).to_csv(
            CFG.RESULTS_DIR / f"{split}_{mode}_rateonly_results.csv", index=False
        )

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                return {}
        return {}


# --- 3. EVALUATOR ---
class Evaluator:
    def __init__(self):
        self.calibrators = {}

    def fit_calibration(self, mode: str):
        """
        Fits calibration on TRAIN predictions only (no leakage).
        Uses rate-only predictions for speed.
        """
        train_gt = pd.read_csv(CFG.DATA_DIR / "train_clean.csv")
        train_pred_path = CFG.RESULTS_DIR / f"train_{mode}_rateonly_results.csv"

        if not train_pred_path.exists():
            logging.warning(f"No train rate-only predictions for {mode}. Skipping calibration.")
            return

        train_pred = pd.read_csv(train_pred_path)
        merged = train_gt.merge(train_pred, on="title").dropna(subset=["rate", "parsed_rate"])

        X = merged["parsed_rate"].astype(float).values.reshape(-1, 1)
        y = merged["rate"].astype(float).values

        if len(X) > 5:
            reg = HuberRegressor().fit(X, y)
            self.calibrators[mode] = reg
            logging.info(f"Calibration fitted for {mode}.")

    def evaluate(self):
        """
        Evaluation on VAL using full predictions.
        If calibration exists, also reports calibrated metrics.
        """
        logging.info("--- Evaluation on VALIDATION ---")
        val_df = pd.read_csv(CFG.DATA_DIR / "val_clean.csv").set_index("title")
        metrics = {}

        for mode in ["zeroshot", "ft"]:
            self.fit_calibration(mode)

            path = CFG.RESULTS_DIR / f"val_{mode}_results.csv"
            if not path.exists():
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


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        default="all",
        choices=["prepare", "train", "infer", "eval", "all"],
    )
    parser.add_argument(
        "--skip_full_infer",
        action="store_true",
        help="Skip full val inference when --step infer/all.",
    )
    parser.add_argument(
        "--skip_rate_only",
        action="store_true",
        help="Skip rate-only train inference when --step infer/all.",
    )
    parser.add_argument("--file", default="data/tp_2020conference.xlsx")
    args = parser.parse_args()

    setup_logging()
    set_seed()

    dp = DataProcessor()
    mm = ModelManager()
    ev = Evaluator()

    if args.step in ["prepare", "all"]:
        dp.prepare_data(args.file)
        dp.create_supervised_json()

    if args.step in ["train", "all"]:
        mm.train_lora()

    if args.step in ["infer", "all"]:
        # FULL inference ONLY on VAL
        if args.skip_full_infer:
            logging.info("Skipping full val inference (--skip_full_infer).")
        else:
            mm.run_inference_full(use_adapter=False, split="val")
            mm.run_inference_full(use_adapter=True, split="val")

        # RATE-ONLY inference on TRAIN for calibration (fast)
        if args.skip_rate_only:
            logging.info("Skipping rate-only train inference (--skip_rate_only).")
        else:
            mm.run_inference_rate_only(use_adapter=False, split="train")
            mm.run_inference_rate_only(use_adapter=True, split="train")

    if args.step in ["eval", "all"]:
        ev.evaluate()


if __name__ == "__main__":
    main()
