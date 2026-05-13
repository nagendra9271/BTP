"""
Qwen2.5-VL Evaluation Script for MOCHEG Synthetic Misinformer Dataset
======================================================================
3-class evaluation: Supported / Refuted / Not Enough Information
  --no_nei flag: drops NEI rows, uses 2-class (Supported / Refuted) only

KEY FIX over previous version:
  • Uses Qwen2_5_VLForConditionalGeneration (not Qwen2VLForConditionalGeneration)
    This eliminates the architecture mismatch and the
    "Image features and image tokens do not match" ValueError.
  • Uses the correct processor: AutoProcessor from transformers >= 4.49
  • target_label is read from the 'new_label' column in your CSV
  • Removed invalid ignore_mismatched_sizes and temperature generation flag

Requirements:
    pip install -U transformers>=4.49.0 accelerate pillow pandas scikit-learn
    pip install qwen-vl-utils
    pip install bitsandbytes   # optional, for 4/8-bit quantisation

Usage:
    # Vision mode (recommended):
    python evaluate_qwen.py \\
        --csv        mocheg_synthetic_12k.csv \\
        --images_dir train/images \\
        --mode       vision \\
        --model      Qwen/Qwen2.5-VL-7B-Instruct \\
        --output     results_vision.csv

    # Text-only:
    python evaluate_qwen.py \\
        --csv        mocheg_synthetic_12k.csv \\
        --images_dir train/images \\
        --mode       text \\
        --model      Qwen/Qwen2.5-7B-Instruct \\
        --output     results_text.csv

    # Quick 50-row smoke test with 4-bit quantisation:
    python evaluate_qwen.py \\
        --csv mocheg_synthetic_12k.csv --images_dir train/images \\
        --mode vision --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --output results_50.csv --max_samples 50 --quantize 4bit

    # Skip NEI claims entirely (2-class: Supported / Refuted):
    python evaluate_qwen.py \\
        --csv mocheg_synthetic_12k.csv --images_dir train/images \\
        --mode vision --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --output results_no_nei.csv --no_nei
"""

import argparse
import json
import os
import sys
import time
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            required=True)
    p.add_argument("--images_dir",     required=True)
    p.add_argument("--mode",           choices=["text", "vision"], default="vision")
    p.add_argument("--model",          default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output",         default="qwen_eval_results.csv")
    p.add_argument("--max_samples",    type=int, default=0,
                   help="0 = use all rows")
    p.add_argument("--quantize",       choices=["4bit", "8bit", "none"],
                   default="none")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--device",         default="auto")
    p.add_argument("--no_evidence",    action="store_true",
                   help="Use only claim (no text evidence)")
    p.add_argument("--no_nei",         action="store_true",
                   help="Drop NEI rows; run 2-class (Supported / Refuted) only")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def _quant_config(quantize):
    if quantize == "none":
        return None
    from transformers import BitsAndBytesConfig
    import torch
    if quantize == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_compute_dtype=torch.float16)
    return BitsAndBytesConfig(load_in_8bit=True)


def load_text_model(model_id, quantize, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[Model] Loading text model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_quant_config(quantize),
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("[Model] Text model ready.\n")
    return tokenizer, model


def load_vision_model(model_id, quantize, device):
    from transformers import AutoProcessor

    print(f"\n[Model] Loading vision model: {model_id}")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        ModelClass = Qwen2_5_VLForConditionalGeneration
        print("[Model] Class: Qwen2_5_VLForConditionalGeneration  ✓")
    except ImportError:
        print("[Model] WARNING: Qwen2_5_VLForConditionalGeneration not found.")
        print("        Your transformers version is too old. Run:")
        print("          pip install -U 'transformers>=4.49.0'")
        print("        Falling back to AutoModelForVision2Seq (may still fail).")
        from transformers import AutoModelForVision2Seq
        ModelClass = AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = ModelClass.from_pretrained(
        model_id,
        quantization_config=_quant_config(quantize),
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("[Model] Vision model ready.\n")
    return processor, model


# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_2CLASS = (
    "You are a fact-checking assistant. "
    "Given a claim and supporting evidence (and optionally an image), "
    "classify the claim as one of:\n"
    "  Supported\n"
    "  Refuted\n\n"
    "Respond with EXACTLY one of those two labels. No explanation."
)

SYSTEM_PROMPT_3CLASS = (
    "You are a fact-checking assistant. "
    "Given a claim and supporting evidence (and optionally an image), "
    "classify the claim as one of:\n"
    "  Supported\n"
    "  Refuted\n"
    "  Not Enough Information\n\n"
    "Respond with EXACTLY one of those three labels. No explanation."
)


def get_system_prompt(no_nei: bool) -> str:
    return SYSTEM_PROMPT_2CLASS if no_nei else SYSTEM_PROMPT_3CLASS


def get_classify_question(no_nei: bool) -> str:
    if no_nei:
        return "Classify: Supported / Refuted?"
    return "Classify: Supported / Refuted / Not Enough Information?"


def build_text_prompt(claim, evidence, no_nei=False):
    ev = evidence[:800] if evidence else "(none)"
    return (
        f"Claim: {claim}\n\n"
        f"Evidence: {ev}\n\n"
        f"{get_classify_question(no_nei)}"
    )


def build_vision_messages(claim, evidence, image_path, no_evidence=False, no_nei=False):
    question = get_classify_question(no_nei)

    if no_evidence:
        text_part = f"Claim: {claim}\n\n{question}"
    else:
        ev = evidence[:800] if evidence else "(none)"
        text_part = f"Claim: {claim}\n\nEvidence: {ev}\n\n{question}"

    content = []
    if image_path and os.path.isfile(image_path):
        content.append({
            "type": "image",
            "image": f"file://{os.path.abspath(image_path)}"
        })
    content.append({"type": "text", "text": text_part})

    return [
        {"role": "system", "content": get_system_prompt(no_nei)},
        {"role": "user",   "content": content},
    ]


# ──────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────

def parse_response(text: str, no_nei: bool = False) -> str:
    """
    Map Qwen free-text output to a label string.
    no_nei=True  → 'supported' | 'refuted' | 'unknown'
    no_nei=False → 'supported' | 'refuted' | 'nei' | 'unknown'
    """
    first = next(
        (ln.strip() for ln in text.split("\n") if ln.strip()), ""
    ).lower()

    if first.startswith("supported"):
        return "supported"
    if first.startswith("refuted"):
        return "refuted"
    if not no_nei:
        if first.startswith("not enough") or "not enough" in first:
            return "nei"

    # substring fallback
    if "refuted" in first:
        return "refuted"
    if "supported" in first:
        return "supported"
    if not no_nei and "not enough" in first:
        return "nei"

    return "unknown"


# ──────────────────────────────────────────────────────────────
# Inference — text
# ──────────────────────────────────────────────────────────────

def infer_text(rows, tokenizer, model, max_new_tokens, no_nei=False):
    import torch
    preds, raws = [], []
    for row in tqdm(rows, desc="Text Inference"):
        messages = [
            {"role": "system", "content": get_system_prompt(no_nei)},
            {"role": "user",   "content": build_text_prompt(
                row["claim"], row["evidence"], no_nei=no_nei)},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            text = (f"System: {get_system_prompt(no_nei)}\n"
                    f"User: {build_text_prompt(row['claim'], row['evidence'], no_nei)}\n"
                    "Assistant:")

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        preds.append(parse_response(response, no_nei=no_nei))
        raws.append(response.strip())
    return preds, raws


# ──────────────────────────────────────────────────────────────
# Inference — vision
# ──────────────────────────────────────────────────────────────

def infer_vision(rows, processor, model, max_new_tokens, no_evidence=False, no_nei=False):
    import torch
    from qwen_vl_utils import process_vision_info

    preds, raws = [], []
    for row in tqdm(rows, desc="Vision Inference"):
        messages = build_vision_messages(
            row["claim"], row["evidence"], row["image_path"],
            no_evidence=no_evidence, no_nei=no_nei
        )

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_ids = [o[len(inp):] for o, inp in zip(out, inputs["input_ids"])]
        response = processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        preds.append(parse_response(response, no_nei=no_nei))
        raws.append(response.strip())

    return preds, raws


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def get_label_order(no_nei: bool):
    if no_nei:
        return ["supported", "refuted"]
    return ["supported", "refuted", "nei"]


def evaluate(df: pd.DataFrame, no_nei: bool = False):
    LABEL_ORDER = get_label_order(no_nei)

    unknown_count = int((df["prediction"] == "unknown").sum())
    df_v = df[df["prediction"] != "unknown"]

    print("\n" + "=" * 64)
    print("  OVERALL EVALUATION" + (" [2-class: no NEI]" if no_nei else " [3-class]"))
    print("=" * 64)
    print(f"  Total samples        : {len(df)}")
    print(f"  Unknown predictions  : {unknown_count} "
          f"({100 * unknown_count / len(df):.1f}%)")
    print(f"  Valid predictions    : {len(df_v)}")

    if len(df_v) > 0:
        acc = accuracy_score(df_v["target_label"], df_v["prediction"])
        print(f"  Accuracy (valid)     : {acc * 100:.2f}%")
        print(f"\n  Classification Report ({'2' if no_nei else '3'}-class):")
        print(classification_report(
            df_v["target_label"], df_v["prediction"],
            labels=LABEL_ORDER, zero_division=0))

        cm = confusion_matrix(
            df_v["target_label"], df_v["prediction"], labels=LABEL_ORDER)
        print("  Confusion Matrix  (rows=True label, cols=Predicted):")
        header = f"{'':>14}" + "".join(f"{l:>14}" for l in LABEL_ORDER)
        print(header)
        for i, row_label in enumerate(LABEL_ORDER):
            row_str = f"{row_label:>14}" + "".join(
                f"{cm[i][j]:>14}" for j in range(len(LABEL_ORDER)))
            print(row_str)

    # ── Per misinformation technique ──────────────────────────
    print("\n" + "-" * 64)
    print("  PER-TECHNIQUE BREAKDOWN")
    print(f"  {'Technique':<32} {'N':>5}  {'Acc':>7}  Hit rates per class")
    print("-" * 64)

    for tech in sorted(df["misinformation_label"].unique()):
        sub   = df[df["misinformation_label"] == tech]
        sub_v = sub[sub["prediction"] != "unknown"]
        n     = len(sub)

        if len(sub_v) == 0:
            print(f"  {tech:<32} {n:>5}  {'N/A':>7}  (all unknown)")
            continue

        acc = accuracy_score(sub_v["target_label"], sub_v["prediction"])

        hit_parts = []
        for lbl in LABEL_ORDER:
            tot = (sub_v["target_label"] == lbl).sum()
            if tot > 0:
                tp = ((sub_v["target_label"] == lbl) &
                      (sub_v["prediction"]   == lbl)).sum()
                hit_parts.append(f"{lbl}={tp/tot*100:.0f}%")

        notes = "  [" + ", ".join(hit_parts) + "]" if hit_parts else ""
        print(f"  {tech:<32} {n:>5}  {acc*100:>6.1f}%{notes}")

    print("=" * 64)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load CSV ──────────────────────────────────────────────
    print(f"\n[Data] Loading {args.csv} ...")
    df = pd.read_csv(args.csv, dtype=str).fillna("")
    print(f"       {len(df)} rows loaded.")
    print(f"       Columns: {list(df.columns)}")

    required = [
        "claim_id", "original_claim", "image_path",
        "grouped_text_evidence", "misinformation_label",
        "generated_misleading_claim", "new_label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns in CSV: {missing}")
        print("       Available columns:", list(df.columns))
        sys.exit(1)

    # ── Normalise target_label early (needed for NEI filter) ──
    df["target_label"] = (
        df["new_label"].str.strip().str.lower()
        .replace({
            "not enough information": "nei",
            "not enough info":        "nei",
            "not_enough_information": "nei",
        })
    )

    # ── Drop NEI rows if --no_nei ──────────────────────────────
    if args.no_nei:
        before = len(df)
        df = df[df["target_label"] != "nei"].reset_index(drop=True)
        print(f"[Data] --no_nei: dropped {before - len(df)} NEI rows "
              f"→ {len(df)} rows remaining.")

    if args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)),
                       random_state=42).reset_index(drop=True)
        print(f"       Subsampled to {len(df)} rows.")

    # ── Build inference rows ──────────────────────────────────
    rows_for_inference = []
    for _, r in df.iterrows():
        label = r["misinformation_label"]
        claim_text = (
            r["original_claim"]
            if label == "Truthful"
            else (r["generated_misleading_claim"] or r["original_claim"])
        )
        img_path = (
            os.path.join(args.images_dir, r["image_path"])
            if r["image_path"] else ""
        )
        rows_for_inference.append({
            "claim":      claim_text,
            "evidence":   r["grouped_text_evidence"],
            "image_path": img_path,
        })

    # ── Load model & infer ────────────────────────────────────
    t0 = time.time()
    if args.mode == "text":
        tokenizer, model = load_text_model(args.model, args.quantize, args.device)
        print(f"[Eval] TEXT inference on {len(rows_for_inference)} samples ...")
        preds, raws = infer_text(
            rows_for_inference, tokenizer, model,
            args.max_new_tokens, no_nei=args.no_nei)
    else:
        processor, model = load_vision_model(args.model, args.quantize, args.device)
        print(f"[Eval] VISION inference on {len(rows_for_inference)} samples ...")
        preds, raws = infer_vision(
            rows_for_inference, processor, model,
            args.max_new_tokens,
            no_evidence=args.no_evidence,
            no_nei=args.no_nei)

    elapsed = time.time() - t0
    print(f"\n[Eval] Done — {elapsed:.1f}s  "
          f"({elapsed / len(rows_for_inference):.2f}s/sample)")

    # ── Attach predictions ────────────────────────────────────
    df["prediction"]   = preds
    df["raw_response"] = raws

    # ── Evaluate ──────────────────────────────────────────────
    evaluate(df, no_nei=args.no_nei)

    # ── Save CSV ──────────────────────────────────────────────
    out_cols = [c for c in [
        "claim_id", "misinformation_label", "target_label",
        "generated_misleading_claim", "image_path",
        "prediction", "raw_response",
        "nei_swapped_entity_type", "nei_swapped_from", "nei_swapped_to",
        "ooc_donor_claim_id",
    ] if c in df.columns]
    df[out_cols].to_csv(args.output, index=False)
    print(f"\n[Output] Results  → {args.output}")

    # ── Save summary JSON ─────────────────────────────────────
    df_v = df[df["prediction"] != "unknown"]
    overall_acc = (
        float(accuracy_score(df_v["target_label"], df_v["prediction"]))
        if len(df_v) > 0 else 0.0
    )
    summary = {
        "model":                 args.model,
        "mode":                  args.mode,
        "no_nei":                args.no_nei,
        "total_samples":         len(df),
        "unknown_predictions":   int((df["prediction"] == "unknown").sum()),
        "accuracy_overall":      round(overall_acc * 100, 2),
        "per_label_counts":      df["misinformation_label"].value_counts().to_dict(),
        "target_label_dist":     df["target_label"].value_counts().to_dict(),
        "prediction_dist":       df["prediction"].value_counts().to_dict(),
    }
    summary_path = args.output.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Output] Summary  → {summary_path}")


if __name__ == "__main__":
    main()