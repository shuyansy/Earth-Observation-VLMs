#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sa2va_batch_vqa.py
------------------
Batch VQA inference with Sa2VA on remote sensing images,
writing predictions to CSV.

Differences from InternVL script:
- Uses Sa2VA predict_forward_with_grounding()
- Adds our custom_prompt BEFORE <image>
- Formats options from a list into:
    A. ...
    B. ...
    C. ...
    D. ...
- Extracts <answer>...</answer> raw content
  (no A/B/C/D mapping).
- (NEW) Carries question_type into output CSV.

Usage:
python sa2va_batch_vqa.py \
    --model-path terrascope_new \
    --input-csv data/landsatvqa/ground_truth_files/Landsat30-AU-VQA-test.csv \
    --image-root data/landsatvqa/VQA \
    --out-csv sa2va_results.csv
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------------------------------------
# -------------------------------------------------------
def extract_raw_answer(response: str) -> str:
    """
     <answer> ... </answer> 
    """
    if not response:
        return ""
    m = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    return response.strip()


# -------------------------------------------------------
# -------------------------------------------------------
def format_options_list(options_raw):
    """
    options_raw:  CSV 
        - Python list  (['Forest edges','Wetland areas',...])
        -  "[...]" 

    :
        "A. Forest edges\nB. Wetland areas\nC. ...\n..."
    """
    if isinstance(options_raw, str):
        options_raw = options_raw.strip()
        if options_raw.startswith("[") and options_raw.endswith("]"):
            try:
                options_list = json.loads(options_raw)
            except Exception:
                options_list = [opt.strip() for opt in options_raw.strip("[]").split(",")]
        else:
            split_by_line = re.split(r'[\n;]+', options_raw)
            options_list = [o.strip() for o in split_by_line if o.strip()]
    elif isinstance(options_raw, (list, tuple)):
        options_list = list(options_raw)
    else:
        options_list = [str(options_raw)]

    letter_seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for idx, opt in enumerate(options_list):
        letter = letter_seq[idx] if idx < len(letter_seq) else f"Option{idx+1}"
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


# -------------------------------------------------------
#    custom_prompt + "<image>\n" + question + "\n" + formatted_options + "\n"
# -------------------------------------------------------
def build_sa2va_query(question_text: str, formatted_options: str, custom_prompt: str) -> str:
    return (
        custom_prompt
        + "<image>\n"
        + question_text.strip()
        + "\n"
        + formatted_options.strip()
        + "\n"
    )


# -------------------------------------------------------
# -------------------------------------------------------
def run_sa2va_vqa(
    model,
    tokenizer,
    df: pd.DataFrame,
    image_root: Path,
    out_csv: Path,
    segmentation_json: Path = None,
    resume: bool = True,
    max_tokens_per_seg: int = 8,
):
    """
    model, tokenizer: Sa2VA 
    df:               DataFrame
                      qa_id, image_path, question, question_type, options(), answer(gt)
    image_root:        (image_root / image_path)
    out_csv:          CSV
    segmentation_json:() segmentation_rle.jsongt_mask_rle
    resume:           
    """

    mask_dict = {}
    if segmentation_json is not None and Path(segmentation_json).exists():
        try:
            with open(segmentation_json, "r", encoding="utf-8") as f:
                mask_data = json.load(f).get("tasks", [])
            for item in mask_data:
                mask_dict[item["image"]] = item["segmentation"]
            print(f"[INFO] Loaded mask dict with {len(mask_dict)} entries from {segmentation_json}")
        except Exception as e:
            print(f"[WARN] Failed to load segmentation JSON ({segmentation_json}): {e}")
            mask_dict = {}

    done_ids = set()
    if resume and out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if r.get("qa_id"):
                    done_ids.add(r["qa_id"])
    if done_ids:
        df = df[~df["qa_id"].isin(done_ids)].copy()

    if df.empty:
        print("[INFO] Nothing new to process (all done).")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    out_fh = open(out_csv, "a", newline="", encoding="utf-8")

    fieldnames = [
        "qa_id",
        "image_path",
        "question",
        "question_type",
        "formatted_options",
        "sa2va_answer",
        "gt_answer",
    ]
    writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    custom_prompt = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in their mind and then provides the user a concise final answer in a "
        "short word or phrase. The reasoning process and answer are enclosed within "
        "<think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>\n\n"
    )

    bar = tqdm(total=len(df), unit="sample", desc="Sa2VA VQA")
    try:
        for _, row in df.iterrows():
            qa_id = str(row["qa_id"])

            img_rel_path = str(row["image_path"]).strip()
            question_text = str(row["question"]).strip()

            question_type = str(row.get("question_type", "")).strip()

            formatted_options = format_options_list(row["options"])

            gt_answer = str(row["answer"]).strip() if "answer" in row else ""

            img_path = image_root / img_rel_path
            if not img_path.exists():
                print(f"[WARN] missing image: {img_path}")
                bar.update()
                continue

            try:
                img = Image.open(str(img_path)).convert("RGB")
            except Exception as e:
                print(f"[ERROR] cannot open image {img_path}: {e}")
                bar.update()
                continue

            query_text = build_sa2va_query(
                question_text=question_text,
                formatted_options=formatted_options,
                custom_prompt=custom_prompt,
            )

            try:
                with torch.inference_mode():
                    result = model.predict_forward_with_grounding(
                        image=img,
                        text=query_text,
                        tokenizer=tokenizer,
                        max_tokens_per_seg=max_tokens_per_seg,
                        # gt_mask_rle=mask_dict.get(img_rel_path),
                    )
                raw_response = result.get("prediction", "")
            except Exception as e:
                print(f"[ERROR] Generation failed for {qa_id}: {e}")
                raw_response = ""

            pred_answer = extract_raw_answer(raw_response)

            writer.writerow(
                {
                    "qa_id": qa_id,
                    "image_path": img_rel_path,
                    "question": question_text,
                    "question_type": question_type,
                    "formatted_options": formatted_options,
                    "sa2va_answer": pred_answer,
                    "gt_answer": gt_answer,
                }
            )
            out_fh.flush()

            print("\n--- SAMPLE ---")
            print("qa_id:", qa_id)
            print("image:", img_rel_path)
            print("type:", question_type)
            print("Q:", question_text[:80], "...")
            print("OPTIONS:\n", formatted_options)
            print("RAW RESPONSE:\n", raw_response[:200], "...")
            print("PRED ANSWER:", pred_answer)
            print("GT ANSWER:", gt_answer)

            bar.update()
    except KeyboardInterrupt:
        print("[INFO] Interrupted, partial results saved.")
    finally:
        bar.close()
        out_fh.close()

    print(f"[✓] Results appended to {out_csv}")


# -------------------------------------------------------
# 5. CLI
# -------------------------------------------------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ speed boost

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="terrascope_new",
        help="Path to Sa2VA model directory"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/landsatvqa/ground_truth_files/Landsat30-AU-VQA-test.csv",
        help="CSV with qa_id,image_path,question,question_type,options(list),answer(gt)"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/landsatvqa/VQA",
        help="Root directory containing the images"
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="eval_results/landsatvqa_results_final.csv",
        help="Where to append predictions"
    )
    parser.add_argument(
        "--segmentation-json",
        type=str,
        default="data/segmentation_rle.json",
        help="(Optional) segmentation_rle.json for gt_mask_rle"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing out_csv and recompute all rows"
    )
    parser.add_argument(
        "--max-tokens-per-seg",
        type=int,
        default=8,
        help="max_tokens_per_seg passed to predict_forward_with_grounding"
    )

    args = parser.parse_args()

    print(f"[INFO] Loading Sa2VA model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    df = pd.read_csv(args.input_csv, keep_default_na=False)

    run_sa2va_vqa(
        model=model,
        tokenizer=tokenizer,
        df=df,
        image_root=Path(args.image_root),
        out_csv=Path(args.out_csv),
        segmentation_json=args.segmentation_json,
        resume=(not args.no_resume),
        max_tokens_per_seg=args.max_tokens_per_seg,
    )

    print("[DONE] sa2va_batch_vqa finished.")


if __name__ == "__main__":
    main()
