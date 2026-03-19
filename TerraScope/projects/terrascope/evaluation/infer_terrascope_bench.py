


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TerraScope-Bench inference from HuggingFace parquet.
Images are read directly from the parquet file, no separate image directory needed.

Usage:
  # Run all tasks
  python infer_terrascope_bench.py

  # Run a specific task
  python infer_terrascope_bench.py --filter-task-type boundary_detection

  # Load from HuggingFace Hub
  python infer_terrascope_bench.py --benchmark-parquet hf://datasets/sy1998/TerraScope-Bench/data/TerraScope_bench.parquet

  # Specify model and device
  python infer_terrascope_bench.py --model-path /path/to/model --device cuda:0
"""

import json
import io
import os
import re
from collections import defaultdict

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(response: str):
    """
    1.  <answer>...</answer> 
    2.  A/B/C/D 
    3. 
    """
    if not response:
        return None

    match_block = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        flags=re.DOTALL | re.IGNORECASE
    )

    if match_block:
        raw_ans = match_block.group(1).strip()
    else:
        raw_ans = response.strip()

    match_letter = re.match(r"^\s*([A-Da-d])[\.\)]?\s*", raw_ans)
    if match_letter:
        return match_letter.group(1).upper()

    return raw_ans


def format_question_with_candidates(question: str, candidates: dict, custom_prompt: str):
    """
        <custom_prompt><image>
        {question}
        A. ...
        B. ...
    """
    formatted = f"{question}\n"
    for key in sorted(candidates.keys()):
        formatted += f"{key}. {candidates[key]}\n"

    final_text = custom_prompt + "<image>\n" + formatted
    return final_text


def load_image_from_parquet_cell(image_cell):
    """ parquet  image  PIL Image dict{"bytes","path"}  bytes"""
    if isinstance(image_cell, dict):
        img_bytes = image_cell.get("bytes", b"")
    elif isinstance(image_cell, bytes):
        img_bytes = image_cell
    else:
        return None

    if not img_bytes:
        return None

    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def run_sa2va_inference(
    model_path: str,
    benchmark_parquet_path: str,
    output_json_path: str,
    filter_task_type: str = None,
    device: str = "cuda:0"
):
    """
     Sa2VA  parquet benchmark 
    """

    print("Loading Sa2VA model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    custom_prompt = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in their mind and then provides the user a concise final answer in a "
        "short word or phrase. The reasoning process and answer are enclosed within "
        "<think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>\n\n"
    )

    print(f"Loading benchmark data from {benchmark_parquet_path}...")
    df = pd.read_parquet(benchmark_parquet_path)
    print(f"Loaded {len(df)} samples")

    if filter_task_type:
        original_count = len(df)
        df = df[df["task_type"] == filter_task_type].reset_index(drop=True)
        print(f"Filtered to {len(df)} samples (task_type={filter_task_type}) from {original_count} total")

    results = []
    correct = 0
    total = 0

    print("Starting inference...")
    for idx in tqdm(range(len(df)), desc="Processing"):
        row = df.iloc[idx]

        img = load_image_from_parquet_cell(row["image"])
        if img is None:
            print(f"Warning: Cannot load image for row {idx}, skipping...")
            continue

        candidates = row["candidates"]
        if isinstance(candidates, str):
            candidates = json.loads(candidates)

        question_text = format_question_with_candidates(
            row["question"],
            candidates,
            custom_prompt=custom_prompt
        )

        try:
            result = model.predict_forward_with_grounding(
                image=img,
                text=question_text,
                tokenizer=tokenizer,
                max_tokens_per_seg=8,
            )
            response = result.get("prediction", "")
        except Exception as e:
            image_name = row["image"].get("path", f"row_{idx}") if isinstance(row["image"], dict) else f"row_{idx}"
            print(f"Error generating response for {image_name}: {e}")
            response = ""

        print("result", response)
        predicted_answer = extract_answer(response)
        gt_answer = row["answer"]

        image_name = row["image"].get("path", f"row_{idx}") if isinstance(row["image"], dict) else f"row_{idx}"

        result_item = {
            "image": image_name,
            "question": row["question"],
            "predicted_answer": predicted_answer,
            "gt_answer": gt_answer,
            "response": response,
            "task_type": row.get("task_type", "unknown")
        }
        results.append(result_item)

        is_correct = (predicted_answer == gt_answer)
        if is_correct:
            correct += 1
        total += 1

        print(f"\n--- Sample {total} ---")
        print(f"Image: {image_name}")
        print(f"Question: {row['question'][:60]}...")
        print(f"Response: {response[:200]}...")
        print(f"Predicted: {predicted_answer} | GT: {gt_answer} | Correct: {is_correct}")

    per_task = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        tt = r["task_type"]
        per_task[tt]["total"] += 1
        if r["predicted_answer"] == r["gt_answer"]:
            per_task[tt]["correct"] += 1

    accuracy = correct / total if total > 0 else 0.0

    output_data = {
        "model": "Sa2VA",
        "model_path": model_path,
        "filter_task_type": filter_task_type,
        "results": results,
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }
    }

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Inference completed!")
    print(f"{'Task':<28}{'Total':>8}{'Correct':>10}{'Accuracy':>12}")
    print("-" * 60)
    for t in sorted(per_task.keys()):
        s = per_task[t]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"{t:<28}{s['total']:>8}{s['correct']:>10}{acc:>11.2f}%")
    print("-" * 60)
    print(f"{'OVERALL':<28}{total:>8}{correct:>10}{accuracy * 100:>11.2f}%")
    print(f"\nResults saved to {output_json_path}")
    print("=" * 60)

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sa2VA benchmark inference from parquet")

    parser.add_argument(
        "--model-path",
        type=str,
        default="terrascope_new",
        help="Path to Sa2VA model"
    )
    parser.add_argument(
        "--benchmark-parquet",
        type=str,
        default="data/TerraScope_bench.parquet",
        help="Path to benchmark parquet file (local or hf://datasets/...)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="terrascope_bench_sa2va_results_final.json",
        help="Output results JSON path"
    )
    parser.add_argument(
        "--filter-task-type",
        type=str,
        default="None",
        help="Filter specific task type (None for all tasks)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )

    args = parser.parse_args()

    filter_type = None if args.filter_task_type.lower() in ["none", "all"] else args.filter_task_type

    run_sa2va_inference(
        model_path=args.model_path,
        benchmark_parquet_path=args.benchmark_parquet,
        output_json_path=args.output_json,
        filter_task_type=filter_type,
        device=args.device
    )
