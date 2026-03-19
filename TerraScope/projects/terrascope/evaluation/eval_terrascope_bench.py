#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from collections import defaultdict

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure. Expect a list or a dict with key 'results'.")

def norm_ans(x):
    """
    -  option_id text
    - A/B/C/D
    """
    if isinstance(x, dict):
        if "option_id" in x and isinstance(x["option_id"], str) and x["option_id"].strip():
            return x["option_id"].strip().upper()
        if "text" in x and isinstance(x["text"], str):
            m = re.match(r"\s*([A-D])\b", x["text"].strip(), flags=re.IGNORECASE)
            return m.group(1).upper() if m else x["text"].strip()
        return ""
    if isinstance(x, str):
        s = x.strip()
        m = re.match(r"\s*([A-D])\b", s, flags=re.IGNORECASE)
        return m.group(1).upper() if m else s.upper()
    return str(x).strip().upper()

def maybe_parse_from_response(sample):
    """
     predicted_answer  response  <answer>...</answer> 
    """
    resp = sample.get("response", "")
    if not isinstance(resp, str):
        return None
    m = re.search(r"<answer>(.*?)</answer>", resp, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    inner = m.group(1)
    m2 = re.search(r"\b([A-D])\b", inner, flags=re.IGNORECASE)
    return m2.group(1).upper() if m2 else None

def main():
    ap = argparse.ArgumentParser(description="Compute per-task accuracy from results JSON.")
    ap.add_argument("--in", dest="input_path", default="eval_results/terrascope_bench_results.json", help="Path to the JSON results file.")
    ap.add_argument("--save-csv", dest="csv_path", default=None, help="Optional path to save per-task accuracy as CSV.")
    args = ap.parse_args()

    samples = load_results(args.input_path)

    per_task_total = defaultdict(int)
    per_task_correct = defaultdict(int)

    total = 0
    correct = 0

    for s in samples:
        task_type = s.get("task_type", "UNKNOWN")
        gt = norm_ans(s.get("gt_answer", ""))
        pred = s.get("predicted_answer", None)
        if pred is None:
            parsed = maybe_parse_from_response(s)
            pred = parsed if parsed is not None else ""
        pred = norm_ans(pred)

        per_task_total[task_type] += 1
        total += 1

        is_right = (pred == gt) and (gt != "")
        if is_right:
            per_task_correct[task_type] += 1
            correct += 1

    print("\nPer-task accuracy")
    print("-" * 60)
    print(f"{'task_type':<28}{'total':>8}{'correct':>10}{'accuracy':>12}")
    print("-" * 60)
    rows = []
    for t in sorted(per_task_total.keys()):
        tot = per_task_total[t]
        cor = per_task_correct[t]
        acc = (cor / tot * 100.0) if tot > 0 else 0.0
        rows.append((t, tot, cor, acc))
        print(f"{t:<28}{tot:>8}{cor:>10}{acc:>11.1f}%")
    print("-" * 60)
    overall = (correct / total * 100.0) if total > 0 else 0.0
    print(f"{'OVERALL':<28}{total:>8}{correct:>10}{overall:>11.1f}%\n")

    if args.csv_path:
        import csv
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task_type", "total", "correct", "accuracy(%)"])
            for t, tot, cor, acc in rows:
                w.writerow([t, tot, cor, f"{acc:.2f}"])
            w.writerow(["OVERALL", total, correct, f"{overall:.2f}"])
        print(f"Saved CSV to: {args.csv_path}")

if __name__ == "__main__":
    main()
