#!/usr/bin/env python3
"""
Evaluation script for disaster benchmark inference results.

Usage:
    python eval_disaster.py --input /path/to/disaster_inference_results.json
    python eval_disaster.py --input /path/to/results.json --reparse   # re-extract answers from model_output
"""

import argparse
import json
import re
from collections import Counter, defaultdict


def extract_answer_strict(output: str) -> str:
    """
    Improved answer extraction from model output.

    Priority:
    1. <answer>A</answer> or <answer>A. 232</answer> → extract letter(s)
    2. <answer>B, E, F, G</answer> → extract multiple letters (multi-label)
    3. Fallback: last letter A-E in the output (not from <think>)
    """
    if not output or output == "ERROR":
        return "N/A"

    # 1. Try <answer>...</answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Multi-label: "B, E, F, G" or "B, E, F, G."
        multi_letters = re.findall(r'\b([A-H])\b', answer_text.upper())
        if multi_letters:
            return ', '.join(sorted(set(multi_letters)))

    # 2. Fallback: search after </think> if present
    think_end = output.rfind('</think>')
    search_region = output[think_end:] if think_end != -1 else output

    # Look for patterns like "A. 232" or standalone "A"
    letter_match = re.findall(r'\b([A-E])\b', search_region.upper())
    if letter_match:
        return letter_match[-1]  # take the last one

    return "N/A"


def normalize_gt(gt_option: str) -> str:
    """Normalize ground truth option string."""
    if not gt_option:
        return "N/A"
    gt = gt_option.strip().rstrip('.')
    # Multi-label: "B, E, F, G" → sorted
    letters = re.findall(r'\b([A-H])\b', gt.upper())
    if letters:
        return ', '.join(sorted(set(letters)))
    return gt.upper()


def evaluate(predictions: list, reparse: bool = False):
    """Evaluate predictions and print metrics."""

    # ── Per-task grouping ──
    by_task = defaultdict(list)
    for p in predictions:
        task = p.get('task', 'Unknown')
        gt = normalize_gt(p.get('ground_truth_option', ''))

        if reparse and 'model_output' in p:
            pred = extract_answer_strict(p['model_output'])
        else:
            pred = p.get('pred_option', 'N/A')
            # Also normalize multi-label pred
            if ',' in pred:
                letters = re.findall(r'[A-H]', pred.upper())
                pred = ', '.join(sorted(set(letters)))
            else:
                pred = pred.strip().upper()

        by_task[task].append({
            'gt': gt,
            'pred': pred,
            'output': p.get('model_output', ''),
        })

    # ── Print results ──
    print("=" * 80)
    print(f"{'DISASTER BENCHMARK EVALUATION':^80}")
    print("=" * 80)

    total_correct = 0
    total_count = 0
    total_na = 0
    total_error = 0

    for task in sorted(by_task.keys()):
        items = by_task[task]
        correct = sum(1 for it in items if it['pred'] == it['gt'])
        na_count = sum(1 for it in items if it['pred'] == 'N/A')
        err_count = sum(1 for it in items if it['pred'] == 'ERROR')
        n = len(items)
        acc = correct / n * 100 if n > 0 else 0

        print(f"\n{'─' * 80}")
        print(f"  Task: {task}")
        print(f"  Samples: {n}  |  Correct: {correct}  |  Accuracy: {acc:.2f}%")
        if na_count > 0:
            print(f"  N/A (no answer extracted): {na_count}")
        if err_count > 0:
            print(f"  ERROR (inference failed): {err_count}")

        # Prediction distribution
        pred_dist = Counter(it['pred'] for it in items)
        gt_dist = Counter(it['gt'] for it in items)
        print(f"  Pred distribution: {dict(pred_dist.most_common())}")
        print(f"  GT   distribution: {dict(gt_dist.most_common())}")

        # Per-option accuracy
        gt_groups = defaultdict(list)
        for it in items:
            gt_groups[it['gt']].append(it['pred'] == it['gt'])
        per_option = {k: f"{sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)"
                      for k, v in sorted(gt_groups.items())}
        print(f"  Per-GT accuracy: {per_option}")

        # Show some wrong examples
        wrong = [it for it in items if it['pred'] != it['gt']]
        if wrong and len(wrong) <= 5:
            print(f"  Wrong examples ({len(wrong)}):")
            for w in wrong[:5]:
                out_short = w['output'][:150].replace('\n', ' ')
                print(f"    pred={w['pred']} gt={w['gt']}  output: {out_short}...")

        total_correct += correct
        total_count += n
        total_na += na_count
        total_error += err_count

    # ── Overall ──
    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"\n{'=' * 80}")
    print(f"  OVERALL")
    print(f"  Total samples: {total_count}")
    print(f"  Total correct: {total_correct}")
    print(f"  Overall accuracy: {overall_acc:.2f}%")
    if total_na > 0:
        print(f"  Total N/A: {total_na} ({total_na/total_count*100:.1f}%)")
    if total_error > 0:
        print(f"  Total ERROR: {total_error} ({total_error/total_count*100:.1f}%)")
    print("=" * 80)

    return {
        'overall_accuracy': overall_acc,
        'total_samples': total_count,
        'total_correct': total_correct,
        'per_task': {
            task: {
                'accuracy': sum(1 for it in items if it['pred'] == it['gt']) / len(items) * 100,
                'count': len(items),
                'correct': sum(1 for it in items if it['pred'] == it['gt']),
            }
            for task, items in by_task.items()
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate disaster benchmark results")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to inference results JSON")
    parser.add_argument("--reparse", action="store_true",
                        help="Re-extract answers from model_output using improved parser")
    parser.add_argument("--save", type=str, default=None,
                        help="Save evaluation metrics to this JSON file")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support both formats: {predictions: [...]} or [...]
    if isinstance(data, dict):
        predictions = data.get('predictions', [])
        print(f"Model: {data.get('model_path', 'N/A')}")
        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    else:
        predictions = data

    if not predictions:
        print("No predictions found!")
        return

    print(f"Loaded {len(predictions)} predictions")
    if args.reparse:
        print("** Re-parsing answers from model_output **")

    metrics = evaluate(predictions, reparse=args.reparse)

    if args.save:
        with open(args.save, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.save}")


if __name__ == "__main__":
    main()
