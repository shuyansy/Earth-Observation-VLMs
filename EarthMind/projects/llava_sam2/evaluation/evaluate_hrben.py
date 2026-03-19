import json
from collections import defaultdict

def load_predictions_by_category(pred_path):
    category_preds = defaultdict(dict)
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("question_id")
            ans = item.get("answer", "")
            category = item.get("category", "")
            if qid is not None and ans is not None and category:
                pred_ans = ans.strip().lower().replace(" ", "").replace(".", "")
                category_preds[category][qid] = pred_ans
    return category_preds


def load_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = {}
    for item in data.get("answers", []):
        if item.get("active", True):
            qid = item.get("question_id")
            ans = item.get("answer", "")
            if qid is not None and ans is not None:
                gt_ans = ans.strip().lower().replace(" ", "").replace(".", "")
                gt[qid] = gt_ans
    return gt


def compute_accuracy(preds, gts):
    correct = 0
    total = 0
    skipped = 0

    for qid, gt_ans in gts.items():
        pred_ans = preds.get(qid)
        if pred_ans is not None:
            total += 1
            if pred_ans == gt_ans:
                correct += 1
        else:
            skipped += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, skipped


pred_path = 'hr_vqaresult/hb_vqa_pre.jsonl'
gt_path = 'data/GeoChat-Bench/USGS_split_test_phili_answers.json'

category_preds = load_predictions_by_category(pred_path)
gts = load_ground_truth(gt_path)


target_categories = ['presence', 'comp']

for category in target_categories:
    print(f"\n📂 Category: {category}")
    preds = category_preds.get(category, {})
    acc, correct, total, skipped = compute_accuracy(preds, gts)
    print(f"✅ Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"⚠️ Skipped (no prediction for GT question_id): {skipped}")
