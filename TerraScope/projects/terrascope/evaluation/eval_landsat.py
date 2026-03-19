# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-

# # import csv
# # import re
# # import pandas as pd
# # from pathlib import Path

# # ########################################
# # ########################################
# # def levenshtein(a: str, b: str) -> int:
# #     """
# #     """
# #     a = a or ""
# #     b = b or ""
# #     la, lb = len(a), len(b)
# #     dp = [[0]*(lb+1) for _ in range(la+1)]
# #     for i in range(la+1):
# #         dp[i][0] = i
# #     for j in range(lb+1):
# #         dp[0][j] = j
# #     for i in range(1, la+1):
# #         ca = a[i-1]
# #         for j in range(1, lb+1):
# #             cb = b[j-1]
# #             cost = 0 if ca == cb else 1
# #             dp[i][j] = min(
# #                 dp[i-1][j] + 1,      # delete
# #                 dp[i][j-1] + 1,      # insert
# #                 dp[i-1][j-1] + cost, # substitute
# #             )
# #     return dp[la][lb]

# # ########################################
# # ########################################
# # def normalize_text(s: str) -> str:
# #     if s is None:
# #         return ""
# #     s = s.strip()
# #     return s.lower()

# # ########################################
# # #    "A. 'Urban fabric'\nB. 'Bare ground'\nC. 'Cultivated terrestrial vegetation'\nD. 'Water bodies'"
# # #
# # #    {
# # #      "A": "Urban fabric",
# # #      "B": "Bare ground",
# # #      "C": "Cultivated terrestrial vegetation",
# # #      "D": "Water bodies",
# # #    }
# # ########################################
# # def parse_formatted_options(formatted: str):
# #     options_dict = {}
# #     if not isinstance(formatted, str):
# #         return options_dict

# #     lines = [l for l in formatted.split("\n") if l.strip()]
# #     for line in lines:
# #         m = re.match(r"^\s*([A-Za-z])\s*[\.\)]\s*(.+)$", line.strip())
# #         if m:
# #             letter = m.group(1).upper()
# #             text = m.group(2).strip()
# #         else:
# #             letter = f"UNK_{len(options_dict)}"
# #             text = line.strip()
# #         text = text.strip("\"'").strip()
# #         options_dict[letter] = text
# #     return options_dict

# # ########################################
# # ########################################
# # def pick_best_option(model_answer_raw: str, options_dict: dict):
# #     """
# #     """
# #     if not options_dict:
# #         return None, "", None

# #     norm_pred = normalize_text(model_answer_raw)

# #     best_letter = None
# #     best_text = ""
# #     best_dist = None

# #     for letter, opt_text in options_dict.items():
# #         dist = levenshtein(norm_pred, normalize_text(opt_text))
# #         if (best_dist is None) or (dist < best_dist):
# #             best_letter = letter
# #             best_text = opt_text
# #             best_dist = dist

# #     return best_letter, best_text, best_dist

# # ########################################
# # ########################################
# # def evaluate_predictions(pred_csv_path: str, out_csv_path: str):
# #     """
# #         qa_id, image_path, question, formatted_options,
# #         sa2va_answer, gt_answer, question_type
# #     )

# #     """
# #     df = pd.read_csv(pred_csv_path, keep_default_na=False)

# #     records = []
# #     correct_count = 0
# #     total = 0
    
# #     task_stats = {}  # {question_type: {"correct": count, "total": count}}

# #     for _, row in df.iterrows():
# #         qa_id = row.get("qa_id", "")
# #         formatted_options = row.get("formatted_options", "")
# #         model_answer_raw = row.get("sa2va_answer", "")
# #         gt_answer_raw = row.get("gt_answer", "")
# #         question_type = row.get("question_type", "Unknown")

# #         options_dict = parse_formatted_options(formatted_options)

# #         best_letter, best_text, best_dist = pick_best_option(model_answer_raw, options_dict)

# #         norm_best_text = normalize_text(best_text)
# #         norm_gt = normalize_text(gt_answer_raw)

# #         is_correct = (norm_best_text == norm_gt)

# #         total += 1
# #         if is_correct:
# #             correct_count += 1
        
# #         if question_type not in task_stats:
# #             task_stats[question_type] = {"correct": 0, "total": 0}
# #         task_stats[question_type]["total"] += 1
# #         if is_correct:
# #             task_stats[question_type]["correct"] += 1

# #         records.append({
# #             "qa_id": qa_id,
# #             "sa2va_answer_raw": model_answer_raw,
# #             "best_letter": best_letter,
# #             "best_text": best_text,
# #             "best_edit_distance": best_dist,
# #             "gt_answer": gt_answer_raw,
# #             "match": int(is_correct),
# #             "formatted_options": formatted_options,
# #             "question": row.get("question", ""),
# #             "image_path": row.get("image_path", ""),
# #             "question_type": question_type,
# #         })

# #     out_df = pd.DataFrame(records)
# #     out_df.to_csv(out_csv_path, index=False, encoding="utf-8")

# #     acc = correct_count / total if total > 0 else 0.0
# #     print("=======================================")
# #     print(f"Total samples: {total}")
# #     print(f"Correct via min-edit-distance: {correct_count}")
# #     print(f"Overall Accuracy: {acc:.2%}")
# #     print("=======================================")
    
# #     print("\nAccuracy by Question Type:")
# #     print("---------------------------------------")
# #     for qtype in sorted(task_stats.keys()):
# #         stats = task_stats[qtype]
# #         task_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
# #         print(f"{qtype:30s}: {stats['correct']:4d}/{stats['total']:4d} = {task_acc:.2%}")
# #     print("=======================================")
    
# #     print(f"\nSaved detailed eval to: {out_csv_path}")

# #     return acc, out_df, task_stats


# # if __name__ == "__main__":
# #     import argparse

# #     ap = argparse.ArgumentParser()
# #     ap.add_argument(
# #         "--pred-csv",
# #         type=str,
# #         default="eval_results/landsatvqa_results.csv",
# #         help="sa2va_results.csv (has columns: qa_id, formatted_options, sa2va_answer, gt_answer, question_type, ...)"
# #     )
# #     ap.add_argument(
# #         "--out-csv",
# #         type=str,
# #         default="sa2va_results_eval.csv",
# #         help="Where to write the per-sample evaluation breakdown"
# #     )
# #     args = ap.parse_args()

# #     evaluate_predictions(args.pred_csv, args.out_csv)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import csv
# import re
# import json
# import pandas as pd
# from pathlib import Path

# ########################################
# ########################################
# def levenshtein(a: str, b: str) -> int:
#     """
#     """
#     a = a or ""
#     b = b or ""
#     la, lb = len(a), len(b)
#     dp = [[0]*(lb+1) for _ in range(la+1)]
#     for i in range(la+1):
#         dp[i][0] = i
#     for j in range(lb+1):
#         dp[0][j] = j
#     for i in range(1, la+1):
#         ca = a[i-1]
#         for j in range(1, lb+1):
#             cb = b[j-1]
#             cost = 0 if ca == cb else 1
#             dp[i][j] = min(
#                 dp[i-1][j] + 1,      # delete
#                 dp[i][j-1] + 1,      # insert
#                 dp[i-1][j-1] + cost, # substitute
#             )
#     return dp[la][lb]

# ########################################
# ########################################
# def normalize_text(s: str) -> str:
#     if s is None:
#         return ""
#     s = s.strip()
#     return s.lower()

# ########################################
# #    "A. 'Urban fabric'\nB. 'Bare ground'\nC. 'Cultivated terrestrial vegetation'\nD. 'Water bodies'"
# #
# #    {
# #      "A": "Urban fabric",
# #      "B": "Bare ground",
# #      "C": "Cultivated terrestrial vegetation",
# #      "D": "Water bodies",
# #    }
# ########################################
# def parse_formatted_options(formatted: str):
#     options_dict = {}
#     if not isinstance(formatted, str):
#         return options_dict

#     lines = [l for l in formatted.split("\n") if l.strip()]
#     for line in lines:
#         m = re.match(r"^\s*([A-Za-z])\s*[\.\)]\s*(.+)$", line.strip())
#         if m:
#             letter = m.group(1).upper()
#             text = m.group(2).strip()
#         else:
#             letter = f"UNK_{len(options_dict)}"
#             text = line.strip()
#         text = text.strip("\"'").strip()
#         options_dict[letter] = text
#     return options_dict

# ########################################
# ########################################
# def pick_best_option(model_answer_raw: str, options_dict: dict):
#     """
#     """
#     if not options_dict:
#         return None, "", None

#     norm_pred = normalize_text(model_answer_raw)

#     best_letter = None
#     best_text = ""
#     best_dist = None

#     for letter, opt_text in options_dict.items():
#         dist = levenshtein(norm_pred, normalize_text(opt_text))
#         if (best_dist is None) or (dist < best_dist):
#             best_letter = letter
#             best_text = opt_text
#             best_dist = dist

#     return best_letter, best_text, best_dist

# ########################################
# ########################################
# def evaluate_predictions(pred_csv_path: str, out_csv_path: str, error_json_path: str = None):
#     """
#         qa_id, image_path, question, formatted_options,
#         sa2va_answer, gt_answer, question_type
#     )

    
#     """
#     df = pd.read_csv(pred_csv_path, keep_default_na=False)

#     records = []
#     correct_count = 0
#     total = 0
    
#     task_stats = {}  # {question_type: {"correct": count, "total": count}}

#     for idx, row in df.iterrows():
#         qa_id = row.get("qa_id", "")
#         formatted_options = row.get("formatted_options", "")
#         model_answer_raw = row.get("sa2va_answer", "")
#         gt_answer_raw = row.get("gt_answer", "")
#         question_type = row.get("question_type", "Unknown")
#         question = row.get("question", "")
#         image_path = row.get("image_path", "")

#         options_dict = parse_formatted_options(formatted_options)

#         best_letter, best_text, best_dist = pick_best_option(model_answer_raw, options_dict)

#         norm_best_text = normalize_text(best_text)
#         norm_gt = normalize_text(gt_answer_raw)

#         is_correct = (norm_best_text == norm_gt)

#         total += 1
#         if is_correct:
#             correct_count += 1
#         else:
#             error_sample = {
#                 "qa_id": qa_id,
#                 "index": int(idx),
#                 "image_path": image_path,
#                 "question": question,
#                 "question_type": question_type,
#                 "formatted_options": formatted_options,
#                 "model_answer_raw": model_answer_raw,
#                 "model_predicted": {
#                     "letter": best_letter,
#                     "text": best_text,
#                     "edit_distance": best_dist
#                 },
#                 "ground_truth": gt_answer_raw,
#                 "normalized_predicted": norm_best_text,
#                 "normalized_gt": norm_gt
#             }
#             error_samples.append(error_sample)
        
#         if question_type not in task_stats:
#             task_stats[question_type] = {"correct": 0, "total": 0}
#         task_stats[question_type]["total"] += 1
#         if is_correct:
#             task_stats[question_type]["correct"] += 1

#         records.append({
#             "qa_id": qa_id,
#             "sa2va_answer_raw": model_answer_raw,
#             "best_letter": best_letter,
#             "best_text": best_text,
#             "best_edit_distance": best_dist,
#             "gt_answer": gt_answer_raw,
#             "match": int(is_correct),
#             "formatted_options": formatted_options,
#             "question": question,
#             "image_path": image_path,
#             "question_type": question_type,
#         })

#     out_df = pd.DataFrame(records)
#     out_df.to_csv(out_csv_path, index=False, encoding="utf-8")

#     if error_json_path is None:
#         out_path = Path(out_csv_path)
#         error_json_path = out_path.parent / f"{out_path.stem}_errors.json"
    
#     with open(error_json_path, 'w', encoding='utf-8') as f:
#         json.dump({
#             "total_errors": len(error_samples),
#             "total_samples": total,
#             "error_rate": f"{(len(error_samples)/total*100):.2f}%" if total > 0 else "0%",
#             "error_samples": error_samples,
#             "error_by_type": {
#                 qtype: {
#                     "errors": stats["total"] - stats["correct"],
#                     "total": stats["total"],
#                     "error_rate": f"{((stats['total']-stats['correct'])/stats['total']*100):.2f}%" if stats["total"] > 0 else "0%"
#                 }
#                 for qtype, stats in task_stats.items()
#             }
#         }, f, indent=2, ensure_ascii=False)

#     acc = correct_count / total if total > 0 else 0.0
#     print("=======================================")
#     print(f"Total samples: {total}")
#     print(f"Correct via min-edit-distance: {correct_count}")
#     print(f"Error samples: {len(error_samples)}")
#     print(f"Overall Accuracy: {acc:.2%}")
#     print("=======================================")
    
#     print("\nAccuracy by Question Type:")
#     print("---------------------------------------")
#     for qtype in sorted(task_stats.keys()):
#         stats = task_stats[qtype]
#         task_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
#         task_errors = stats["total"] - stats["correct"]
#         print(f"{qtype:30s}: {stats['correct']:4d}/{stats['total']:4d} = {task_acc:.2%} (Errors: {task_errors})")
#     print("=======================================")
    
#     print(f"\nSaved detailed eval to: {out_csv_path}")
#     print(f"Saved error samples to: {error_json_path}")

#     return acc, out_df, task_stats, error_samples


# if __name__ == "__main__":
#     import argparse

#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--pred-csv",
#         type=str,
#         default="eval_results/landsatvqa_results.csv",
#         help="sa2va_results.csv (has columns: qa_id, formatted_options, sa2va_answer, gt_answer, question_type, ...)"
#     )
#     ap.add_argument(
#         "--out-csv",
#         type=str,
#         default="sa2va_results_eval.csv",
#         help="Where to write the per-sample evaluation breakdown"
#     )
#     ap.add_argument(
#         "--error-json",
#         type=str,
#         default="error.json",
#         help="Where to write the error samples JSON file (default: same dir as out-csv with _errors.json suffix)"
#     )
#     args = ap.parse_args()

#     evaluate_predictions(args.pred_csv, args.out_csv, args.error_json)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import json
import pandas as pd
from pathlib import Path

########################################
########################################
def levenshtein(a: str, b: str) -> int:
    """
    DP
    =ab()
    """
    a = a or ""
    b = b or ""
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1):
        dp[i][0] = i
    for j in range(lb+1):
        dp[0][j] = j
    for i in range(1, la+1):
        ca = a[i-1]
        for j in range(1, lb+1):
            cb = b[j-1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost, # substitute
            )
    return dp[la][lb]

########################################
########################################
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.strip("\"'")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

########################################
#    "A. 'Urban fabric'\nB. 'Bare ground'\nC. 'Cultivated terrestrial vegetation'\nD. 'Water bodies'"
#
#    {
#      "A": "Urban fabric",
#      "B": "Bare ground",
#      "C": "Cultivated terrestrial vegetation",
#      "D": "Water bodies",
#    }
########################################
def parse_formatted_options(formatted: str):
    options_dict = {}
    if not isinstance(formatted, str):
        return options_dict

    lines = [l for l in formatted.split("\n") if l.strip()]
    for line in lines:
        m = re.match(r"^\s*([A-Za-z])\s*[\.\)]\s*(.+)$", line.strip())
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
        else:
            letter = f"UNK_{len(options_dict)}"
            text = line.strip()
        text = text.strip("\"'").strip()
        options_dict[letter] = text
    return options_dict

########################################
########################################
def extract_option_letter(model_answer_raw: str):
    """
    A, B, C, D
     (is_letter_format, letter)
    """
    if not model_answer_raw:
        return False, None
    
    cleaned = model_answer_raw.strip().upper()
    
    patterns = [
        r'^([A-Z])$',
        r'^([A-Z])\.$',
        r'^([A-Z])\)$',
        r'^\(([A-Z])\)$',
        r'^\s*([A-Z])$',
        r'^Option\s*([A-Z])$',
        r'^[:]\s*([A-Z])$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            return True, match.group(1).upper()
    
    start_pattern = r'^([A-Z])[\.\),:\s]'
    match = re.match(start_pattern, cleaned)
    if match:
        return True, match.group(1).upper()
    
    return False, None

########################################
########################################
def pick_best_option_improved(model_answer_raw: str, options_dict: dict):
    """
    
     (best_letter, best_text, matching_method, confidence_score)
    matching_method: "letter_match"  "edit_distance"
    """
    if not options_dict:
        return None, "", "none", 0.0
    
    is_letter, letter = extract_option_letter(model_answer_raw)
    
    if is_letter and letter in options_dict:
        return letter, options_dict[letter], "letter_match", 1.0
    
    norm_pred = normalize_text(model_answer_raw)
    
    best_letter = None
    best_text = ""
    best_dist = None
    
    for opt_letter, opt_text in options_dict.items():
        dist = levenshtein(norm_pred, normalize_text(opt_text))
        if (best_dist is None) or (dist < best_dist):
            best_letter = opt_letter
            best_text = opt_text
            best_dist = dist
    
    if best_dist is not None:
        max_len = max(len(norm_pred), len(normalize_text(best_text)))
        confidence = 1.0 - (best_dist / max_len) if max_len > 0 else 0.0
    else:
        confidence = 0.0
    
    return best_letter, best_text, "edit_distance", confidence

########################################
########################################
def evaluate_predictions(pred_csv_path: str, out_csv_path: str, error_json_path: str = None):
    """
    
    pred_csv_path: sa2vacsv (
        qa_id, image_path, question, formatted_options,
        sa2va_answer, gt_answer, question_type
    )
    
    out_csv_path: csv
    
    error_json_path: JSONout_csv_errors.json
    """
    df = pd.read_csv(pred_csv_path, keep_default_na=False)
    
    records = []
    error_samples = []
    correct_count = 0
    total = 0
    
    matching_stats = {"letter_match": 0, "edit_distance": 0}
    
    task_stats = {}  # {question_type: {"correct": count, "total": count}}
    
    for idx, row in df.iterrows():
        qa_id = row.get("qa_id", "")
        formatted_options = row.get("formatted_options", "")
        model_answer_raw = row.get("sa2va_answer", "")
        gt_answer_raw = row.get("gt_answer", "")
        question_type = row.get("question_type", "Unknown")
        question = row.get("question", "")
        image_path = row.get("image_path", "")
        
        options_dict = parse_formatted_options(formatted_options)
        
        best_letter, best_text, matching_method, confidence = pick_best_option_improved(
            model_answer_raw, options_dict
        )
        
        if matching_method in matching_stats:
            matching_stats[matching_method] += 1
        
        norm_best_text = normalize_text(best_text)
        norm_gt = normalize_text(gt_answer_raw)
        
        is_correct = (norm_best_text == norm_gt)
        
        total += 1
        if is_correct:
            correct_count += 1
        else:
            is_letter_output, detected_letter = extract_option_letter(model_answer_raw)
            
            error_sample = {
                "qa_id": qa_id,
                "index": int(idx),
                "image_path": image_path,
                "question": question,
                "question_type": question_type,
                "formatted_options": formatted_options,
                "parsed_options": options_dict,
                "model_answer_raw": model_answer_raw,
                "is_letter_output": is_letter_output,
                "detected_letter": detected_letter,
                "model_predicted": {
                    "letter": best_letter,
                    "text": best_text,
                    "matching_method": matching_method,
                    "confidence": round(confidence, 3)
                },
                "ground_truth": gt_answer_raw,
                "normalized_predicted": norm_best_text,
                "normalized_gt": norm_gt,
                "error_analysis": {
                    "model_output_type": "letter" if is_letter_output else "text",
                    "correct_letter": None,
                    "model_chose_wrong_option": None
                }
            }
            
            correct_letter = None
            for letter, text in options_dict.items():
                if normalize_text(text) == norm_gt:
                    correct_letter = letter
                    break
            
            error_sample["error_analysis"]["correct_letter"] = correct_letter
            error_sample["error_analysis"]["model_chose_wrong_option"] = (
                best_letter != correct_letter if correct_letter else "GT not in options"
            )
            
            error_samples.append(error_sample)
        
        if question_type not in task_stats:
            task_stats[question_type] = {"correct": 0, "total": 0}
        task_stats[question_type]["total"] += 1
        if is_correct:
            task_stats[question_type]["correct"] += 1
        
        records.append({
            "qa_id": qa_id,
            "sa2va_answer_raw": model_answer_raw,
            "best_letter": best_letter,
            "best_text": best_text,
            "matching_method": matching_method,
            "confidence": round(confidence, 3),
            "gt_answer": gt_answer_raw,
            "match": int(is_correct),
            "formatted_options": formatted_options,
            "question": question,
            "image_path": image_path,
            "question_type": question_type,
        })
    
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")
    
    if error_json_path is None:
        out_path = Path(out_csv_path)
        error_json_path = out_path.parent / f"{out_path.stem}_errors.json"
    
    error_type_stats = {
        "letter_output_errors": sum(1 for e in error_samples if e["error_analysis"]["model_output_type"] == "letter"),
        "text_output_errors": sum(1 for e in error_samples if e["error_analysis"]["model_output_type"] == "text"),
        "wrong_option_chosen": sum(1 for e in error_samples if e["error_analysis"]["model_chose_wrong_option"] == True),
        "gt_not_in_options": sum(1 for e in error_samples if e["error_analysis"]["model_chose_wrong_option"] == "GT not in options")
    }
    
    with open(error_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_samples": total,
                "correct": correct_count,
                "errors": len(error_samples),
                "accuracy": f"{(correct_count/total*100):.2f}%" if total > 0 else "0%",
                "error_rate": f"{(len(error_samples)/total*100):.2f}%" if total > 0 else "0%"
            },
            "matching_methods_used": matching_stats,
            "error_type_distribution": error_type_stats,
            "error_by_question_type": {
                qtype: {
                    "errors": stats["total"] - stats["correct"],
                    "total": stats["total"],
                    "accuracy": f"{(stats['correct']/stats['total']*100):.2f}%" if stats["total"] > 0 else "0%",
                    "error_rate": f"{((stats['total']-stats['correct'])/stats['total']*100):.2f}%" if stats["total"] > 0 else "0%"
                }
                for qtype, stats in task_stats.items()
            },
            "error_samples": error_samples
        }, f, indent=2, ensure_ascii=False)
    
    acc = correct_count / total if total > 0 else 0.0
    print("=======================================")
    print(f"Total samples: {total}")
    print(f"Correct: {correct_count}")
    print(f"Errors: {len(error_samples)}")
    print(f"Overall Accuracy: {acc:.2%}")
    print("=======================================")
    
    print("\nMatching Methods Used:")
    print("---------------------------------------")
    for method, count in matching_stats.items():
        print(f"{method:20s}: {count:4d} ({count/total*100:.1f}%)")
    print("=======================================")
    
    if error_samples:
        print("\nError Type Analysis:")
        print("---------------------------------------")
        for error_type, count in error_type_stats.items():
            print(f"{error_type:25s}: {count:4d}")
        print("=======================================")
    
    print("\nAccuracy by Question Type:")
    print("---------------------------------------")
    for qtype in sorted(task_stats.keys()):
        stats = task_stats[qtype]
        task_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        task_errors = stats["total"] - stats["correct"]
        print(f"{qtype:30s}: {stats['correct']:4d}/{stats['total']:4d} = {task_acc:.2%} (Errors: {task_errors})")
    print("=======================================")
    
    print(f"\nSaved detailed eval to: {out_csv_path}")
    print(f"Saved error analysis to: {error_json_path}")
    
    return acc, out_df, task_stats, error_samples


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-csv",
        type=str,
        default="eval_results/landsatvqa_results_final.csv",
        help="sa2va_results.csv (has columns: qa_id, formatted_options, sa2va_answer, gt_answer, question_type, ...)"
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="sa2va_results_eval.csv",
        help="Where to write the per-sample evaluation breakdown"
    )
    ap.add_argument(
        "--error-json",
        type=str,
        default=None,
        help="Where to write the error samples JSON file (default: same dir as out-csv with _errors.json suffix)"
    )
    args = ap.parse_args()
    
    evaluate_predictions(args.pred_csv, args.out_csv, args.error_json)