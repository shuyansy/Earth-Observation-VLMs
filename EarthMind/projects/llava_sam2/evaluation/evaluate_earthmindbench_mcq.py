import os
import json
from collections import defaultdict

# ===== 配置路径 =====
json_folder = "/data/Earthmind_proj/pair_multimcq_rgb_final1"  # 替换为你的路径

# ===== 初始化 =====
task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# ===== 遍历所有任务文件 =====
for file_name in os.listdir(json_folder):
    if not file_name.endswith(".json"):
        continue

    # 🧩 任务名提取规则：
    # 去掉后缀，只保留前两个下划线前的部分，例如：
    #   "scene_classification_unmatched_result.json" → "scene_classification"
    #   "ms_scene_classification_qa_result.json" → "scene_classification"
    name = file_name.replace(".json", "")
    if "scene_classification" in name:
        task_name = "scene_classification"
    elif "object_counting" in name:
        task_name = "object_counting"
    elif "spatial_relationship" in name:
        task_name = "spatial_relationship"
    elif "object_existence" in name:
        task_name = "object_existence"
    elif "hallucination_detection" in name:
        task_name = "hallucination_detection"
    else:
        # 默认：去掉 "_all" 或 "_qa" 等尾巴
        task_name = name.split("_")[0]

    file_path = os.path.join(json_folder, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct = 0
    total = 0
    for item in data:
        pred = item.get("pred")
        gt = item.get("ground_truth")
        if pred is not None and gt is not None:
            total += 1
            if pred == gt:
                correct += 1

    task_stats[task_name]["correct"] += correct
    task_stats[task_name]["total"] += total

    acc = correct / total if total > 0 else 0.0
    # print(f"{file_name}: ✅ Accuracy = {acc:.4f} ({correct}/{total})")

# ===== 汇总每个任务 =====
print("\n====== accuracy ======")
task_accs = []
for task, v in task_stats.items():
    c, t = v["correct"], v["total"]
    acc = c / t if t > 0 else 0.0
    task_accs.append(acc)
    print(f"{task}: ✅ Merged Accuracy = {acc:.4f} ({c}/{t})")

# ===== 计算整体平均 =====
if task_accs:
    avg = sum(task_accs) / len(task_accs)
    print(f"\n📊 Average Accuracy across {len(task_accs)} merged tasks = {avg:.4f}")
else:
    print("\n❌ 没有找到有效的任务数据。")
