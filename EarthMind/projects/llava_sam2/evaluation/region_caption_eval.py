import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tempfile
import os

def evaluate_caption(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    annotations = []
    predictions = []
    images = []

    for idx, item in enumerate(data):
        image_id = item["image_id"]
        caption = item["caption"]
        prediction = item["prediction"]
        
        annotations.append({
            "image_id": image_id,
            "caption": caption,
            "id": idx  
        })

        predictions.append({
            "image_id": image_id,
            "caption": prediction
        })

        images.append({"id": image_id})

    
    images = list({img['id']: img for img in images}.values())

    # 保存成临时文件
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as gt_f:
        json.dump({
            "images": images,
            "annotations": annotations
        }, gt_f)
        gt_file = gt_f.name

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as res_f:
        json.dump(predictions, res_f)
        res_file = res_f.name

    # 加载 COCO
    coco = COCO(gt_file)
    coco_res = coco.loadRes(res_file)

    # 评估
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    # 删除临时文件
    os.remove(gt_file)
    os.remove(res_file)

    return coco_eval.eval

if __name__ == "__main__":
    result = evaluate_caption('region_cap_pred.json')  # 换成你的 JSON 路径
    for metric, score in result.items():
        print(f"{metric}: {score:.4f}")
