import argparse
import os
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import cv2
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--model_path', default="/data/new2/train_finalnew")
    parser.add_argument('--results_dir', default="aid_vqaresult", help='The dir to save results.')
    parser.add_argument('--select', type=int, default=-1)
    parser.add_argument("--annotation_file",
                        default="/data/Sa2VA-main/GeoChat-Bench/aid.jsonl", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--image_dir", default="/data/Sa2VA-main/aid_test_image", type=str,
                        help="Replace with 'data/visual_genome/images' for VG")
    args = parser.parse_args()
    return args



def evaluation_metrics(data_path):
        
    base = [json.loads(q) for q in open(data_path, "r")]
    correct=0
    incorrect=0
    for answers in tqdm(base):
        gt=answers['question_id'].split('/')[0].lower()
        answer=answers['answer'].replace(' ','').lower().replace('.','')
        if gt==answer:
                correct=correct+1
        else:
                incorrect=incorrect+1
        # else:
        #     continue
    print('correct:',correct)
    print('incorrect:',incorrect)
    print('Total:',correct+incorrect)
    print('Acc:',(correct/(correct+incorrect)))


if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda:0",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    os.makedirs(cfg.results_dir, exist_ok=True)
    results_path = f"{cfg.results_dir}/aid_vqa_pre.json"
    anno_file=cfg.annotation_file
    # with open(anno_file,"r") as f:
    #     data=json.load(f)
    data = [json.loads(q) for q in open(os.path.expanduser(cfg.annotation_file), "r")]
    all_submit=[]
    
    for i in tqdm(data):
        img_id = i['image']
        anno_id = i['question_id']
        gt_ans = i['ground_truth']
        image_path = os.path.join(cfg.image_dir, img_id)

        instruction = "<image>" + i["text"]

        vid_frames = []
        img = Image.open(image_path).convert('RGB')
        vid_frames.append(img)

        if cfg.select > 0:
            img_frame = vid_frames[cfg.select - 1]
            print(f"Selected frame {cfg.select}")
            print(f"The input is:\n{instruction}")
            result = model.predict_forward(
                image=img_frame,
                text=instruction,
                tokenizer=tokenizer,
            )
        else:
            print(f"The input is:\n{instruction}")
            result = model.predict_forward(
                video=vid_frames,
                text=instruction,
                tokenizer=tokenizer,
            )

        prediction = result['prediction']
        result_text = prediction.replace("<|end|>", "")
        print(f"The output is:\n{result_text}")

        # 注意这里重新构建字典！
        submit = {
            "question_id": img_id,    # 注意是 img_id
            "image_id": img_id,
            "answer": result_text,
            "ground_truth": gt_ans
        }
        all_submit.append(submit)


    results_path = f"{cfg.results_dir}/aid_vqa_pre.jsonl"
    

    with open(results_path, 'w') as jsonl_file:
        for entry in all_submit:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    evaluation_metrics(results_path)



  