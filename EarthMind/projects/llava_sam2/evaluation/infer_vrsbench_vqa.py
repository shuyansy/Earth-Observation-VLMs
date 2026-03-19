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
    parser.add_argument('--model_path', default="/data/Sa2VA-main/train_weight_newnew")
    parser.add_argument('--results_dir', default="vrs_vqaresult", help='The dir to save results.')
    parser.add_argument('--select', type=int, default=-1)
    parser.add_argument("--annotation_file",
                        default="/data/groundingLMM2/data/vrsbenchvqa/VRSBench_EVAL_vqa.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--image_dir", default="/data/groundingLMM2/data/vrsbenchvqa/Images_val", type=str,
                        help="Replace with 'data/visual_genome/images' for VG")
    args = parser.parse_args()
    return args


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
    results_path = f"{cfg.results_dir}/vrsbench_vqa_new.json"
    anno_file=cfg.annotation_file
    with open(anno_file,"r") as f:
        data=json.load(f)
    all_submit=[]
    for i in tqdm(data):
        img_id = i['image_id']
        anno_id=i['question_id']
        gt_ans=i['ground_truth']
        image_path = os.path.join(cfg.image_dir,img_id)

        
        instruction ="<image>"+ i["question"] + "Answer the question using a single word or phrase."

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
        result=prediction.replace("<|end|>","")
        print(f"The output is:\n{result}")

        
        q_type = i['type']
        submit={}
        submit["question_id"]=anno_id
        submit["image_id"]=img_id
        submit["question"]=i["question"]
        submit["ground_truth"]=gt_ans 
        submit["type"]=q_type      
        submit["answer"]=result
        all_submit.append(submit)

        print("#########",i["question"],result)

    with open(results_path, 'w') as json_file:
        json.dump(all_submit, json_file, indent=2)


  