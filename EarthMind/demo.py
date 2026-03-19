import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import cv2
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--image_folder', default="demo_images", help='Path to image file')
    parser.add_argument('--model_path', default="/scqian/EarthMind-4B")
    parser.add_argument('--work-dir', default="result", help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please segment the left chimney.")
    parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args


def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

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

    image_files = []
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    for filename in sorted(list(os.listdir(cfg.image_folder))):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)
            image_paths.append(os.path.join(cfg.image_folder, filename))

    vid_frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        vid_frames.append(img)


    if cfg.select > 0:
        img_frame = vid_frames[cfg.select - 1]

        print(f"Selected frame {cfg.select}")
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            image=img_frame,
            text=cfg.text,
            tokenizer=tokenizer,
        )
    else:
        print("##########",vid_frames[0])
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            video=vid_frames,
            text=cfg.text,
            tokenizer=tokenizer,
        )

    prediction = result['prediction']
    print(f"The output is:\n{prediction}")

    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks = result['prediction_masks'][_seg_idx]
        for frame_idx in range(len(vid_frames)):
            pred_mask = pred_masks[frame_idx]
            if cfg.work_dir:
                os.makedirs(cfg.work_dir, exist_ok=True)
                visualize(pred_mask, image_paths[frame_idx], cfg.work_dir)
            else:
                os.makedirs('./temp_visualize_results', exist_ok=True)
                visualize(pred_mask, image_paths[frame_idx], './temp_visualize_results')
    else:
        pass
