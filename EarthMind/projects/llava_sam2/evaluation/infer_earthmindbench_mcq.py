import argparse
import os
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_BAND_RE = re.compile(r"_B(\d{2}|8A)\b", re.IGNORECASE)

def read_band(fp):
    arr = np.array(Image.open(fp))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def resize_to(arr, H, W):
    if arr.shape == (H, W):
        return arr
    return np.array(Image.fromarray(arr).resize((W, H), resample=Image.BILINEAR))

def _stretch_to_uint8(a: np.ndarray):
    # 2–98 分位拉伸，便于可视化/兼容常规 RGB transform
    a = a.astype(np.float32)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a, dtype=np.uint8)
    p2, p98 = np.percentile(v, 2), np.percentile(v, 98)
    if p98 - p2 < 1e-6:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - p2) / (p98 - p2)
    a = np.clip(a, 0, 1)
    return (a * 255.0).astype(np.uint8)


def ms_filelist_to_rgb_and_extra_groups(
    file_list: list,
    to_uint8: bool = True,
    pad_mode: str = "repeat"  # "repeat" 或 "zero"：最后一组不满 3 通道的填充策略
):
    """
    返回若干 PIL.Image（RGB 模式）：
      [0] = 标准 RGB：B04(红), B03(绿), B02(蓝)
      [1..] = 其余通道按 3 个一组组成的"伪 RGB"
    
    Args:
        file_list: TIF文件路径列表，例如 ['/path/1_B01.tif', '/path/1_B02.tif', ...]
        to_uint8: 是否转换为uint8
        pad_mode: 填充模式 "repeat" 或 "zero"
    """
    if not file_list:
        raise ValueError(f"Empty file list provided")
    
    # 从文件路径中提取波段信息
    band_to_file = {}
    for fp in file_list:
        if not fp.lower().endswith(".tif"):
            continue
        filename = os.path.basename(fp)
        # 假设文件名格式为 "数字_Bxx.tif" 例如 "1_B01.tif"
        # 匹配波段名称 (B01, B02, ..., B8A等)
        m = re.search(r'_B(\d+A?)\.tif$', filename, re.IGNORECASE)
        if m:
            band_name = "B" + m.group(1).upper()
            band_to_file[band_name] = fp
    
    # 必须有 B02,B03,B04
    for b in ("B02", "B03", "B04"):
        if b not in band_to_file:
            raise RuntimeError(f"Missing required band {b} in file list")
    
    # 读取三波段 + 对齐尺寸
    r = read_band(band_to_file["B04"])
    g = read_band(band_to_file["B03"])
    b = read_band(band_to_file["B02"])
    
    H = max(r.shape[0], g.shape[0], b.shape[0])
    W = max(r.shape[1], g.shape[1], b.shape[1])
    
    r = resize_to(r, H, W)
    g = resize_to(g, H, W)
    b = resize_to(b, H, W)
    
    if to_uint8:
        R = Image.fromarray(_stretch_to_uint8(r), mode='L')
        G = Image.fromarray(_stretch_to_uint8(g), mode='L')
        B = Image.fromarray(_stretch_to_uint8(b), mode='L')
    else:
        # 若你有别的归一化方案，可以在这里改
        R = Image.fromarray(_stretch_to_uint8(r), mode='L')
        G = Image.fromarray(_stretch_to_uint8(g), mode='L')
        B = Image.fromarray(_stretch_to_uint8(b), mode='L')
    
    images = [Image.merge('RGB', (R, G, B))]  # 第 0 张：标准 RGB
    
    # 其余通道顺序（按 Sentinel-2 常见波段，去掉 02,03,04）
    rest = ["B01", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    rest_files = [band_to_file[b] for b in rest if b in band_to_file]
    
    # 读并对齐到同一尺寸
    rest_arrs = []
    for fp in rest_files:
        a = read_band(fp)
        a = resize_to(a, H, W)
        rest_arrs.append(a)
    
    # 按 3 通道一组组成伪 RGB
    i = 0
    while i < len(rest_arrs):
        group = rest_arrs[i:i+3]
        if len(group) < 3:
            if pad_mode == "repeat":
                while len(group) < 3:
                    group.append(group[-1])
            else:  # zero
                while len(group) < 3:
                    group.append(np.zeros_like(group[0], dtype=group[0].dtype))
        
        if to_uint8:
            R = Image.fromarray(_stretch_to_uint8(group[0]), mode='L')
            G = Image.fromarray(_stretch_to_uint8(group[1]), mode='L')
            B = Image.fromarray(_stretch_to_uint8(group[2]), mode='L')
        else:
            R = Image.fromarray(_stretch_to_uint8(group[0]), mode='L')
            G = Image.fromarray(_stretch_to_uint8(group[1]), mode='L')
            B = Image.fromarray(_stretch_to_uint8(group[2]), mode='L')
        
        images.append(Image.merge('RGB', (R, G, B)))
        i += 3
    
    return images

def weights_to_map_16x16(w_patch, patch_size=(448, 448)):
    if not isinstance(w_patch, torch.Tensor):
        w = torch.tensor(w_patch)
    else:
        w = w_patch
    if w.ndim == 2 and w.size(-1) == 1:
        w = w.squeeze(-1)                   # (256,)
    w = w.reshape(16, 16)[None, None, ...]  # (1,1,16,16)

    # 关键：最近邻，不要 bilinear
    w = w.to(torch.float32)
    w = F.interpolate(w, size=patch_size, mode='nearest')  # (1,1,H,W)
    w = w[0, 0]

    # 可选：归一化到 [0,1]（保持对比）
    mn, mx = w.min(), w.max()
    w = (w - mn) / (mx - mn + 1e-6)
    return w


# --- 按 rows×cols 拼接回整图 ---
def stitch_grid(weight_list, rows, cols, patch_size=(448, 448)):
    H, W = patch_size
    assert len(weight_list) >= rows * cols
    idx = 0
    rows_cat = []
    for _ in range(rows):
        row = torch.cat([weight_list[idx + c] for c in range(cols)], dim=1)  # (H, cols*W)
        rows_cat.append(row)
        idx += cols
    full = torch.cat(rows_cat, dim=0)  # (rows*H, cols*W)
    return full

def make_dense_map_from_blocks(weight_blocks, rows, cols, patch_size=(448,448)):
    """
    weight_blocks: (blocks, 256) 或 (blocks, 256, 1) 的 torch.Tensor
    返回: np.ndarray, (H_full, W_full), 已归一化到 [0,1]
    """
    blocks = rows * cols
    w = weight_blocks[:blocks].to(torch.float32)  # 先截取、转 float32
    maps = [weights_to_map_16x16(w[i], patch_size) for i in range(blocks)]
    full_t = stitch_grid(maps, rows, cols, patch_size)  # torch.float32
    return full_t.cpu().numpy()

def plot_img_and_heatmap(rgb_img_pil, sar_img_pil,
                         rgb_weight_blocks, sar_weight_blocks,
                         rows, cols, save_path=None, show=True,
                         cmap='viridis'):
    """
    rgb_img_pil, sar_img_pil: PIL.Image (原图，会被 resize 成 rows*448 × cols*448)
    rgb_weight_blocks, sar_weight_blocks: Tensor，(blocks, 256) 或 (blocks, 256, 1)
    rows, cols: 动态切块得到的行列数（result['sta']）
    """

    patch_size = (448, 448)
    H_full, W_full = rows * patch_size[0], cols * patch_size[1]

    # 生成密集权重图（每像素都有值）
    rgb_dense = make_dense_map_from_blocks(rgb_weight_blocks, rows, cols, patch_size)  # (H_full, W_full)
    sar_dense = make_dense_map_from_blocks(sar_weight_blocks, rows, cols, patch_size)  # (H_full, W_full)

    # 把原图 resize 到相同大小
    rgb_np = np.array(rgb_img_pil.resize((W_full, H_full)))
    sar_np = np.array(sar_img_pil.resize((W_full, H_full)))

    # 画两行两列：左原图，右权重
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(rgb_np); plt.title('RGB image'); plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(rgb_dense, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    plt.title('RGB weight (dense)'); plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(sar_np); plt.title('SAR image'); plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(sar_dense, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    plt.title('SAR weight (dense)'); plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    else:
        plt.close()




try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--model_path', default="/data/Earthmind_proj/final_retrain_multi")
    parser.add_argument('--results_dir', default="pair_multimcq_rgb_final1", help='The dir to save results.')
    parser.add_argument('--select', type=int, default=-1)
    # parser.add_argument("--annotation_files", nargs='+', default=["/data/Earthmind_proj/data/formal_data_test/ms_scene_classification_qa.json","data/formal_data_test/scene_classification_unmatched.json","data/formal_data_test/object_existence_all_unmatched.json","data/formal_data_test/spatial_relationship_all_unmatched.json","data/formal_data_test/hallucination_detection_all_unmatched.json","data/formal_data_test/object_counting_all_unmatched.json",\
    # ],
    #                     help="List of annotation JSON files.")
    parser.add_argument("--annotation_files", nargs='+', default=["/data/Earthmind_proj/data/formal_data_test/ms_scene_classification_qa.json"],help="List of annotation JSON files.")
    parser.add_argument("--image_dir", default="/data/Earthmind_proj/data/pair_data/test", type=str,
                        help="Root folder for SAR/RGB images.")
    parser.add_argument("--ms_image_dir", default="/data/Earthmind_proj/data/pair_data/test/ms_sar", type=str,
                        help="Root folder for SAR/RGB images.")
    return parser.parse_args()

if __name__ == "__main__":
    cfg = parse_args()

    # 模型加载
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype="auto",
        device_map="cuda:0",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True
    )

    os.makedirs(cfg.results_dir, exist_ok=True)

    for anno_file in cfg.annotation_files:
        with open(anno_file, "r") as f:
            data = json.load(f)

        all_submit = []
        false = []
        print("#######",anno_file)

        for i in tqdm(data, desc=f"Processing {os.path.basename(anno_file)}"):

            
            file_name = i["file_name"]
            
            if "ms" not in anno_file:
                print("2222222222")
                image_path = os.path.join(cfg.image_dir, "sar/img", file_name.replace(".json", ".png"))
                rgb_image_path = os.path.join(cfg.image_dir, "rgb/img", file_name.replace(".json", ".png"))

                options = i.get("candidate", [])
                option_str = "\n" + "\n".join(options) if options else ""
                instruction = "<image>" + i["question"] + option_str + "\nonly output the correct option."

                try:
                    img = Image.open(image_path).convert('RGB')
                    rgb_img = Image.open(rgb_image_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading image: {image_path} or {rgb_image_path} - {e}")
                    false.append(file_name)
                    continue
            else:
                print("111111111")
                image_path = os.path.join(cfg.ms_image_dir, "sar/img", file_name+".png")
            
                options = i.get("candidate", [])
                option_str = "\n" + "\n".join(options) if options else ""
                instruction = "<image>" + i["question"] + option_str

                img = Image.open(image_path).convert('RGB')
                rgb_dir = os.path.join(cfg.ms_image_dir, "ms/img")
        
                # Method 1: Using glob to find all matching TIF files
                rgb_image_pattern = os.path.join(rgb_dir, f"{file_name}_*.tif")
                rgb_image_list = glob.glob(rgb_image_pattern)
                rgb_image_list.sort()  # Sort to ensure consistent order (B01, B02, etc.)
              
                rgb_img =ms_filelist_to_rgb_and_extra_groups(rgb_image_list, to_uint8=True, pad_mode="repeat")



            result = model.predict_forward_multi(
                image=img,
                rgb_image=rgb_img,
                text=instruction,
                tokenizer=tokenizer,
            )

            vis_weight=result['vis_weight']
            rgb_weight,sar_weight=vis_weight[0],vis_weight[1]   #[5 256 1]; [5 256 1]
            # print("####",rgb_weight.shape,sar_weight.shape)
            cols,rows=result['sta'][0],result['sta'][1]
            

        #     save_path=f'/scqian/Earthmind_proj/vis_qkv/{file_name.replace(".json", ".png")}'
        #     # 在 visualize_full_weights_on_image 里，拼接、平滑、逐像素归一化后已经有：
        #     # rgb_norm, sar_norm ∈ [0,1], base_np 为 H×W×3
        #     plot_img_and_heatmap(
        #     rgb_img_pil=rgb_img,
        #     sar_img_pil=img,                  # 你的 SAR PIL 图
        #     rgb_weight_blocks=rgb_weight,     # [blocks, 256, 1]
        #     sar_weight_blocks=sar_weight,     # [blocks, 256, 1]
        #     rows=rows, cols=cols,
        #     save_path=save_path,
        #     show=False,                       # 或 True
        #     cmap='viridis'                    # 或 'inferno' 等
        # )

            
       
            prediction = result['prediction'].replace("<|end|>", "")
            print(instruction)
            print(prediction)
            submit = {
                "image_id": file_name,
                "pred": prediction,
                "ground_truth": i["answer"],
                "type": i["task_type"]
            }
            all_submit.append(submit)

        # 保存结果
        base_name = os.path.splitext(os.path.basename(anno_file))[0]
        output_path = os.path.join(cfg.results_dir, f"{base_name}_result.json")
        with open(output_path, 'w') as json_file:
            json.dump(all_submit, json_file, indent=2)

        print(f"Finished processing {anno_file}, failed count: {len(false)}")
        if false:
            print("Failed images:", false)
