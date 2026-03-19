import copy
import random
import glob
import json
import logging
import os
from typing import Literal
from transformers import AutoTokenizer
import torch

from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from xtuner.dataset.utils import encode_fn
from xtuner.dataset.map_fns import llava_map_fn

from projects.glamm.datasets.utils.utils import expand2square

from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from .utils import dynamic_preprocess


import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image




import os, re
import numpy as np
from PIL import Image

_BAND_RE = re.compile(r"_B(\d{2}|8A)\b", re.IGNORECASE)

def _read_band(fp):
    arr = np.array(Image.open(fp))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def _resize_to(arr, H, W):
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

def ms_folder_to_rgb_and_extra_groups(
    image_folder: str,
    to_uint8: bool = True,
    pad_mode: str = "repeat"  # "repeat" 或 "zero"：最后一组不满 3 通道的填充策略
):
    """
    返回若干 PIL.Image（RGB 模式）：
      [0] = 标准 RGB：B04(红), B03(绿), B02(蓝)
      [1..] = 其余通道按 3 个一组组成的“伪 RGB”
    """
    tifs = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(".tif")]
    if not tifs:
        raise ValueError(f"No images found in folder: {image_folder}")

    band_to_file = {}
    for fp in tifs:
        m = _BAND_RE.search(os.path.basename(fp))
        if m:
            band_to_file["B" + m.group(1).upper()] = fp

    # 必须有 B02,B03,B04
    for b in ("B02", "B03", "B04"):
        if b not in band_to_file:
            raise RuntimeError(f"Missing required band {b} in {image_folder}")

    # 读取三波段 + 对齐尺寸
    r = _read_band(band_to_file["B04"])
    g = _read_band(band_to_file["B03"])
    b = _read_band(band_to_file["B02"])
    H = max(r.shape[0], g.shape[0], b.shape[0])
    W = max(r.shape[1], g.shape[1], b.shape[1])
    r = _resize_to(r, H, W); g = _resize_to(g, H, W); b = _resize_to(b, H, W)

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

    # 其余通道顺序（按 Sentinel-2 常见 12 波段，去掉 02,03,04）
    rest = ["B05","B06","B07","B8A","B11","B12","B01","B09"]
    rest_files = [band_to_file[b] for b in rest if b in band_to_file]

    # 读并对齐到同一尺寸
    rest_arrs = []
    for fp in rest_files:
        a = _read_band(fp)
        a = _resize_to(a, H, W)
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



# 1) 读取一个 sample 目录下的所有 .tif → [C,H,W]
def read_tif_stack(image_folder: str) -> torch.Tensor:
    files = sorted(
        f for f in os.listdir(image_folder) if f.lower().endswith(".tif")
    )
    if not files:
        raise ValueError(f"No .tif found: {image_folder}")
    arrs = []
    H0 = W0 = None
    for f in files:
        a = np.array(Image.open(os.path.join(image_folder, f)))
        if a.ndim == 3:  # 偶尔会读到 HxWx3，取第0通道
            a = a[..., 0]
        if H0 is None:
            H0, W0 = a.shape
        elif a.shape != (H0, W0):
            # 如需强制对齐到第一张的分辨率
            a = np.array(Image.fromarray(a).resize((W0, H0), resample=Image.BILINEAR))
        arrs.append(a.astype(np.float32))
    x = np.stack(arrs, axis=0)  # [C,H,W]
    return torch.from_numpy(x)

# 2) 选择最接近宽高比的网格 (cols, rows)，与你原来的 target_ratios 一致
def find_closest_grid(aspect: float, min_num=1, max_num=6):
    cands = [(i, j) for n in range(min_num, max_num+1)
             for i in range(1, n+1) for j in range(1, n+1)
             if 1 <= i*j <= max_num and i*j >= min_num]
    # 与原逻辑一致：先按块数排序，再挑 i/j 最接近当前宽高比
    cands = sorted(cands, key=lambda x: x[0]*x[1])
    best = min(cands, key=lambda ij: abs((ij[0]/ij[1]) - aspect))
    return best  # (cols, rows)

# 3) 纯张量版“动态切块”：输入 [C,H,W]，输出 [M,C,S,S]（S=image_size）
def dynamic_tile_tensor(chw: torch.Tensor,
                        image_size: int = 448,
                        min_num: int = 1,
                        max_num: int = 6,
                        use_thumbnail: bool = False):
    assert chw.ndim == 3, f"expect [C,H,W], got {chw.shape}"
    C, H, W = chw.shape
    aspect = W / H
    cols, rows = find_closest_grid(aspect, min_num, max_num)   # 与你原函数一致的网格选择
    target_w = cols * image_size
    target_h = rows * image_size

    # 先整体 resize 到规则尺寸（与原函数先 resize 再均匀切块一致）
    x = chw.unsqueeze(0)            # [1,C,H,W]
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)  # [1,C,H',W']
    x = x.squeeze(0)                # [C,H',W']

    # 均匀切成 cols*rows 个 tile
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * image_size
            x0 = c * image_size
            crop = x[:, y0:y0+image_size, x0:x0+image_size]    # [C,S,S]
            tiles.append(crop)
    out = torch.stack(tiles, dim=0)  # [M,C,S,S], M=cols*rows

    # 可选 thumbnail：按原逻辑在多块时追加一张整图缩略图
    if use_thumbnail and out.shape[0] != 1:
        thumb = F.interpolate(chw.unsqueeze(0), size=(image_size, image_size),
                              mode="bilinear", align_corners=False).squeeze(0)  # [C,S,S]
        out = torch.cat([out, thumb.unsqueeze(0)], dim=0)  # [M+1,C,S,S]
    return out, (cols, rows)

# 4) （可选）按通道归一化：强烈建议用“全数据集统计”的 mean/std
def normalize_per_channel(batched: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    # batched: [M,C,S,S], mean/std: [C]
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (batched - mean[None, :, None, None]) / std[None, :, None, None]


class InfinityMMDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 max_length=8192,
                 offline_save_path='./work_dirs/infinityMM.json',
                 ):
        self.offline_save_path = offline_save_path
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self._system = ''

        self.template = prompt_template
        self.max_length = max_length

        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        self.data = self._load_annotations(data_path)
        self._max_refetch = 1000

    def _load_annotations(self, data_path):
        if os.path.exists(self.offline_save_path):
            with open(self.offline_save_path, 'r') as f:
                ret = json.load(f)
            print(f"Load InfinityMM file list from {self.offline_save_path}, {len(ret)} items !!!")
            return ret
        sub_folders = []
        for sub_folder in os.listdir(data_path):
            if '.' not in sub_folder:
                # a folder
                if "LVIS_111k" in sub_folder:
                    # special case, have subsub folder
                    subsub_folders = os.listdir(os.path.join(data_path, sub_folder))
                    for subsub_folder in subsub_folders:
                        sub_folders.append(os.path.join(data_path, sub_folder, subsub_folder))
                else:
                    sub_folders.append(os.path.join(data_path, sub_folder))

        all_jsons = []
        for sub_folder in sub_folders:
            print(f"Processing {sub_folder} !!!")
            _files = os.listdir(sub_folder)
            _num = 0
            for _file in _files:
                if '.json' in _file:
                    _json_path = os.path.join(sub_folder, _file)
                    _num += 1
                    all_jsons.append(os.path.join(sub_folder, _file))
            print(f"Finished {sub_folder} has {_num} items.")

        with open(self.offline_save_path, 'w') as f:
            json.dump(all_jsons, f)

        return all_jsons

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def prepare_data(self, index):
        data_path = self.data[index]

        with open(data_path, 'r') as f:
            data_dict = json.load(f)
        if 'image' in data_dict.keys():
            data_dict['image'] = data_path.replace('.json', '.jpg')

        if data_dict is None:
            return None

        out_data_dict = {}

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None

            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            out_data_dict['pixel_values'] = pixel_values

            num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(
                1, 3, self.image_size, self.image_size)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for i, msg in enumerate(conversations):
            if msg['from'] == 'human':

                # change to 1 image
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>\n', '').replace('<image>', '')
                    if i == 0:
                        msg['value'] = "<image>\n" + msg['value']

                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        return {'input_ids': input_ids, 'labels': labels}



class MS_LLaVADataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 skip_pure_text=False,
                 ):

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = image_folder
        self.template = prompt_template
        self.max_length = max_length

        self._system = ''

        self.arch_type = arch_type
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))


        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)

        self.data = self._load_annotations(data_path, image_folder)
        self._max_refetch = 1000

        self.skip_pure_text = skip_pure_text

    def _load_annotations(self, data_path, image_folder=None):
        data = json.load(open(data_path))
        return data

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length
    
    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        
        if data_dict is None:
            return None
        
        out_data_dict = {}

        if self.skip_pure_text and data_dict.get('image', None) is None:
            return None

      

        if data_dict.get('image', None) is not None:
            image_folder = os.path.join(self.image_folder, data_dict['image'])

           
            # # 获取所有符合要求的图片（按文件名排序）
            # image_files = sorted([
            #     os.path.join(image_folder, f)
            #     for f in os.listdir(image_folder)
            #     if f.lower().endswith('.tif') 
            # ])

            # if not image_files:
            #     raise ValueError(f"No images found in folder: {image_folder}")
    
            # image = ms_folder_to_rgb_pil(image_folder, to_uint8=True)

            # images = dynamic_preprocess(image, self.min_dynamic_patch,
            #                                 self.max_dynamic_patch,
            #                                 self.image_size, self.use_thumbnail)
            # pixel_values = [self.transformer(image) for image in images]
            # pixel_values = torch.stack(pixel_values)
            # out_data_dict['pixel_values'] = pixel_values

            # 从 12 个 .tif 中得到：1 张标准 RGB + 若干组伪 RGB
            rgb_like_images = ms_folder_to_rgb_and_extra_groups(image_folder, to_uint8=True, pad_mode="repeat")

            pixel_values_list = []
            for img_rgb_like in rgb_like_images:
                sub_images = dynamic_preprocess(
                    img_rgb_like,
                    self.min_dynamic_patch,
                    self.max_dynamic_patch,
                    self.image_size,
                    self.use_thumbnail
                )
                pixel_values_list.extend([self.transformer(si) for si in sub_images])

            pixel_values = torch.stack(pixel_values_list)  # 形状：[M_total, 3, 448, 448]（示例）
            out_data_dict['pixel_values'] = pixel_values
            # print("######", pixel_values.shape)


    
            num_image_tokens = pixel_values.shape[0] * self.patch_token

            image_token_str = f'{self.IMG_START_TOKEN}' \
                            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                            f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(
                1, 3, self.image_size, self.image_size)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        return {'input_ids': input_ids, 'labels': labels}



class Multi_LLaVADataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 data_root=None,
                 data_prefix=dict(
                        rgb_path='rgb/',
                        sar_path='sar/'
                    ),
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 skip_pure_text=False,
                 ):

        self.tokenizer = BUILDER.build(tokenizer)
#         print("🔥 tokenizer config:", tokenizer)
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             tokenizer['pretrained_model_name_or_path'],
#             trust_remote_code=tokenizer.get('trust_remote_code', False),
#             padding_side=tokenizer.get('padding_side', 'right')
# )
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.data_root=data_root
        # self.image_folder = data_root
        self.rgb_folder = os.path.join(self.data_root, data_prefix.get('rgb_path', ''))
        self.sar_folder = os.path.join(self.data_root, data_prefix.get('sar_path', ''))



        self.template = prompt_template
        self.max_length = max_length

        self._system = ''

        self.arch_type = arch_type
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))


        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)

        self.data = self._load_annotations(data_path)
        self._max_refetch = 1000

        self.skip_pure_text = skip_pure_text

    def _load_annotations(self, data_path):
        data = json.load(open(data_path))
        return data

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length
    
    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        
        if data_dict is None:
            return None
        
        out_data_dict = {}

        if self.skip_pure_text and data_dict.get('image', None) is None:
            return None

        if data_dict.get('image', None) is not None:
            # image_file = os.path.join(self.image_folder, data_dict['image'])

            rgb_image_file = os.path.join(self.rgb_folder, data_dict['image'])
            sar_image_file = os.path.join(self.sar_folder, data_dict['image'])


            try:
                # image = Image.open(image_file).convert('RGB')
                rgb_image = Image.open(rgb_image_file).convert('RGB')
                sar_image = Image.open(sar_image_file).convert('RGB')
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None
            if self.preprocessor is not None:
                print("@@@@@@@@@@@@@@@")
                # images = dynamic_preprocess(image, self.min_dynamic_patch,
                #                             self.max_dynamic_patch,
                #                             self.image_size, self.use_thumbnail)
                # images = [image]
                rgb_images = [rgb_image]
                sar_images = [sar_image]    
                if self.arch_type == 'qwen':
                    _sar_data_dict = self.preprocessor(sar_images, do_resize=True)
                    _sar_data_dict['pixel_values'] = torch.tensor(_sar_data_dict['pixel_values'], dtype=torch.float)
                    _sar_data_dict['image_grid_thw'] = torch.tensor(_sar_data_dict['image_grid_thw'], dtype=torch.int)
                    num_image_tokens = int(_sar_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    out_data_dict['pixel_values'] = _sar_data_dict['pixel_values']

                    _rgb_data_dict = self.preprocessor(rgb_images, do_resize=True)
                    _rgb_data_dict['pixel_values'] = torch.tensor(_rgb_data_dict['pixel_values'], dtype=torch.float)
                    out_data_dict['rgb_pixel_values'] = _rgb_data_dict['pixel_values']

                elif self.arch_type == 'llava':
                    _sar_data_dict = self.preprocessor(sar_images, do_resize=True, size=(self.image_size, self.image_size))
                    _sar_data_dict['pixel_values'] = np.stack(_sar_data_dict['pixel_values'], axis=0)
                    _sar_data_dict['pixel_values'] = torch.tensor(_sar_data_dict['pixel_values'], dtype=torch.float)
                    num_image_tokens = _sar_data_dict['pixel_values'].shape[0] * self.patch_token
                    out_data_dict['pixel_values'] = _sar_data_dict['pixel_values']

                    _rgb_data_dict = self.preprocessor(rgb_images, do_resize=True, size=(self.image_size, self.image_size))
                    _rgb_data_dict['pixel_values'] = np.stack(_rgb_data_dict['pixel_values'], axis=0)
                    _rgb_data_dict['pixel_values'] = torch.tensor(_rgb_data_dict['pixel_values'], dtype=torch.float)
                    out_data_dict['rgb_pixel_values'] = _rgb_data_dict['pixel_values']
                else:
                    raise NotImplementedError

                out_data_dict.update(_data_dict)
            else:
                # images = dynamic_preprocess(image, self.min_dynamic_patch,
                #                             self.max_dynamic_patch,
                #                             self.image_size, self.use_thumbnail)
                rgb_images = dynamic_preprocess(rgb_image, self.min_dynamic_patch,
                                                self.max_dynamic_patch,
                                                self.image_size, self.use_thumbnail)
                sar_images = dynamic_preprocess(sar_image, self.min_dynamic_patch,
                                                self.max_dynamic_patch,
                                                self.image_size, self.use_thumbnail)


               
                pixel_values = [self.transformer(image) for image in sar_images]
                pixel_values = torch.stack(pixel_values)
                out_data_dict['pixel_values'] = pixel_values

                rgb_pixel_values = [self.transformer(image) for image in rgb_images]
                rgb_pixel_values = torch.stack(rgb_pixel_values)
                out_data_dict['rgb_pixel_values'] = rgb_pixel_values

                num_image_tokens = pixel_values.shape[0] * self.patch_token


            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(
                1, 3, self.image_size, self.image_size)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        # print("#########",out_conversation)
        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        return {'input_ids': input_ids, 'labels': labels}




class LLaVADataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 skip_pure_text=False,
                 ):

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = image_folder
        self.template = prompt_template
        self.max_length = max_length

        self._system = ''

        self.arch_type = arch_type
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))


        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)

        self.data = self._load_annotations(data_path, image_folder)
        self._max_refetch = 1000

        self.skip_pure_text = skip_pure_text

    def _load_annotations(self, data_path, image_folder=None):
        data = json.load(open(data_path))
        return data

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length
    
    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        
        if data_dict is None:
            return None
        
        out_data_dict = {}

        if self.skip_pure_text and data_dict.get('image', None) is None:
            return None

        if data_dict.get('image', None) is not None:
            image_file = os.path.join(self.image_folder, data_dict['image'])
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None
            if self.preprocessor is not None:
                # images = dynamic_preprocess(image, self.min_dynamic_patch,
                #                             self.max_dynamic_patch,
                #                             self.image_size, self.use_thumbnail)
                images = [image]
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(images, do_resize=True)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
                else:
                    raise NotImplementedError
                out_data_dict.update(_data_dict)
            else:
                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)
                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                out_data_dict['pixel_values'] = pixel_values

                num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(
                1, 3, self.image_size, self.image_size)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        return {'input_ids': input_ids, 'labels': labels}


if __name__ == '__main__':
    from transformers import CLIPImageProcessor, AutoTokenizer
    from third_parts.segment_anything.utils.transforms import ResizeLongestSide
    pretrained_model = 'MBZUAI/GLaMM-GranD-Pretrained'
    llm_name_or_path = 'lmsys/vicuna-7b-v1.5'

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path)
    image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path='openai/clip-vit-large-patch14-336')
    extra_image_processor = dict(
        type=ResizeLongestSide,
        target_length=1024,
    )
    from xtuner.utils.templates import PROMPT_TEMPLATE
    prompt_template = PROMPT_TEMPLATE.vicuna
    from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, template_map_fn
    from projects.glamm.datasets.collate_fns.glamm_collate_fn import glamm_collate_fn

    dataset = LLaVADataset(
        tokenizer=tokenizer,
        data_path='data/llava_data/LLaVA-Instruct-150K/llava_instruct_150k.json',
        prompt_template=prompt_template,
        special_tokens=['[SEG]'],
        image_folder='data/coco/train2017/',
    )
    for i in range(1000):
        dataset[i]
