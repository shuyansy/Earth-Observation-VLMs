"""
SAR+RGB paired dataset for ChatEarthNet.
Loads both S2 (RGB) and S1 (SAR VH+VV) images, concatenates their visual tokens,
and feeds them together to the LLM.

Based on VideoCoTDataset which already handles multi-image inputs.
"""
import logging
import os
import json
import copy
from typing import Literal

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from mmengine import print_log
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import pycocotools.mask as maskUtils

from .encode_fn import video_lisa_encode_fn_new
from .gcg_process import glamm_grounded_cot_map_fn


class SARRGBDataset(Dataset):
    """
    SAR+RGB paired dataset.
    Each sample loads an RGB (S2) image and a SAR (S1 VH+VV→RGB) image,
    processes them both, and concatenates their visual tokens for the LLM.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    def __init__(self,
                 rgb_image_folder,
                 sar_image_folder,
                 data_path,
                 tokenizer,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 max_tokens_per_object=8,
                 ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.lazy = lazy
        self.max_length = max_length
        self.repeats = repeats
        self.rgb_image_folder = rgb_image_folder
        self.sar_image_folder = sar_image_folder
        self.max_tokens_per_object = max_tokens_per_object

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        # Load JSON data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print_log(f"Loaded {len(self.data)} SAR+RGB samples from {data_path}",
                  logger='current', level=logging.INFO)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        else:
            self.extra_image_processor = None

        self._system = ''

        # Image processing parameters (same as VideoCoTDataset - simple resize)
        self.downsample_ratio = 0.5
        self.image_size = 448
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        print(f"Building SAR+RGB dataset with {len(self.data)} items.")

    def __len__(self):
        return len(self.data) * self.repeats

    @property
    def modality_length(self):
        return [10000] * len(self.data)

    def real_len(self):
        return len(self.data)

    def _load_sar_as_rgb(self, image_name):
        """Load SAR VH and VV channels and convert to 3-channel RGB image."""
        vh_path = os.path.join(self.sar_image_folder, 'vh', image_name)
        vv_path = os.path.join(self.sar_image_folder, 'vv', image_name)

        vh = np.array(Image.open(vh_path).convert('L'))  # (H, W)
        vv = np.array(Image.open(vv_path).convert('L'))  # (H, W)

        # Stack VH, VV, and average as 3 channels
        avg = ((vh.astype(np.float32) + vv.astype(np.float32)) / 2).astype(np.uint8)
        sar_rgb = np.stack([vh, vv, avg], axis=-1)  # (H, W, 3)
        return Image.fromarray(sar_rgb, mode='RGB')

    def decode_mask(self, mask_annos, ori_height, ori_width):
        """Decode RLE masks."""
        binary_masks = []
        for object_mask in mask_annos:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for rle in object_mask:
                m = maskUtils.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        return torch.from_numpy(masks)

    def _get_visual_tokens_from_mask_simple(self, mask, ori_height, ori_width,
                                            max_tokens_per_object=8):
        """Simple version: image is resized to 448x448, single patch."""
        tokens_per_side = int(np.sqrt(self.patch_token))

        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        mask_resized = np.array(
            Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
        ) > 127

        mask_on_token_grid = np.array(
            Image.fromarray((mask_resized * 255).astype(np.uint8)).resize(
                (tokens_per_side, tokens_per_side), Image.NEAREST
            )
        ) > 127

        selected_tokens = []
        for row in range(tokens_per_side):
            for col in range(tokens_per_side):
                if mask_on_token_grid[row, col]:
                    token_idx = row * tokens_per_side + col
                    selected_tokens.append(token_idx)

        if len(selected_tokens) > max_tokens_per_object:
            selected_tokens = self._spatial_uniform_sample(
                selected_tokens, mask_on_token_grid,
                tokens_per_side, tokens_per_side,
                tokens_per_side, 1, max_tokens_per_object
            )

        if len(selected_tokens) == 0:
            y_indices, x_indices = np.where(mask_on_token_grid)
            if len(y_indices) > 0:
                center_y = int(y_indices.mean())
                center_x = int(x_indices.mean())
                selected_tokens = [center_y * tokens_per_side + center_x]
            else:
                selected_tokens = [0]

        return selected_tokens

    def _spatial_uniform_sample(self, token_indices, mask_resized,
                                total_tokens_width, total_tokens_height,
                                tokens_per_side, n_patch_cols, target_count):
        """Spatial uniform sampling from selected tokens."""
        token_positions = []
        for token_idx in token_indices:
            patch_idx = token_idx // (tokens_per_side * tokens_per_side)
            token_in_patch = token_idx % (tokens_per_side * tokens_per_side)
            patch_row = patch_idx // n_patch_cols
            patch_col = patch_idx % n_patch_cols
            local_row = token_in_patch // tokens_per_side
            local_col = token_in_patch % tokens_per_side
            global_row = patch_row * tokens_per_side + local_row
            global_col = patch_col * tokens_per_side + local_col
            token_positions.append((global_row, global_col, token_idx))

        mask_rows, mask_cols = np.where(mask_resized)
        if len(mask_rows) == 0:
            return token_indices[:target_count]

        min_row, max_row = mask_rows.min(), mask_rows.max()
        min_col, max_col = mask_cols.min(), mask_cols.max()

        grid_size = int(np.ceil(np.sqrt(target_count)))
        row_step = max(1, (max_row - min_row + 1) / grid_size)
        col_step = max(1, (max_col - min_col + 1) / grid_size)

        sampled = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell_row_min = min_row + i * row_step
                cell_row_max = min_row + (i + 1) * row_step
                cell_col_min = min_col + j * col_step
                cell_col_max = min_col + (j + 1) * col_step
                cell_tokens = [
                    (r, c, idx) for r, c, idx in token_positions
                    if cell_row_min <= r < cell_row_max and cell_col_min <= c < cell_col_max
                ]
                if cell_tokens:
                    cell_center_r = (cell_row_min + cell_row_max) / 2
                    cell_center_c = (cell_col_min + cell_col_max) / 2
                    closest = min(cell_tokens,
                                  key=lambda x: (x[0] - cell_center_r)**2 + (x[1] - cell_center_c)**2)
                    sampled.append(closest[2])
                if len(sampled) >= target_count:
                    break
            if len(sampled) >= target_count:
                break
        return sampled[:target_count]

    def extract_region_info_from_masks(self, masks, ori_height, ori_width):
        """
        Extract region info from both RGB and SAR images for each mask.

        Since RGB and SAR are spatially aligned, the same mask applies to both.
        For each spatial position we provide paired (RGB, SAR) indices.
        At model forward time, the query-similarity selection picks the better one.

        Token layout in vit_embeds_flat:
          - RGB token indices: 0 ~ patch_token-1
          - SAR token indices: patch_token ~ 2*patch_token-1
        """
        num_objects = masks.shape[0]
        region_info = []
        sar_token_offset = self.patch_token

        for obj_idx in range(num_objects):
            mask = masks[obj_idx]

            # Get spatial positions from the mask (local indices 0~patch_token-1)
            local_indices = self._get_visual_tokens_from_mask_simple(
                mask, ori_height, ori_width, self.max_tokens_per_object
            )

            rgb_indices = local_indices
            sar_indices = [idx + sar_token_offset for idx in local_indices]

            region_info.append({
                # paired indices for query-similarity selection in model forward
                'rgb_token_indices': rgb_indices,
                'sar_token_indices': sar_indices,
                'num_tokens': len(local_indices),
                'fusion_mode': 'query_select',  # flag for model to use selection logic
            })

        return region_info

    def prepare_text(self, num_image_tokens_list):
        """Prepare image token strings for multiple images."""
        frame_tokens_list = []
        for num_tokens in num_image_tokens_list:
            frame_token_str = (f'{self.IMG_START_TOKEN}'
                               f'{self.IMG_CONTEXT_TOKEN * num_tokens}'
                               f'{self.IMG_END_TOKEN}')
            frame_tokens_list.append(frame_token_str)
        return '\n'.join(frame_tokens_list)

    def insert_region_placeholders_dynamic(self, data_dict):
        """Insert <REGION> tokens into conversation output."""
        if 'conversation' not in data_dict or 'region_info' not in data_dict:
            return data_dict

        conversation = data_dict['conversation']
        region_info = data_dict['region_info']

        for i, turn in enumerate(conversation):
            if 'output' not in turn:
                continue

            output = turn['output']
            parts = output.split('[SEG]')

            num_segs = min(len(parts) - 1, len(region_info))

            modified_output = parts[0]
            for obj_idx in range(num_segs):
                num_tokens = region_info[obj_idx]['num_tokens']
                region_str = '<REGION>' * num_tokens
                modified_output += f'[SEG]{region_str}{parts[obj_idx + 1]}'

            for remaining_idx in range(num_segs + 1, len(parts)):
                modified_output += f'[SEG]{parts[remaining_idx]}'

            data_dict['conversation'][i]['output'] = modified_output

        return data_dict

    def preprocess_data(self, data_dict):
        """Add system prompt."""
        custom_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in their mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>

        """
        conversation = data_dict.get('conversation', [])
        for conv_item in conversation:
            if 'input' in conv_item and '<img>' in conv_item['input']:
                conv_item['input'] = conv_item['input'].replace(
                    '<img>', f'{custom_prompt}<img>',
                    1
                )
        return data_dict

    def __getitem__(self, index):
        index = index % self.real_len()
        sample = self.data[index]

        image_name = sample['image']

        # Load RGB image
        rgb_path = os.path.join(self.rgb_image_folder, image_name)
        rgb_image = Image.open(rgb_path).convert('RGB')
        ori_width, ori_height = rgb_image.size

        # Load SAR image (VH+VV -> 3-channel)
        sar_image = self._load_sar_as_rgb(image_name)

        # Process both images for vision encoder
        pixel_values_list = []
        for image in [rgb_image, sar_image]:
            pv = self.transformer(image)  # (3, 448, 448)
            pixel_values_list.append(pv)

        data_dict = {}
        # Stack: (2, 3, 448, 448) - two images for visual token extraction
        data_dict['pixel_values'] = torch.stack(pixel_values_list, dim=0)

        # g_pixel_values: only RGB for SAM2 grounding decoder (masks are on RGB)
        # SAR does not participate in mask prediction, only in visual token selection
        if self.extra_image_processor is not None:
            g_image = np.array(rgb_image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            data_dict['g_pixel_values'] = g_pixel_values

        # Process masks (from RGB image)
        masks = self.decode_mask(sample['masks'], ori_height, ori_width)
        if masks is None:
            return self.__getitem__(0)
        data_dict['masks'] = masks

        # Extract region info (masks correspond to RGB)
        region_info = self.extract_region_info_from_masks(
            masks, ori_height, ori_width
        )
        data_dict['region_info'] = region_info

        # Build conversation with two image token blocks
        # <img>RGB_TOKENS</img>\n<img>SAR_TOKENS</img>
        num_tokens_list = [self.patch_token, self.patch_token]
        frame_tokens = self.prepare_text(num_tokens_list)

        # Parse conversation from annotation
        conversation = []
        for conv in sample['conversations']:
            role = 'input' if conv['from'] == 'human' else 'output'
            value = conv['value']

            if role == 'input':
                # Replace <image> with both RGB and SAR token blocks
                value = value.replace('<image>', frame_tokens)
                conversation.append({'input': value, 'system': self._system})
            else:
                conversation[-1]['output'] = value

        data_dict['conversation'] = conversation

        # Insert <REGION> placeholders
        data_dict = self.insert_region_placeholders_dynamic(data_dict)
        data_dict = self.preprocess_data(data_dict)

        # Apply template
        result = self.template_map_fn(data_dict)
        data_dict.update(result)

        # Encode
        result = video_lisa_encode_fn_new(
            data_dict, tokenizer=self.tokenizer,
            max_length=self.max_length, with_image_token=True
        )
        data_dict.update(result)
        data_dict['type'] = 'video'  # multi-image type for collate_fn

        return data_dict
