import logging
import os
from typing import Literal
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import copy
from .encode_fn import video_lisa_encode_fn, video_lisa_encode_fn_new
import json
import random
import pycocotools.mask as maskUtils
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os

class VideoCoTDataset(Dataset):
    """
    Video-based Chain-of-Thought Dataset for disaster assessment
    Supports interleaved <REGION> token insertion
    Simple version: directly resize images to 448x448 (like VideoReVOSDataset)
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    def __init__(self,
                 json_file,
                 image_folder,
                 extra_image_processor=None,
                 tokenizer=None,
                 template_map_fn=None,
                 max_length=8196,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',
                 preprocessor=None,
                 max_tokens_per_object=8,
                 ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.lazy = lazy
        self.max_length = max_length
        self.repeats = repeats
        self.image_folder = image_folder
        self.max_tokens_per_object = max_tokens_per_object

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        # Load JSON data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        print_log(f"Loaded {len(self.data)} samples from {json_file}", 
                  logger='current', level=logging.INFO)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        else:
            self.extra_image_processor = None

        self._system = ''

        # Image processing parameters
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336

        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        if self.arch_type == 'qwen':
            self.patch_token = 1

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

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        print(f"Building Video CoT dataset with {len(self.data)} items.")

    def __len__(self):
        return len(self.data) * self.repeats
    
    @property
    def modality_length(self):
        return [10000] * len(self.data)

    def real_len(self):
        return len(self.data)

    def decode_mask(self, mask_annos, image_size):
        """
        Decode RLE masks
        mask_annos: list of mask dicts with 'size' and 'counts'
        image_size: (height, width)
        """
        masks = []
        for mask_anno in mask_annos:
            if mask_anno is None:
                mask = np.zeros(image_size, dtype=np.uint8)
            else:
                mask = maskUtils.decode(mask_anno)
                if mask.ndim == 3:
                    mask = mask.sum(axis=2).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
            masks.append(mask)
        
        masks = np.stack(masks, axis=0)  # (num_masks, h, w)
        masks = torch.from_numpy(masks)
        return masks
    

    def _spatial_uniform_sample(
        self,
        token_indices,
        mask_resized,
        total_tokens_width,
        total_tokens_height,
        tokens_per_side,
        n_patch_cols,
        target_count
    ):
        """
        Spatially uniform sampling from selected tokens
        
        Strategy: divide mask region into grid, sample from each grid cell
        """
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

    def _get_visual_tokens_from_mask_simple(self, mask, ori_height, ori_width, 
                                            max_tokens_per_object=8):
        """
        Simple version: image is resized to 448x448, single patch
        """
        """
        Simple version for multi-image inference: 
        - Image is resized to 448x448
        - Single patch (no dynamic preprocessing)
        - For multi-image inference scenarios
        """
        tokens_per_side = int(np.sqrt(self.patch_token))
        
        # Step 1: Resize mask to 448x448 (same as image preprocessing)
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        mask_resized = np.array(
            Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), 
                Image.NEAREST
            )
        ) > 127
        
        # Step 2: Map mask to token grid
        mask_on_token_grid = np.array(
            Image.fromarray((mask_resized * 255).astype(np.uint8)).resize(
                (tokens_per_side, tokens_per_side),
                Image.NEAREST
            )
        ) > 127
        
        # Step 3: Find selected tokens
        selected_tokens = []
        for row in range(tokens_per_side):
            for col in range(tokens_per_side):
                if mask_on_token_grid[row, col]:
                    token_idx = row * tokens_per_side + col
                    selected_tokens.append(token_idx)

        # Step 4: Sample if too many tokens
        if len(selected_tokens) > max_tokens_per_object:
            selected_tokens = self._spatial_uniform_sample(
                selected_tokens,
                mask_on_token_grid,
                tokens_per_side,  # total_tokens_width
                tokens_per_side,  # total_tokens_height
                tokens_per_side,
                1,
                max_tokens_per_object
            )
        
        # Step 5: Ensure at least 1 token
        if len(selected_tokens) == 0:
            y_indices, x_indices = np.where(mask_on_token_grid)
            if len(y_indices) > 0:
                center_y = int(y_indices.mean())
                center_x = int(x_indices.mean())
                token_idx = center_y * tokens_per_side + center_x
                selected_tokens = [token_idx]
            else:
                selected_tokens = [0]
        
        return selected_tokens

    

    def extract_region_info_from_masks(self, masks, ori_heights, ori_widths):
        """
        Extract region info - one image per mask
        
        Args:
            masks: (num_objects, h, w) - should be 2 masks
            ori_heights: [pre_height, post_height]
            ori_widths: [pre_width, post_width]
        
        Returns:
            region_info: List of dicts - one entry per mask
        """
        num_objects = masks.shape[0]
        region_info = []
        
        num_pre_masks = num_objects // 2  # 1
        num_post_masks = num_objects - num_pre_masks  # 1
        
        tokens_per_image = self.patch_token
        pre_token_offset = 0
        post_token_offset = tokens_per_image
        
        # print(f"🔍 Extract region info (paired mode):")
        # print(f"   Total masks: {num_objects}")
        # print(f"   Pre masks: {num_pre_masks}, Post masks: {num_post_masks}")
        # print(f"   Tokens per image: {tokens_per_image}")
        # print(f"   Total tokens: {tokens_per_image * 2}")
        
        for obj_idx in range(num_pre_masks):
            mask = masks[obj_idx]
            
            pre_visual_token_indices = self._get_visual_tokens_from_mask_simple(
                mask, 
                ori_heights[0],
                ori_widths[0],
                self.max_tokens_per_object
            )
            
            region_info.append({
                'visual_token_indices': pre_visual_token_indices,
                'num_tokens': len(pre_visual_token_indices),
                'image_type': 'pre',
                'image_idx': 0,
            })
            
            # print(f"   Pre mask {obj_idx}: {len(pre_visual_token_indices)} tokens "
            #     f"(range: [{min(pre_visual_token_indices) if pre_visual_token_indices else 0}, "
            #     f"{max(pre_visual_token_indices) if pre_visual_token_indices else 0}])")
        
        for obj_idx in range(num_post_masks):
            mask = masks[num_pre_masks + obj_idx]
            
            post_visual_token_indices = self._get_visual_tokens_from_mask_simple(
                mask, 
                ori_heights[1],
                ori_widths[1],
                self.max_tokens_per_object
            )
            
            post_visual_token_indices_global = [
                idx + post_token_offset for idx in post_visual_token_indices
            ]
            
            region_info.append({
                'visual_token_indices': post_visual_token_indices_global,
                'num_tokens': len(post_visual_token_indices_global),
                'image_type': 'post',
                'image_idx': 1,
            })
            
            # print(f"   Post mask {obj_idx}: {len(post_visual_token_indices_global)} tokens "
            #     f"(range: [{min(post_visual_token_indices_global) if post_visual_token_indices_global else 0}, "
            #     f"{max(post_visual_token_indices_global) if post_visual_token_indices_global else 0}])")
        
        return region_info

    def insert_region_placeholders_dynamic(self, data_dict):
        """
        Dynamically insert <REGION> tokens into conversation output
        """
        if 'conversation' not in data_dict or 'region_info' not in data_dict:
            print(f"⚠️  Missing required fields")
            return data_dict
        
        conversation = data_dict['conversation']
        region_info = data_dict['region_info']
        
        for i, turn in enumerate(conversation):
            if 'output' not in turn:
                continue
            
            output = turn['output']
            
            # Split by [SEG]
            parts = output.split('[SEG]')
            
            if len(parts) - 1 != len(region_info):
                print(f"⚠️  [SEG] count mismatch: {len(parts)-1} vs {len(region_info)}")
                num_segs = min(len(parts) - 1, len(region_info))
            else:
                num_segs = len(region_info)
            
            # Reassemble with <REGION> tokens
            modified_output = parts[0]
            
            for obj_idx in range(num_segs):
                num_tokens = region_info[obj_idx]['num_tokens']
                region_str = '<REGION>' * num_tokens
                
                modified_output += f'[SEG]{region_str}{parts[obj_idx + 1]}'
            
            # Add remaining parts if any
            for remaining_idx in range(num_segs + 1, len(parts)):
                modified_output += f'[SEG]{parts[remaining_idx]}'
            
            data_dict['conversation'][i]['output'] = modified_output
        
        return data_dict

    def prepare_text(self, num_image_tokens_list):
        """
        Prepare conversation text for multiple images
        
        Args:
            num_image_tokens_list: List of token counts for each image
        """
        frame_tokens_list = []
        for num_tokens in num_image_tokens_list:
            frame_token_str = f'{self.IMG_START_TOKEN}' \
                            f'{self.IMG_CONTEXT_TOKEN * num_tokens}' \
                            f'{self.IMG_END_TOKEN}'
            frame_tokens_list.append(frame_token_str)
        
        # Join all frame tokens
        frame_tokens = '\n'.join(frame_tokens_list)
        
        return frame_tokens

    def __getitem__(self, index):
        index = index % self.real_len()
        sample = self.data[index]
        
        # Get image paths
        pre_image_path = os.path.join(self.image_folder, sample['pre_image'])
        post_image_path = os.path.join(self.image_folder, sample['post_image'])
        
        # Load and process images
        pixel_values_list = []
        extra_pixel_values = []
        ori_sizes = []
        
        for image_path in [pre_image_path, post_image_path]:
            image = Image.open(image_path).convert('RGB')
            ori_width, ori_height = image.size
            ori_sizes.append((ori_height, ori_width))
            
            if self.extra_image_processor is not None:
                g_image = np.array(image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)
            
            if self.preprocessor is not None:
                pixel_values_list.append(image)
            else:
                # ✅ Simple: directly resize and transform (like VideoReVOSDataset)
                image = self.transformer(image)  # (3, 448, 448)
                pixel_values_list.append(image)
        
        # Process images
        data_dict = {}
        num_video_tokens = None
        
        if self.preprocessor is not None:
            # qwen/llava preprocessing
            if self.arch_type == 'qwen':
                all_images = pixel_values_list
                _data_dict = self.preprocessor(all_images, do_resize=True, 
                                               size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                num_frames = _data_dict['image_grid_thw'].shape[0]
                num_video_tokens = num_frame_tokens * num_frames
            elif self.arch_type == 'llava':
                all_images = pixel_values_list
                _data_dict = self.preprocessor(all_images, do_resize=True, 
                                               size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
            data_dict.update(_data_dict)
        else:
            # ✅ InternVL: stack images (2, 3, 448, 448)
            all_pixel_values = torch.stack(pixel_values_list, dim=0)
            data_dict['pixel_values'] = all_pixel_values
        
        if self.extra_image_processor is not None:
            data_dict['g_pixel_values'] = extra_pixel_values
        
        # Process masks
        all_masks = []
        for mask_group in sample['masks']:
            for mask_anno in mask_group:
                all_masks.append(mask_anno)
        
        # Decode all masks
        masks = self.decode_mask(all_masks, image_size=ori_sizes[0])
        data_dict['masks'] = masks
        
        # Extract region info
        ori_heights = [ori_sizes[0][0], ori_sizes[1][0]]
        ori_widths = [ori_sizes[0][1], ori_sizes[1][1]]
        
        region_info = self.extract_region_info_from_masks(
            masks=masks,
            ori_heights=ori_heights,
            ori_widths=ori_widths
        )
        
        data_dict['region_info'] = region_info


        # vis_dir = './visualizations_video_cot'
        # os.makedirs(vis_dir, exist_ok=True)
        
        # pre_name = os.path.splitext(os.path.basename(sample['pre_image']))[0]
        # post_name = os.path.splitext(os.path.basename(sample['post_image']))[0]
        # save_name = f'{pre_name}_{post_name}_tokens.png'
        
        # self.visualize_region_tokens(
        #     pre_image_path=pre_image_path,
        #     post_image_path=post_image_path,
        #     masks=masks,
        #     region_info=region_info,
        #     ori_heights=ori_heights,
        #     ori_widths=ori_widths,
        #     save_path=os.path.join(vis_dir, save_name)
        # )
        
        # ✅ Validate token indices
        total_tokens = data_dict['pixel_values'].shape[0] * self.patch_token
        # print(f"✅ Validation:")
        # print(f"   pixel_values shape: {data_dict['pixel_values'].shape}")
        # print(f"   Total visual tokens: {total_tokens}")
        
        for i, info in enumerate(region_info):
            indices = info['visual_token_indices']
            if len(indices) > 0:
                min_idx, max_idx = min(indices), max(indices)
                if max_idx >= total_tokens:
                    print(f"   ❌ Object {i}: max index {max_idx} >= total {total_tokens}!")
                    raise ValueError(f"Token index out of range!")
                # else:
                    # print(f"   ✅ Object {i}: indices in valid range [0, {total_tokens-1}]")
        
        # Parse conversation
        conversation = []
        for conv in sample['conversations']:
            role = 'input' if conv['from'] == 'human' else 'output'
            value = conv['value']
            
            if role == 'input':
                # ✅ Fixed token count: 256 per image
                num_tokens_list = [self.patch_token, self.patch_token]
                frame_tokens = self.prepare_text(num_tokens_list)
                
                value = value.replace('<image>', frame_tokens)
                conversation.append({'input': value, 'system': self._system})
            else:
                conversation[-1]['output'] = value
        
        data_dict['conversation'] = conversation
        
        # Insert <REGION> placeholders
        data_dict = self.insert_region_placeholders_dynamic(data_dict)
        data_dict = self.preprocess_data(data_dict)

        # Handle qwen token replacement
        if num_video_tokens is not None:
            assert self.patch_token == 1
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, 
                                         self.IMG_CONTEXT_TOKEN * num_video_tokens)
            data_dict['conversation'][0]['input'] = input_str
        
        # Apply template
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        
        # Encode
        result = video_lisa_encode_fn_new(data_dict, tokenizer=self.tokenizer, 
                                         max_length=self.max_length,
                                         with_image_token=True)
        data_dict.update(result)
        # print(data_dict)
        data_dict['type'] = 'video'
        
        return data_dict
    
    def preprocess_data(self, data_dict):
        """Preprocess a single data sample"""
        custom_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in their mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>

        """
        
        conversation = data_dict.get('conversation', [])
        for conv_item in conversation:
            if 'input' in conv_item and '<img>' in conv_item['input']:
                conv_item['input'] = conv_item['input'].replace(
                    '<img>',
                    f'{custom_prompt}<img>',
                    1
                )
        return data_dict
    

    def visualize_region_tokens(
        self,
        pre_image_path,
        post_image_path,
        masks,
        region_info,
        ori_heights,
        ori_widths,
        save_path=None
    ):
        """
        mask
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from scipy import ndimage
        
        tokens_per_side = int(np.sqrt(self.patch_token))
        
        pre_image = Image.open(pre_image_path).convert('RGB')
        post_image = Image.open(post_image_path).convert('RGB')
        
        # Resize
        pre_resized = pre_image.resize((self.image_size, self.image_size))
        post_resized = post_image.resize((self.image_size, self.image_size))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax_pre = axes[0]
        ax_post = axes[1]
        
        ax_pre.imshow(pre_resized)
        ax_pre.set_title('Pre-disaster', fontsize=14, fontweight='bold', color='darkblue')
        ax_pre.axis('off')
        
        ax_post.imshow(post_resized)
        ax_post.set_title('Post-disaster', fontsize=14, fontweight='bold', color='darkred')
        ax_post.axis('off')
        
        token_grid_size = self.image_size / tokens_per_side
        for i in range(tokens_per_side + 1):
            pos = i * token_grid_size
            ax_pre.axhline(y=pos, color='white', linewidth=0.3, alpha=0.3)
            ax_pre.axvline(x=pos, color='white', linewidth=0.3, alpha=0.3)
            ax_post.axhline(y=pos, color='white', linewidth=0.3, alpha=0.3)
            ax_post.axvline(x=pos, color='white', linewidth=0.3, alpha=0.3)
        
        pre_token_count = 0
        post_token_count = 0
        
        for obj_idx, info in enumerate(region_info):
            mask = masks[obj_idx]
            image_type = info['image_type']
            
            # Resize mask
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            mask_resized = np.array(
                Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                    (self.image_size, self.image_size),
                    Image.NEAREST
                )
            ) > 127
            
            mask_edges = mask_resized ^ ndimage.binary_erosion(mask_resized, iterations=2)
            
            if image_type == 'pre':
                mask_overlay = np.zeros((self.image_size, self.image_size, 4))
                mask_overlay[mask_edges, :] = [0, 0, 1, 1.0]
                mask_overlay[mask_resized, :] = [0, 0, 1, 0.25]
                ax_pre.imshow(mask_overlay)
                
                indices = info['visual_token_indices']
                for token_idx in indices:
                    row = token_idx // tokens_per_side
                    col = token_idx % tokens_per_side
                    token_x = (col + 0.5) * token_grid_size
                    token_y = (row + 0.5) * token_grid_size
                    ax_pre.plot(token_x, token_y, 'go', markersize=6, alpha=0.8,
                            markeredgecolor='white', markeredgewidth=0.5)
                
                pre_token_count += len(indices)
                
            else:  # post
                mask_overlay = np.zeros((self.image_size, self.image_size, 4))
                mask_overlay[mask_edges, :] = [1, 0, 0, 1.0]
                mask_overlay[mask_resized, :] = [1, 0, 0, 0.25]
                ax_post.imshow(mask_overlay)
                
                post_token_offset = self.patch_token
                indices = info['visual_token_indices']
                for token_idx in indices:
                    relative_idx = token_idx - post_token_offset
                    row = relative_idx // tokens_per_side
                    col = relative_idx % tokens_per_side
                    token_x = (col + 0.5) * token_grid_size
                    token_y = (row + 0.5) * token_grid_size
                    ax_post.plot(token_x, token_y, 'go', markersize=6, alpha=0.8,
                            markeredgecolor='white', markeredgewidth=0.5)
                
                post_token_count += len(indices)
        
        info_text_pre = f"Selected tokens: {pre_token_count}"
        ax_pre.text(0.02, 0.98, info_text_pre, transform=ax_pre.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        info_text_post = f"Selected tokens: {post_token_count}"
        ax_post.text(0.02, 0.98, info_text_post, transform=ax_post.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        fig.suptitle(
            f'Visual Token Sampling for Disaster Assessment\n'
            f'Image size: {self.image_size}x{self.image_size}, '
            f'Tokens per image: {self.patch_token} ({tokens_per_side}x{tokens_per_side})',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        legend_elements = [
            mpatches.Patch(facecolor='blue', alpha=0.3, edgecolor='blue', label='Pre-disaster mask'),
            mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red', label='Post-disaster mask'),
            plt.Line2D([0], [0], marker='o', color='w', label='Selected tokens',
                    markerfacecolor='g', markeredgecolor='white', markersize=8),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                bbox_to_anchor=(0.5, 0.02), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
