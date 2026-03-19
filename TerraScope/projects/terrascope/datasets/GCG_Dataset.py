import json
import os
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn,video_lisa_encode_fn_new
from .utils import dynamic_preprocess

from .gcg_process import glamm_openpsg_map_fn, glamm_flickr_map_fn, glamm_granf_map_fn, glamm_refcocog_map_fn,glamm_grounded_cot_map_fn



class GCGDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
    ):
        super().__init__()
        assert lazy
        self.lazy = lazy
        self.max_length = max_length

        json_data = self.json_file_preprocess(data_path)
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self.image_folder = image_folder

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.repeats = repeats

        self._system = ''

        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
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

        self.single_image_mode = single_image_mode

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            if self.lazy:
                cur_len = 100
            else:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
            length_list.append(cur_len)
        return length_list * self.repeats

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask.frPyObjects([seg], ori_height, ori_width)
                m = mask.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_refcocog_map_fn(data_dict)
        return data_dict

    def replace_image_str(self, data_dict, image_str):
        data_dict['conversation'][0]['input'] = \
            data_dict['conversation'][0]['input'].replace(DEFAULT_IMAGE_TOKEN, image_str)
        return data_dict


    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        # parse datasets
        result = self.dataset_map_fn(data_dict)
        data_dict.update(result)

        # process image
        image_file = data_dict['image']
        image = Image.open(os.path.join(self.image_folder,
                                        image_file)).convert('RGB')
        ori_width, ori_height = image.size
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            data_dict['g_pixel_values'] = g_pixel_values

        if self.single_image_mode:
            images = [image]
        else:
            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
        pixel_values = [self.transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

        num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = f'{self.IMG_START_TOKEN}' \
                        f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{self.IMG_END_TOKEN}'

        data_dict = self.replace_image_str(data_dict, image_token_str)
        
              # process mask
        data_dict['masks'] = self.decode_mask(data_dict['masks'], 
                                            ori_height=ori_height, 
                                            ori_width=ori_width)
                                
        if data_dict['masks'] is None:
            return self.__getitem__(0)
        
        region_info = self.extract_region_info_from_masks(
            masks=data_dict['masks'],
            pixel_values=pixel_values,
            ori_height=ori_height,
            ori_width=ori_width,
            max_tokens_per_object=8
        )
        # print(region_info[0]["num_tokens"])
        data_dict['region_info'] = region_info

        data_dict = self.insert_region_placeholders_dynamic(data_dict)
        # ===================================================
        data_dict = self.preprocess_data(data_dict)
        result = self.template_map_fn(data_dict)
        # print("result", result)
        data_dict.update(result)

  
        result = video_lisa_encode_fn_new(data_dict, tokenizer=self.tokenizer, 
                                    max_length=self.max_length,
                                    with_image_token=True)
        # print("1111", result)

        data_dict.update(result)
        

        # ###########visualization################
        # image_path = os.path.join(self.image_folder, image_file)
        # image_name = os.path.splitext(image_file)[0]
        # vis_dir = './visualizations_rs'
        # os.makedirs(vis_dir, exist_ok=True)
        # # self.visualize_all_patches_and_tokens(
        # #     image_path=image_path,
        # #     pixel_values=pixel_values,
        # #     ori_height=ori_height,
        # #     ori_width=ori_width,
        # #     save_path=os.path.join(vis_dir, f'{image_name}_patches.png')
        # # )
        
        # self.visualize_region_tokens(
        #     image_path=image_path,
        #     masks=data_dict['masks'],
        #     pixel_values=pixel_values,
        #     region_info=region_info,
        #     ori_height=ori_height,
        #     ori_width=ori_width,
        #     save_path=os.path.join(vis_dir, f'{image_name}_objects.png')
        # )

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
                    f'{custom_prompt}<img>'
                )
        return data_dict


    def _get_patch_layout_accurate(self, image_size_tuple, ori_height, ori_width, min_num=1, max_num=6, image_size=448):
        """
        InternVLdynamic_preprocesspatch layout
        
        Args:
            image_size_tuple: (num_patches,) num_patches
            ori_height, ori_width: 
            min_num, max_num, image_size: dynamic_preprocess
        
        Returns:
            patch_layout: list of dictpatch**resize**
            target_aspect_ratio: (rows, cols) grid
            resized_width, resized_height: resize
        """
        aspect_ratio = ori_width / ori_height
        
        target_ratios = {(i, j)
                        for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num}
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = ori_width * ori_height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        target_aspect_ratio = best_ratio
        
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        patch_layout = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            patch_layout.append({
                'patch_idx': i,
                'x_start': box[0],
                'x_end': box[2],
                'y_start': box[1],
                'y_end': box[3],
            })
        
        return patch_layout, target_aspect_ratio, target_width, target_height
    


    def _get_visual_tokens_from_mask_accurate(
        self,
        mask, 
        ori_height, 
        ori_width,
        pixel_values,
        max_tokens_per_object=64
    ):
        """
        InternVL
        
        1. resizetarget_width x target_height
        2. blocks
        3. maskresize
        """
        tokens_per_patch = self.patch_token
        tokens_per_side = int(np.sqrt(tokens_per_patch))
        
        patch_layout, target_aspect_ratio, target_width, target_height = \
            self._get_patch_layout_accurate(
                pixel_values.shape[0], 
                ori_height, 
                ori_width,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size
            )
        
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        
        mask_resized_to_target = np.array(
            Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (target_width, target_height), 
                Image.NEAREST
            )
        ) > 127
        
        n_cols = target_aspect_ratio[0]
        n_rows = target_aspect_ratio[1]
        
        total_tokens_width = n_cols * tokens_per_side
        total_tokens_height = n_rows * tokens_per_side
        
        mask_on_token_grid = np.array(
            Image.fromarray((mask_resized_to_target * 255).astype(np.uint8)).resize(
                (total_tokens_width, total_tokens_height),
                Image.NEAREST
            )
        ) > 127
        
        selected_tokens = []
        
        for patch_info in patch_layout:
            patch_idx = patch_info['patch_idx']
            
            patch_col = patch_idx % n_cols
            patch_row = patch_idx // n_cols
            
            token_col_start = patch_col * tokens_per_side
            token_col_end = (patch_col + 1) * tokens_per_side
            token_row_start = patch_row * tokens_per_side
            token_row_end = (patch_row + 1) * tokens_per_side
            
            patch_start_token = patch_idx * tokens_per_patch
            
            for local_row in range(tokens_per_side):
                for local_col in range(tokens_per_side):
                    global_token_row = token_row_start + local_row
                    global_token_col = token_col_start + local_col
                    
                    if (global_token_row < total_tokens_height and 
                        global_token_col < total_tokens_width and
                        mask_on_token_grid[global_token_row, global_token_col]):
                        
                        token_idx_in_patch = local_row * tokens_per_side + local_col
                        global_token_idx = patch_start_token + token_idx_in_patch
                        selected_tokens.append(global_token_idx)
        
        if len(selected_tokens) > max_tokens_per_object:
            selected_tokens = self._spatial_uniform_sample(
                selected_tokens,
                mask_on_token_grid,
                total_tokens_width,
                total_tokens_height,
                tokens_per_side,
                n_cols,
                max_tokens_per_object
            )
        
        if len(selected_tokens) == 0:
            y_indices, x_indices = np.where(mask_on_token_grid)
            if len(y_indices) > 0:
                center_y = int(y_indices.mean())
                center_x = int(x_indices.mean())
                
                patch_row = center_y // tokens_per_side
                patch_col = center_x // tokens_per_side
                patch_idx = patch_row * n_cols + patch_col
                
                if patch_idx < len(patch_layout):
                    local_row = center_y % tokens_per_side
                    local_col = center_x % tokens_per_side
                    token_idx = patch_idx * tokens_per_patch + local_row * tokens_per_side + local_col
                    selected_tokens = [token_idx]
                else:
                    selected_tokens = [0]
            else:
                selected_tokens = [0]
        
        return selected_tokens





    def extract_region_info_from_masks(
        self, 
        masks, 
        pixel_values,
        ori_height, 
        ori_width,
        max_tokens_per_object=64
    ):
        """
        region info - InternVL
        """
        num_objects = masks.shape[0]
        region_info = []
        
        for obj_idx in range(num_objects):
            mask = masks[obj_idx]
            
            visual_token_indices = self._get_visual_tokens_from_mask_accurate(
                mask, 
                ori_height, 
                ori_width,
                pixel_values,
                max_tokens_per_object
            )
            
            region_info.append({
                'visual_token_indices': visual_token_indices,
                'num_tokens': len(visual_token_indices),
            })
        
        return region_info
    

    

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
    
    

    
    def insert_region_placeholders_dynamic(self, data_dict):
        """
        <REGION> - 
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
            
            parts = output.split('[SEG]')
            
            if len(parts) - 1 != len(region_info):
                print(f"⚠️  [SEG] count mismatch: {len(parts)-1} vs {len(region_info)}")
                num_segs = min(len(parts) - 1, len(region_info))
            else:
                num_segs = len(region_info)
            
            modified_output = parts[0]
            
            for obj_idx in range(num_segs):
                num_tokens = region_info[obj_idx]['num_tokens']
                region_str = '<REGION>' * num_tokens
                
                modified_output += f'[SEG]{region_str}{parts[obj_idx + 1]}'
            
            for remaining_idx in range(num_segs + 1, len(parts)):
                modified_output += f'[SEG]{parts[remaining_idx]}'
            
            data_dict['conversation'][i]['output'] = modified_output
            
        
        return data_dict
    


    def visualize_region_tokens(
        self,
        image_path,
        masks,
        pixel_values,
        region_info,
        ori_height,
        ori_width,
        labels=None,
        save_path=None
    ):
        """
        objectmaskvisual tokens
        InternVL
        """
        num_objects = len(region_info)
        num_patches = pixel_values.shape[0]
        tokens_per_patch = self.patch_token
        tokens_per_side = int(np.sqrt(tokens_per_patch))
        
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.basename(image_path)
        
        patch_layout, target_aspect_ratio, target_width, target_height = \
            self._get_patch_layout_accurate(
                num_patches, 
                ori_height, 
                ori_width,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size
            )
        
        n_cols = target_aspect_ratio[0]
        n_rows = target_aspect_ratio[1]
        
        resized_image = image.resize((target_width, target_height))
        
        vis_n_cols = 3
        vis_n_rows = (num_objects + vis_n_cols - 1) // vis_n_cols
        
        fig, axes = plt.subplots(vis_n_rows, vis_n_cols, figsize=(20, 7 * vis_n_rows))
        if vis_n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for obj_idx in range(num_objects):
            ax = axes[obj_idx]
            
            ax.imshow(resized_image)
            
            if labels is not None and obj_idx < len(labels):
                title = f'Object {obj_idx}: {labels[obj_idx]}\n{region_info[obj_idx]["num_tokens"]} tokens'
            else:
                title = f'Object {obj_idx}: {region_info[obj_idx]["num_tokens"]} tokens'
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            mask = masks[obj_idx].cpu().numpy()
            mask_resized = np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (target_width, target_height), 
                    Image.NEAREST
                )
            ) > 127
            
            from scipy import ndimage
            mask_edges = mask_resized ^ ndimage.binary_erosion(mask_resized, iterations=2)
            
            mask_overlay = np.zeros((target_height, target_width, 4))
            mask_overlay[mask_edges, :] = [1, 0, 0, 1.0]
            mask_overlay[mask_resized, :] = [1, 0, 0, 0.2]
            ax.imshow(mask_overlay)
            
            for patch_info in patch_layout:
                rect = patches.Rectangle(
                    (patch_info['x_start'], patch_info['y_start']),
                    patch_info['x_end'] - patch_info['x_start'],
                    patch_info['y_end'] - patch_info['y_start'],
                    linewidth=2, edgecolor='gray', facecolor='none',
                    linestyle='--', alpha=0.5
                )
                ax.add_patch(rect)
                
                ax.text(
                    patch_info['x_start'] + 20,
                    patch_info['y_start'] + 30,
                    f"P{patch_info['patch_idx']}",
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7)
                )
            
            visual_indices = region_info[obj_idx]['visual_token_indices']
            
            for token_idx in visual_indices:
                patch_idx = token_idx // tokens_per_patch
                token_idx_in_patch = token_idx % tokens_per_patch
                
                token_row = token_idx_in_patch // tokens_per_side
                token_col = token_idx_in_patch % tokens_per_side
                
                patch_col = patch_idx % n_cols
                patch_row = patch_idx // n_cols
                
                patch_x_start = patch_col * self.image_size
                patch_y_start = patch_row * self.image_size
                
                token_x = patch_x_start + (token_col + 0.5) * self.image_size / tokens_per_side
                token_y = patch_y_start + (token_row + 0.5) * self.image_size / tokens_per_side
                
                ax.plot(token_x, token_y, 'go', markersize=4, alpha=0.6)
            
            info_text = (
                f"Tokens: {region_info[obj_idx]['num_tokens']}\n"
                f"Indices: {visual_indices[:3]}..."
            )
            ax.text(
                0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
            )
        
        for idx in range(num_objects, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(
            f'Visual Tokens for {image_name}\n'
            f'Original: {ori_width}x{ori_height} → Resized: {target_width}x{target_height} → '
            f'Grid: {n_cols}x{n_rows} ({num_patches} patches)',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


    def visualize_all_patches_and_tokens(
        self,
        image_path,
        pixel_values,
        ori_height,
        ori_width,
        save_path=None
    ):
        """
        patchtoken grid
        InternVL
        """
        num_patches = pixel_values.shape[0]
        tokens_per_patch = self.patch_token
        tokens_per_side = int(np.sqrt(tokens_per_patch))
        
        image = Image.open(image_path).convert('RGB')
        
        patch_layout, target_aspect_ratio, target_width, target_height = \
            self._get_patch_layout_accurate(
                num_patches, 
                ori_height, 
                ori_width,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size
            )
        
        n_cols = target_aspect_ratio[0]
        n_rows = target_aspect_ratio[1]
        
        resized_image = image.resize((target_width, target_height))
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        ax.imshow(resized_image)
        ax.set_title(
            f'Image: {ori_width}x{ori_height} → Resized: {target_width}x{target_height}\n'
            f'Grid: {n_cols}x{n_rows} = {num_patches} patches, {tokens_per_patch} tokens/patch ({tokens_per_side}×{tokens_per_side})',
            fontsize=14, fontweight='bold'
        )
        ax.axis('off')
        
        colors = plt.cm.rainbow(np.linspace(0, 1, num_patches))
        
        for patch_idx, patch_info in enumerate(patch_layout):
            rect = patches.Rectangle(
                (patch_info['x_start'], patch_info['y_start']),
                patch_info['x_end'] - patch_info['x_start'],
                patch_info['y_end'] - patch_info['y_start'],
                linewidth=3, edgecolor=colors[patch_idx], facecolor='none'
            )
            ax.add_patch(rect)
            
            center_x = (patch_info['x_start'] + patch_info['x_end']) / 2
            center_y = (patch_info['y_start'] + patch_info['y_end']) / 2
            
            patch_col = patch_idx % n_cols
            patch_row = patch_idx // n_cols
            
            ax.text(
                center_x, center_y,
                f"Patch {patch_idx}\n"
                f"Grid: ({patch_row}, {patch_col})\n"
                f"{tokens_per_patch} tokens\n"
                f"({tokens_per_side}×{tokens_per_side})",
                color='white', fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=colors[patch_idx], alpha=0.7)
            )
            
            step = 4
            for i in range(0, tokens_per_side + 1, step):
                x = patch_info['x_start'] + i * self.image_size / tokens_per_side
                ax.plot([x, x], [patch_info['y_start'], patch_info['y_end']], 
                    'w-', linewidth=0.5, alpha=0.3)
                y = patch_info['y_start'] + i * self.image_size / tokens_per_side
                ax.plot([patch_info['x_start'], patch_info['x_end']], [y, y],
                    'w-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Patch visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class RefCOCOgGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
        )

    def json_file_preprocess(self, data_path):
        json_data = json.load(open(data_path))

        # convert {id: dict} to dict(..., id=xx)
        for idx in range(len(json_data)):
            id = list(json_data[idx].keys())[0]
            json_data[idx] = json_data[idx][id]
            json_data[idx].update({'id': id})
        return json_data

class GranDfGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
        )

    def dataset_map_fn(self, data_dict):
        # data_dict = glamm_granf_map_fn(data_dict)
        data_dict= glamm_grounded_cot_map_fn(data_dict)
        return data_dict

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)

            for rle in object_mask:
                m = mask.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

class OpenPsgGCGDataset(GranDfGCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
        )
    def dataset_map_fn(self, data_dict):
        data_dict = glamm_openpsg_map_fn(data_dict)
        return data_dict


class FlickrGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
        )

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_flickr_map_fn(data_dict)
        return data_dict

    def json_file_preprocess(self, data_path):
        def filter_images(data_infos, min_size):
            return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]

        # convert {id: dict} to dict(..., id=xx)
        from pycocotools.coco import COCO
        self.coco = COCO(data_path)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        for img_id in self.image_ids:
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{data_path}'!"
        print(f'Removed {removed_img_count} images.')
        data_infos = [data_infos[i] for i in filter_images(data_infos, min_size=32)]

        # obtain_annotations
        for data_info in data_infos:
            ann_ids = self.coco.getAnnIds(imgIds=data_info['id'])
            ann_info = self.coco.loadAnns(ann_ids)
            data_info.update({'ann_info': ann_info})
        return data_infos

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = mask.decode(object_mask).astype(np.uint8)
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks