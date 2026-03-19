import json
import os
import random

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask
import numpy as np
import copy

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess

from .grand_process import glamm_grand_map_fn

class GranDDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self,
                 image_folder,
                 json_folder=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 image_list_save_path='./work_dirs/grand_image.json',
                 json_list_save_path='./work_dirs/grand_jsons.json',
    ):
        super().__init__()
        assert lazy
        self.lazy = lazy
        self.max_length = max_length

        self.image_list_save_path = image_list_save_path
        self.json_list_save_path = json_list_save_path

        json_files, image_path_dict = self.json_file_preprocess(image_folder, json_folder)
        self.json_data = json_files
        self.image_path_dict = image_path_dict

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

    def json_file_preprocess(self, image_folder, json_folder):

        # list jsons
        print("Processing GRAND json files !!!")
        if os.path.exists(self.json_list_save_path):
            with open(self.json_list_save_path, 'r') as f:
                json_files = json.load(f)
        else:
            json_files = os.listdir(json_folder)
            _json_files = []
            for _file in json_files:
                if '.json' in _file:
                    _json_files.append(os.path.join(json_folder, _file))
            json_files = _json_files
            with open(self.json_list_save_path, 'w') as f:
                json.dump(json_files, f)
        print(f"Finished, {len(json_files)} json files !")

        # list images
        print("Processing GRAND image files !!!")
        if os.path.exists(self.image_list_save_path):
            with open(self.image_list_save_path, 'r') as f:
                image_path_dict = json.load(f)
        else:
            sub_folders = os.listdir(image_folder)
            _sub_folders = []
            for folder_name in sub_folders:
                if 'sa_00' in folder_name:
                    _sub_folders.append(folder_name)
            sub_folders = _sub_folders
            sub_folders = [os.path.join(image_folder, folder_name) for folder_name in sub_folders]

            image_path_dict = {}
            for sub_folder in sub_folders:
                files = os.listdir(sub_folder)
                for _file in files:
                    if '.jpg' in _file:
                        image_path_dict[_file] = os.path.join(sub_folder, _file)

            with open(self.image_list_save_path, 'w') as f:
                json.dump(image_path_dict, f)
        print(f"Finished, {len(image_path_dict)} image files !")

        return json_files, image_path_dict

    @property
    def modality_length(self):
        length_list = [10000] * len(self.json_data)
        return length_list * self.repeats

    def __len__(self):
        return len(self.json_data) * self.repeats

    def real_len(self):
        return len(self.json_data)

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                m = mask.decode(seg)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_grand_map_fn(data_dict)
        return data_dict

    def replace_image_str(self, data_dict, image_str):
        data_dict['conversation'][0]['input'] = \
            data_dict['conversation'][0]['input'].replace(DEFAULT_IMAGE_TOKEN, image_str)
        return data_dict

    def __getitem__(self, index):

        index = index % self.real_len()
        json_file_path = self.json_data[index]
        with open(json_file_path, 'r') as f:
            json_dict = json.load(f)

        image_name = list(json_dict.keys())[0]

        if image_name not in self.image_path_dict.keys():
            return self.__getitem__(random.randint(0, len(self.json_data) - 1))
        image_path = self.image_path_dict[image_name]

        json_dict = json_dict[image_name]
        # parse datasets
        result = self.dataset_map_fn(json_dict)
        json_dict.update(result)
        data_dict = json_dict

        data_dict['image'] = image_path

        # process image
        image_file = data_dict['image']
        try:
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self.json_data) - 1))
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

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length,
                                      with_image_token=True)
        data_dict.update(result)
        # process mask
        data_dict['masks'] = self.decode_mask(data_dict['masks'], ori_height=ori_height, ori_width=ori_width)

        if data_dict['masks'] is None:
            return self.__getitem__(random.randint(0, len(self.json_data) - 1))

        return data_dict