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

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess

from .gcg_process import glamm_openpsg_map_fn, glamm_flickr_map_fn, glamm_granf_map_fn, glamm_refcocog_map_fn

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

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length,
                                      with_image_token=True)
        data_dict.update(result)
        # process mask
        data_dict['masks'] = self.decode_mask(data_dict['masks'], ori_height=ori_height, ori_width=ori_width)

        if data_dict['masks'] is None:
            return self.__getitem__(0)

        return data_dict

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
        data_dict = glamm_granf_map_fn(data_dict)
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