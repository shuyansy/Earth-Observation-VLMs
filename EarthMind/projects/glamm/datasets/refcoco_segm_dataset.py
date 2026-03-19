import copy
import random
import glob
import json
import logging
import os
import torch

from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER

from xtuner.dataset.utils import encode_fn
from xtuner.dataset.map_fns import llava_map_fn

from projects.glamm.datasets.utils.utils import expand2square

from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from third_parts.mmdet.datasets.refcoco import RefCocoDataset


class ReferSegmDataset(RefCocoDataset):
    def __init__(self,
                 data_root,
                 ann_file=None,
                 split_file=None,
                 image_processor=None,
                 extra_image_processor=None,
                 data_prefix=dict(img_path='train2014/'),
                 tokenizer=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 num_classes_per_sample=3):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file=split_file,
        )
        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""

        self.question_templates = SEG_QUESTIONS
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.num_classes_per_sample = num_classes_per_sample
        self.tokenizer = BUILDER.build(tokenizer)

        self.tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
        reg_tokens = ['<bbox>', '<point>']
        segmentation_tokens = ['[SEG]']
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.max_length = max_length
        self.template_map_fn = BUILDER.build(template_map_fn)

        self.image_processor = BUILDER.build(image_processor)
        size = self.image_processor.crop_size
        if isinstance(size, dict):
            self.image_w, self.image_h = size['width'], size['height']
        self.pad_image_to_square = pad_image_to_square

    @property
    def modality_length(self):
        import pickle
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        # for idx in range(len(self)):
        #     if self.serialize_data:
        #         start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
        #         end_addr = self.data_address[idx].item()
        #         bytes = memoryview(
        #             self.data_bytes[start_addr:end_addr])  # type: ignore
        #         data_dict = pickle.loads(bytes) 
        #     else:
        #         data_dict = copy.deepcopy(self.data_list[idx])
        return length_list

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = Image.open(image_path).convert('RGB')
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(
                g_image).permute(2, 0, 1).contiguous()
            ann_info['g_pixel_values'] = g_pixel_values

        width, height = image.size
        if self.pad_image_to_square:
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        ann_info['pixel_values'] = image

        masks, phrases = [], []
        instances, text = ann_info['instances'], ann_info['text']
        index = np.random.choice(range(len(instances)), min(
            len(instances), self.num_classes_per_sample))
        for idx in index:
            inst = instances[idx]
            phrase = text[idx].lower()
            phrases.append(phrase)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in inst["mask"]:
                rles = mask_utils.frPyObjects([seg], height, width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask)

        ann_info.update({
            'masks': masks,
            'phrases': phrases,
        })
        return ann_info

    def __getitem__(self, idx):
        data_dict = {}
        ann_info = super().__getitem__(idx)
        ann_info = self._parse_annotations(ann_info)

        data_dict['g_pixel_values'] = ann_info.pop('g_pixel_values')
        data_dict['pixel_values'] = ann_info.pop('pixel_values')
        if len(ann_info['masks']) == 0:
            return self.__getitem__(0)
        data_dict['masks'] = torch.from_numpy(
            np.stack(ann_info['masks'], axis=0))

        conversation = []
        for i, phrase in enumerate(ann_info['phrases']):
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            conversation.append(
                {'input': question, 'output': random.choice(ANSWER_LIST)})

        data_dict['conversation'] = conversation
        result = self.template_map_fn(data_dict)
        data_dict.update(result)

        result = encode_fn(data_dict, tokenizer=self.tokenizer,
                           max_length=self.max_length, with_image_token=True)
        data_dict.update(result)

        return data_dict

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

    dataset = ReferSegmDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        extra_image_processor=extra_image_processor,
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        ann_file='refcoco+/instances.json',
        split_file='refcoco+/refs(unc).p',
    )
    for i in range(1000):
        dataset[i]
