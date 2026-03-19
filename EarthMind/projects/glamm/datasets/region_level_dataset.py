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

from projects.glamm.datasets.utils.utils import ANSWER_LIST, REGION_QUESTIONS
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class RegionDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 repeats=1,
                 num_classes_per_sample=3,
                 extra_image_processor=None):
        super().__init__()

        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.question_templates = REGION_QUESTIONS

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

        self.text_data = self._load_annotations(data_path, image_folder)
        self.image_folder = image_folder

        self.image_processor = BUILDER.build(image_processor)
        size = self.image_processor.crop_size

        if isinstance(size, dict):
            self.image_w, self.image_h = size['width'], size['height']
        elif isinstance(size, int):
            self.image_h, self.image_w = size, size
        else:
            self.image_w, self.image_h = size

        self.pad_image_to_square = pad_image_to_square
        self.repeats = repeats

    def _load_annotations(self, data_path, image_folder=None):
        self.coco = COCO(data_path)
        img_ids = self.coco.getImgIds()
        data_infos = []
        for img_id in img_ids:
            info = self.coco.loadImgs([img_id])[0]
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            if min(info['height'], info['width']) < 32:
                continue
            data_infos.append(info)
        return data_infos

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = 100
            length_list.append(cur_len)
        return length_list * self.repeats

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    def region_processor(self, orig_size, post_size, bboxes, labels):
        orig_h, orig_w = orig_size
        post_h, post_w = post_size
        y_scale = post_h / orig_h
        x_scale = post_w / orig_w
        shuffle_ids = torch.randperm(len(labels))[:self.num_classes_per_sample]
        selected_bboxes = bboxes[shuffle_ids]

        # Ensure selected_bboxes is two-dimensional
        if len(selected_bboxes.shape) == 1:
            selected_bboxes = np.expand_dims(selected_bboxes, axis=0)

        selected_labels = [labels[i] for i in shuffle_ids]
        selected_bboxes[:, [0, 2]] *= x_scale
        selected_bboxes[:, [1, 3]] *= y_scale
        selected_bboxes = torch.tensor(
            selected_bboxes, dtype=torch.float32) / post_h
        return selected_bboxes, selected_labels

    def _parse_annotations(self, img_info):
        data_dict = {}
        bboxes, captions = [], []
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
        image_path = os.path.join(self.image_folder, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(
                g_image).permute(2, 0, 1).contiguous()
            data_dict['g_pixel_values'] = g_pixel_values

        orig_w, orig_h = image.size
        if self.pad_image_to_square:
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        post_h, post_w = image.shape[1:3]
        data_dict['pixel_values'] = image

        for ann in ann_info:
            if ann.get('ignore', False) or ann['area'] <= 0 or ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, orig_w) - max(x1, 0))
            inter_h = max(0, min(y1 + h, orig_h) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if bbox:
                bboxes.append(bbox)
                captions.append(img_info['caption'])

        if len(bboxes) == 0:
            return self.__getitem__(0)

        bboxes = np.array(bboxes, dtype=np.float32)
        seg_map = img_info['file_name'].replace('jpg', 'png')
        bboxes, captions = self.region_processor((orig_h, orig_w), (post_h, post_w), bboxes, captions)

        data_dict['bboxes'] = bboxes
        data_dict['captions'] = captions
        data_dict['seg_map'] = seg_map
        return data_dict

    def create_conversation(self, captions):
        questions = []
        answers = []
        for i, label in enumerate(captions):
            question = random.choice(self.question_templates).strip().replace('<region>', f'region{i + 1} <bbox>')
            questions.append(question)
            answers.append(label)

        conversation = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conversation.append({'input': question, 'output': answer})
        return conversation

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = {}
        ann_info = copy.deepcopy(self.text_data[index])
        ann_info = self._parse_annotations(ann_info)

        data_dict['g_pixel_values'] = ann_info.pop('g_pixel_values', None)
        data_dict['pixel_values'] = ann_info.pop('pixel_values')
        data_dict['bboxes'] = ann_info.pop('bboxes', None)

        conversation = self.create_conversation(ann_info['captions'])
        data_dict['conversation'] = conversation

        result = self.template_map_fn(data_dict)
        data_dict.update(result)

        result = encode_fn(data_dict, tokenizer=self.tokenizer,
                           max_length=self.max_length, with_image_token=True)
        data_dict.update(result)

        return data_dict

class RefCocoGRegionDataset(RegionDataset):
    pass

class VisualGenomeRegionDataset(RegionDataset):
    def _parse_annotations(self, img_info):
        data_dict = {}
        bboxes, captions = [], []
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
        image_path = os.path.join(self.image_folder, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(
                g_image).permute(2, 0, 1).contiguous()
            data_dict['g_pixel_values'] = g_pixel_values

        orig_w, orig_h = image.size
        if self.pad_image_to_square:
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        post_h, post_w = image.shape[1:3]
        data_dict['pixel_values'] = image

        for ann in ann_info:
            if ann.get('ignore', False) or ann['area'] <= 0 or ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, orig_w) - max(x1, 0))
            inter_h = max(0, min(y1 + h, orig_h) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if bbox:
                bboxes.append(bbox)
                captions.append(ann['caption'].strip())

        if len(bboxes) == 0:
            return self.__getitem__(0)

        bboxes = np.array(bboxes, dtype=np.float32)
        seg_map = img_info['file_name'].replace('jpg', 'png')
        bboxes, captions = self.region_processor((orig_h, orig_w), (post_h, post_w), bboxes, captions)

        data_dict['bboxes'] = bboxes
        data_dict['captions'] = captions
        data_dict['seg_map'] = seg_map
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
    dataset = VisualGenomeRegionDataset(
        image_folder='./data/visual_genome/images',
        image_processor=image_processor,
        data_path='data/visual_genome/train.json',
        tokenizer=tokenizer,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        max_length=2048,
        pad_image_to_square=False,
        repeats=1,
        num_classes_per_sample=3,
        extra_image_processor=None)

    for i in range(1000):
        print(dataset[i])
