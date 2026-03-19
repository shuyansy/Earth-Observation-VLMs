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

from projects.glamm.datasets.utils.utils import GCG_QUESTIONS, ANSWER_LIST
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
class GCGDataset(Dataset):
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
        self.question_templates = GCG_QUESTIONS
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

        self.text_data = self.json_file_preprocess(data_path, image_folder)
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

    def json_file_preprocess(self, data_path, image_folder=None):
        with open(data_path, 'r') as f:
            json_data = json.load(f)
        return json_data

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

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image) # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            ann_info['g_pixel_values'] = g_pixel_values

        width, height = image.size
        if self.pad_image_to_square:
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        ann_info['pixel_values'] = image

        caption = ann_info['caption'].strip('"').strip()
        masks, phrases, tokens_positive = [], [], []
        for word, grounding in ann_info["groundings"].items():
            phrases.append(word)
            tokens_positive.append(grounding["token_positives"])

            # Convert segmentation to binary mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for rle in grounding["rle_masks"]:
                m = mask_utils.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask)

        def sort_by_start_index(items, order):
            return [items[i] for i in order]
        
        phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
        masks = sort_by_start_index(masks, phrase_order)
        phrases = sort_by_start_index(phrases, phrase_order)
        tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

        ann_info.update({
            'image_path': image_path,
            'caption': caption,
            'masks': masks,
            'phrases': phrases,
            'tokens_positive': tokens_positive,
        })
        return ann_info

    def create_conversation(self, caption, tokens_positive):
        question = random.choice(self.question_templates).strip()

        # Prepare caption with tags
        def tag_caption(caption, tokens):
            for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
                caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
            return caption

        detailed_answer = tag_caption(caption, tokens_positive)

        question = 'The <image> provides an overview of the picture.\n' + question
        conversation = [{'input': question, 'output': detailed_answer}]
        return conversation
    
    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = {}
        ann_info = copy.deepcopy(self.text_data[index])
        ann_info = self._parse_annotations(ann_info)
        
        data_dict['g_pixel_values'] = ann_info.pop('g_pixel_values')
        data_dict['pixel_values'] = ann_info.pop('pixel_values')
        if len(ann_info['masks']) == 0:
            return self.__getitem__(0)
        data_dict['masks'] = torch.from_numpy(np.stack(ann_info['masks'], axis=0))

        conversation = self.create_conversation(ann_info['caption'], ann_info['tokens_positive'])
        data_dict['conversation'] = conversation

        result = self.template_map_fn(data_dict)
        data_dict.update(result)

        result = encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length, with_image_token=True)
        data_dict.update(result)

        return data_dict

class GranDfGCGDataset(GCGDataset):
    pass
class RefCOCOgGCGDataset(GCGDataset):
    def json_file_preprocess(self, data_path, image_folder=None):
        with open(data_path, 'r') as f:
            json_data = json.load(f)
        return [list(line.values())[0] for line in json_data]

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['img_file_name'])
        image = Image.open(image_path).convert('RGB')
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image) # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            ann_info['g_pixel_values'] = g_pixel_values

        width, height = image.size
        if self.pad_image_to_square:
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        ann_info['pixel_values'] = image

        caption = ann_info['caption'].strip('"').strip().lower()
        masks, phrases, tokens_positive = [], [], []
        for detail in ann_info['refs']:
            phrase = detail['sentence']
            if phrase.lower() in caption:
                phrases.append(phrase)
                index = caption.find(phrase)
                end_index = index + len(phrase) if index != -1 else -1
                tokens_positive.append([index, end_index])

                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in detail["segmentation"]:
                    rles = mask_utils.frPyObjects([seg], height, width)
                    m = mask_utils.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                masks.append(binary_mask)

        def sort_by_start_index(items, order):
            return [items[i] for i in order]
        
        phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
        masks = sort_by_start_index(masks, phrase_order)
        phrases = sort_by_start_index(phrases, phrase_order)
        tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

        ann_info.update({
            'image_path': image_path,
            'caption': caption,
            'masks': masks,
            'phrases': phrases,
            'tokens_positive': tokens_positive,
        })
        return ann_info

class OpenPsgGCGDataset(GCGDataset):
    pass

class Flickr30kGCGDataset(GCGDataset):

    def json_file_preprocess(self, data_path, image_folder=None):
        def filter_images(data_infos, min_size):
            return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]
        
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

        return data_infos
    
    def _parse_annotations(self, img_info):
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        ann_info = self.coco.loadAnns(ann_ids)
        
        annotations = {'phrases': [], 'caption': img_info['caption'], 'masks': [], 'tokens_positive': []}
        image_path = os.path.join(self.image_folder, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image) # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            annotations['g_pixel_values'] = g_pixel_values

        width, height = image.size
        if self.pad_image_to_square:
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        annotations['pixel_values'] = image

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0 or ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            tokens_positive = ann['tokens_positive']
            phrase = [img_info['caption'][span[0]:span[1]] for span in tokens_positive]
            annotations['phrases'].append(phrase[0])
            annotations['tokens_positive'].append(tokens_positive[0])

            rle = ann['sam_mask']
            mask_decoded = mask_utils.decode(rle).astype(np.uint8)
            annotations['masks'].append(mask_decoded)

        def sort_by_start_index(items, order):
            return [items[i] for i in order]
        
        phrase_order = sorted(range(len(annotations['tokens_positive'])), key=lambda x: annotations['tokens_positive'][x][0])
        annotations['masks'] = sort_by_start_index(annotations['masks'], phrase_order)
        annotations['phrases'] = sort_by_start_index(annotations['phrases'], phrase_order)
        annotations['tokens_positive'] = sort_by_start_index(annotations['tokens_positive'], phrase_order)

        return annotations

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
    dataset = Flickr30kGCGDataset(
        image_folder='data/flickr30k/flickr30k-images/',
        image_processor=image_processor,
        data_path='./data/GranDf/annotations/train/flickr_mergedGT_GCG_train.json',
        tokenizer=tokenizer,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        max_length=2048,
        pad_image_to_square=True,
        repeats=1,
        num_classes_per_sample=3,
        extra_image_processor=extra_image_processor)
    
    for i in range(1000):
        print(dataset[i])