import json
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
import numpy as np
import copy

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess

import random

import torch.nn.functional as F

class OspreyDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    LIMIT = ''

    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'

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
        self.text_data = json_data

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
        self.patch_size = patch_size
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

    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = self.annToMask(object_mask, ori_height, ori_width)
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def _process_conversation(self, converations, n_regions, region_pixels):
        start_region_str = '<image> There are {} part regions in the picture: '.format(n_regions)
        for i in range(n_regions):
            start_region_str = start_region_str + \
                               f"region{i+1}" + self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN * region_pixels[i] + self.VP_END_TOKEN
            if i == n_regions - 1:
                start_region_str = start_region_str + '.\n'
            else:
                start_region_str = start_region_str + ', '

        for i, item in enumerate(converations):
            item['value'] = item['value'].replace('<', '').replace('>', '')
            if item['from'] == 'human':
                item['value'] = item['value'] + self.LIMIT
            # first conv process
            if i == 0:
                assert item['from'] == "human"
                item['value'] =  start_region_str + item['value']

        messages = converations
        input = ''

        conversation = []
        while messages and messages[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            messages = messages[1:]
        for msg in messages:
            if msg['from'] == 'human':
                if DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                        '').strip()
                    msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                    msg['value'] = msg['value'].strip()
                input += msg['value']

            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        return conversation

    def _get_region_infos(self, masks):
        # masks tensor, (n_obj, h, w)
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                  int(self.image_size // self.patch_size * self.downsample_ratio)),
            mode='nearest').squeeze(0)
        region_pixels = []
        for mask in masks:
            region_pixels.append(mask.bool().to(torch.int64).sum())
        return masks, region_pixels

    def dataset_map_fn(self, data_dict):
        file_name = data_dict['file_name'] # image file name
        conversations = data_dict['conversations']
        masks = [anno["segmentation"] for anno in data_dict["annotation"]]
        height = data_dict['height']
        width = data_dict['width']
        _ret = {}

        _ret['image'] = file_name
        _ret['height'] = height
        _ret['width'] = width

        masks = self.decode_mask(masks, height, width)
        masks, region_pixels = self._get_region_infos(masks)

        if masks is None:
            return None

        conversations = self._process_conversation(conversations, len(masks), region_pixels)
        _ret['conversation'] = conversations
        _ret['prompt_masks'] = masks
        return _ret

    def replace_image_str(self, data_dict, image_str):
        data_dict['conversation'][0]['input'] = \
            data_dict['conversation'][0]['input'].replace(DEFAULT_IMAGE_TOKEN, image_str)
        return data_dict

    def __getitem__(self, index):

        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        # parse datasets
        result = self.dataset_map_fn(data_dict) # {'image', 'height', 'width', 'conversation', 'masks'}
        if result is None or result['prompt_masks'] is None:
            return self.__getitem__(0)

        data_dict = result

        # process image
        image_file = data_dict['image']
        if isinstance(self.image_folder, list):
            for image_folder in self.image_folder:
                image_path = os.path.join(image_folder, image_file)
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    break
        else:
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
        ori_width, ori_height = image.size

        if self.single_image_mode:
            images = [image]
        else:
            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
        vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
        data_dict['vp_overall_mask'] = vp_overall_mask

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
        # data_dict['prompt_masks'] = data_dict['prompt_masks']

        if data_dict['prompt_masks'] is None:
            return self.__getitem__(0)

        return data_dict


DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give about the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

class OspreyDescriptionDataset(OspreyDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'

    LIMIT=''

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
        super(OspreyDescriptionDataset, self).__init__(
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
        file_name = data_dict['file_name'] # image file name
        descriptions = data_dict['description']
        masks = [anno["segmentation"] for anno in data_dict["annotation"]]
        height = data_dict['height']
        width = data_dict['width']
        _ret = {}

        _ret['image'] = file_name
        _ret['height'] = height
        _ret['width'] = width

        masks = self.decode_mask(masks, height, width)
        masks, region_pixels = self._get_region_infos(masks)

        if masks is None:
            return None

        conversations = self._process_conversation(descriptions, len(masks), region_pixels)
        _ret['conversation'] = conversations
        _ret['prompt_masks'] = masks
        return _ret

    def _process_conversation(self, descriptions, n_regions, region_pixels):
        start_region_str = '<image> There are {} part regions in the picture: '.format(n_regions)
        for i in range(n_regions):
            start_region_str = start_region_str + \
                               f"region{i+1}" + self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN * region_pixels[i] + self.VP_END_TOKEN
            if i == n_regions - 1:
                start_region_str = start_region_str + '.\n'
            else:
                start_region_str = start_region_str + ', '

        converations = []
        for i, item in enumerate(descriptions):
            question = random.choice(DETAILED_QUESTIONS).strip().replace('<region>', f"region{i+1}") + self.LIMIT
            answer = item.replace('<', '').replace('>', '')
            # first conv process
            if i == 0:
                question = start_region_str + question
            converations.append({'from': 'human', 'value': question})
            converations.append({'from': 'gpt', 'value': answer})

        messages = converations
        input = ''

        conversation = []
        while messages and messages[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            messages = messages[1:]
        for msg in messages:
            if msg['from'] == 'human':
                if DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                        '').strip()
                    msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                    msg['value'] = msg['value'].strip()
                input += msg['value']

            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError
        return conversation


class OspreyShortDescriptionDataset(OspreyDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'

    LIMIT = ' Answer the question using a single word or phrase.'

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
        super(OspreyShortDescriptionDataset, self).__init__(
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