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

from xtuner.registry import BUILDER

from xtuner.dataset.utils import encode_fn
from xtuner.dataset.map_fns import llava_map_fn

from projects.glamm.datasets.utils.utils import expand2square

from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class SemanticSegDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 num_proc=8,
                 lazy=False,
                 repeats=1,
                 gcg_format=False,
                 num_classes_per_sample=3,
                 extra_image_processor=None):
        super().__init__()
        self.gcg_format = gcg_format
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

        assert offline_processed_text_folder or (data_path and tokenizer)
        self.lazy = lazy

        self.max_length = max_length
        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            self.image_label_datas = self.json_file_preprocess(data_path, image_folder)

        self.image_folder = image_folder

        if isinstance(image_processor, dict) or isinstance(image_processor, Config) or isinstance(image_processor, ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        size = self.image_processor.crop_size

        if isinstance(size, dict):
            self.image_w, self.image_h = size['width'], size['height']
        elif isinstance(size, int):
            self.image_h, self.image_w = size, size
        else:
            self.image_w, self.image_h = size

        self.pad_image_to_square = pad_image_to_square
        self.down_ratio = 1
        self.repeats = repeats

    def json_file_preprocess(self, data_path, image_folder):
        # ade20k
        with open(data_path, 'r') as file:
            ade20k_classes = json.load(file)
        ade20k_image_dir = image_folder
        ade20k_images = [os.path.join(ade20k_image_dir, img) for img in os.listdir(ade20k_image_dir) if
                         img.endswith('.jpg')]
        ade20k_labels = [img.replace(".jpg", ".png").replace(
            "images", "annotations") for img in ade20k_images]
        self.classes = np.array(ade20k_classes)

        ret = []
        for image, label in zip(ade20k_images, ade20k_labels):
            ret.append({"image": image, "label": label})
        return ret

    def __len__(self):
        return len(self.image_label_datas) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.image_label_datas:
            length_list.append(100)
        length_list = length_list * self.repeats
        return length_list

    def real_len(self):
        return len(self.image_label_datas)

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))

        # ade20k
        label = np.where(label == 0, 255, label - 1)
        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            return None, None

        selected_labels = np.random.choice(unique_labels, min(
            len(unique_labels), self.num_classes_per_sample), replace=False)
        label = torch.from_numpy(label).long()
        masks = torch.stack([label == class_id for class_id in selected_labels], dim=0)
        return masks, selected_labels

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.image_label_datas[index])

        assert 'image' in data_dict.keys()
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(image_file).convert('RGB')
            if hasattr(self, 'extra_image_processor'):
                g_image = np.array(image) # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                data_dict['g_pixel_values'] = g_pixel_values

            ori_width, ori_height = image.size
            if self.pad_image_to_square:
                image = expand2square(image, tuple(int(x * 255)
                                      for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image

            # process and get masks
            data_dict['masks'], class_id = self.decode_mask(data_dict['label'])
            if class_id is None:
                return self.__getitem__(0)

            if self.gcg_format:
                pass
            else:
                conversation = []
                for i, c_id in enumerate(class_id):
                    question = random.choice(SEG_QUESTIONS).format(
                        class_name=self.classes[c_id].lower())
                    if i == 0:
                        question = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n""" + question
                    conversation.append(
                        {'input': question, 'output': random.choice(ANSWER_LIST)})

            data_dict.update({'conversation': conversation})
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['masks'] = None

        if self.lazy:
            result = self.template_map_fn(data_dict)
            data_dict.update(result)

            result = encode_fn(data_dict, tokenizer=self.tokenizer,
                               max_length=self.max_length, with_image_token=True)
            data_dict.update(result)

        return data_dict

class ADE20kSemanticSegDataset(SemanticSegDataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 num_proc=8,
                 lazy=False,
                 repeats=1,
                 gcg_format=False,
                 num_classes_per_sample=3,
                 extra_image_processor=None):
        super().__init__(
            image_folder=image_folder,
            image_processor=image_processor,
            data_path=data_path,
            tokenizer=tokenizer,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            num_proc=num_proc,
            lazy=lazy,
            repeats=repeats,
            gcg_format=gcg_format,
            num_classes_per_sample=num_classes_per_sample,
            extra_image_processor=extra_image_processor,
        )

class COCOStuffSemanticSegDataset(SemanticSegDataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 num_proc=8,
                 lazy=False,
                 repeats=1,
                 label_path=None,
                 gcg_format=False,
                 num_classes_per_sample=3,
                 extra_image_processor=None):
        self.label_path = label_path
        super().__init__(
            image_folder=image_folder,
            image_processor=image_processor,
            data_path=data_path,
            tokenizer=tokenizer,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            num_proc=num_proc,
            lazy=lazy,
            repeats=repeats,
            gcg_format=gcg_format,
            num_classes_per_sample=num_classes_per_sample,
            extra_image_processor=extra_image_processor,
        )
        self.cocostuff_class2index = {c: i for i, c in enumerate(self.classes)}

    def json_file_preprocess(self, data_path, image_folder):
        # coco stuff
        assert self.label_path is not None
        with open(data_path, 'r') as file:
            cocostuff_classes = [line.strip().split(": ")[-1]
                                 for line in file.readlines()[1:]]
        coco_stuff_image_dir = image_folder
        coco_stuff_label_dir = self.label_path
        coco_stuff_labels = glob.glob(
            os.path.join(coco_stuff_label_dir, "*.png"))

        coco_stuff_images = [label.replace(".png", ".jpg").replace(coco_stuff_label_dir, coco_stuff_image_dir)
                             for label in coco_stuff_labels]

        self.classes = np.array(cocostuff_classes)

        ret = []
        for image, label in zip(coco_stuff_images, coco_stuff_labels):
            ret.append({"image": image, "label": label})
        return ret

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))

        # coco stuff
        ignored_classes = [index for class_name,
                           index in self.cocostuff_class2index.items() if "-" in class_name]
        label = np.where(np.isin(label, ignored_classes), 255, label)

        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            print("No valid label !!!")
            return None, None

        # only choose 1
        selected_labels = np.random.choice(unique_labels, min(
            len(unique_labels), self.num_classes_per_sample), replace=False)

        label = torch.from_numpy(label).long()
        masks = torch.stack(
            [label == class_id for class_id in selected_labels], dim=0)
        return masks, selected_labels

class PascalPartSemanticSegDataset(SemanticSegDataset):

    def json_file_preprocess(self, data_path, image_folder):
        self.coco_api = COCO(data_path)
        img_ids = self.coco_api.getImgIds()
        all_classes = self.coco_api.loadCats(self.coco_api.getCatIds())
        class_map_pascal_part = {}
        for cat in all_classes:
            cat_main, cat_part = cat["name"].strip().split(":")
            name = (cat_main, cat_part)
            class_map_pascal_part[cat["id"]] = name
        self.classes = class_map_pascal_part
        return img_ids

    def __getitem__(self, index):
        index = index % self.real_len()
        img_id = self.image_label_datas[index]
        img_info = self.coco_api.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        data_dict = {}

        image_file = os.path.join(self.image_folder, file_name)
        image = Image.open(image_file).convert('RGB')

        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            data_dict['g_pixel_values'] = g_pixel_values

        if self.pad_image_to_square:
            image = expand2square(
                image,  tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        data_dict['pixel_values'] = image

        annotation_ids = self.coco_api.getAnnIds(imgIds=img_info["id"])
        annotations = self.coco_api.loadAnns(annotation_ids)

        if not annotations:
            return self.__getitem__(0)

        sampled_anns = np.random.choice(annotations, min(
            len(annotations), self.num_classes_per_sample), replace=False)

        conversation = []
        for i, ann in enumerate(sampled_anns):
            cat_id = ann['category_id']
            sampled_cls = self.classes[cat_id]
            if isinstance(sampled_cls, tuple):
                obj, part = sampled_cls
                name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
            else:
                name = sampled_cls
            question = random.choice(SEG_QUESTIONS).format(class_name=name)
            if i == 0:
                question = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n""" + question
            conversation.append(
                {'input': question, 'output': random.choice(ANSWER_LIST)})

        masks = [self.coco_api.annToMask(ann) for ann in sampled_anns]
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)

        data_dict['masks'] = masks
        data_dict['conversation'] = conversation

        if self.lazy:
            result = self.template_map_fn(data_dict)
            data_dict.update(result)

            result = encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length, with_image_token=True)
            data_dict.update(result)

        return data_dict

class PacoSemanticSegDataset(PascalPartSemanticSegDataset):
    def json_file_preprocess(self, data_path, image_folder):
        self.coco_api = COCO(data_path)
        all_classes = self.coco_api.loadCats(self.coco_api.getCatIds())
        class_map_paco = {}
        for cat in all_classes:
            cat_split = cat["name"].strip().split(":")
            if len(cat_split) == 1:
                name = cat_split[0].split("_(")[0]
            else:
                assert len(cat_split) == 2
                obj, part = cat_split
                obj = obj.split("_(")[0]
                part = part.split("_(")[0]
                name = (obj, part)
            class_map_paco[cat["id"]] = name
        self.classes = class_map_paco
        return self.coco_api.getImgIds()