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

from .encode_fn import video_lisa_encode_fn
import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

class VideoReVOSDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 select_number=5,
                 sampled_frames=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 frame_contiguous_sample=False,
                 use_fast=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 # only work if use_fast = True
                 n_fast_images=50,
                 fast_pool_size=4,
                 fast_token_after_question=False,
    ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        assert offline_processed_text_folder or (expression_file and tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''


        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
            self.vid2metaid = vid2metaid
            self.videos = list(self.vid2metaid.keys())
            self.mask_dict = mask_dict
            self.json_datas = metas
            json_datas = metas
            json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            if self.lazy:
                self.text_data = build_origin_dataset(json_data, 'train')
            else:
                raise NotImplementedError

        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = repeats

        self._system = ''

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

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        self.frame_contiguous_sample = frame_contiguous_sample

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        # exist_thr
        self.exist_thr = 8
        self.fast_token_after_question = fast_token_after_question
        if self.fast_token_after_question:
            assert self.use_fast

        print("Video res dataset, include {} items.".format(len(self.vid2metaid)))

    def __len__(self):
        return len(self.vid2metaid) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.vid2metaid:
            cur_len = 10000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.vid2metaid)

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']  # str
                meta['mask_anno_id'] = exp_dict['anno_id']

                if 'obj_id' in exp_dict.keys():
                    meta['obj_id'] = exp_dict['obj_id']
                else:
                    meta['obj_id'] = [0, ]  # Ref-Youtube-VOS only has one object per expression
                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id

                meta['length'] = vid_len
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        # process mask annotation files
        with open(mask_file, 'rb') as f:
            mask_dict = json.load(f)

        return vid2metaid, metas, mask_dict

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, video_masks, image_size):
        ret_masks = []
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                if len(ret_masks) != 0:
                    _object_masks = ret_masks[0] * 0
                else:
                    _object_masks = np.zeros(
                        (self.sampled_frames, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                for i_frame in range(len(object_masks[0])):
                    _mask = np.zeros(image_size, dtype=np.uint8)
                    for i_anno in range(len(object_masks)):
                        if object_masks[i_anno][i_frame] is None:
                            continue
                        m = maskUtils.decode(object_masks[i_anno][i_frame])
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        _mask = _mask | m
                    _object_masks.append(_mask)
                _object_masks = np.stack(_object_masks, axis=0)
            # if self.pad_image_to_square:
            #     _object_masks = expand2square_mask(_object_masks)
            ret_masks.append(_object_masks)
        _shape = ret_masks[0].shape
        for item in ret_masks:
            if item.shape != _shape:
                print([_ret_mask.shape for _ret_mask in ret_masks])
                return None
        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_frames, h, w)

        ret_masks = torch.from_numpy(ret_masks)
        # ret_masks = F.interpolate(ret_masks, size=(self.image_size // self.down_ratio,
        #                           self.image_size // self.down_ratio), mode='nearest')
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def dataset_map_fn(self, data_dict, select_k=5):
        images = []

        len_frames = len(data_dict[0]['frames'])
        for objet_info in data_dict:
            assert len_frames == len(objet_info['frames'])

        # prepare images, random select k frames
        if len_frames > select_k + 1:
            if self.frame_contiguous_sample and random.random() < 0.5:
                # do contiguous sample
                selected_start_frame = np.random.choice(len_frames - select_k, 1, replace=False)
                selected_frame_indexes = [selected_start_frame[0] + _i for _i in range(select_k)]
            else:
                selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        selected_frame_indexes.sort()

        if self.use_fast:
            # sample fast branch
            fast_interval = len_frames / (self.n_fast_images + 1e-4)
            sampled_fast_frame_idxs = [min(int(i * fast_interval), len_frames - 1) for i in range(self.n_fast_images)]
            fast_video_frames = []
            for selected_frame_index in sampled_fast_frame_idxs:
                frame_id = data_dict[0]['frames'][selected_frame_index]
                fast_video_frames.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))
        else:
            fast_video_frames = None
            sampled_fast_frame_idxs = None

        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))

        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]
        if self.use_fast:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token,
                                          n_fast_images=len(fast_video_frames),)
        else:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token)


        # prepare masks
        video_masks = []
        for object_info in data_dict:
            anno_ids = object_info['mask_anno_id']
            # print('anno_ids: ', anno_ids)
            obj_masks = []
            for anno_id in anno_ids:
                anno_id = str(anno_id)
                frames_masks = self.mask_dict[anno_id]
                frames_masks_ = []
                for frame_idx in selected_frame_indexes:
                    frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                obj_masks.append(frames_masks_)
            video_masks.append(obj_masks)

        if self.use_fast:
            fast_video_masks = []
            assert sampled_fast_frame_idxs is not None
            for object_info in data_dict:
                anno_ids = object_info['mask_anno_id']
                obj_masks = []
                for anno_id in anno_ids:
                    anno_id = str(anno_id)
                    frames_masks = self.mask_dict[anno_id]
                    frames_masks_ = []
                    for frame_idx in sampled_fast_frame_idxs:
                        frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                    obj_masks.append(frames_masks_)
                fast_video_masks.append(obj_masks)
        else:
            fast_video_masks = None

        ret = {'images': images, 'video_masks': video_masks, 'conversation': text_dict['conversation'],
               'fast_images': fast_video_frames, 'fast_video_masks': fast_video_masks}
        return ret

    def prepare_text(self, n_frames, expressions, num_image_tokens=256, n_fast_images=50):

        if self.use_fast and not self.fast_token_after_question:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'
        if self.fast_token_after_question:
            assert self.use_fast
            after_question_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}'
        else:
            after_question_str = ''

        questions = []
        answers = []
        for i, exp in enumerate(expressions):
            # the exp is a question
            if '?' in exp:
                questions.append(exp)
            else:
                exp = exp.replace('.', '').strip()
                question_template = random.choice(SEG_QUESTIONS)
                questions.append(question_template.format(class_name=exp.lower()))

            answers.append(random.choice(ANSWER_LIST))
        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = frame_token_str + '\n'
                # frame_tokens = '=' + ' '
                frame_tokens = frame_tokens * n_frames
                frame_tokens = frame_tokens.strip()
                frame_tokens = fast_frame_token_str + frame_tokens
                qa_list.append(
                    {'from': 'human', 'value': frame_tokens + question + after_question_str}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': question + after_question_str}
                )
            qa_list.append(
                {'from': 'gpt', 'value': answer}
            )

        input = ''
        conversation = []
        for msg in qa_list:
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        # add system information
        conversation[0].update({'system': self._system})
        return {'conversation': conversation}

    def __getitem__(self, index):
        index = index % self.real_len()
        selected_video_objects = self.vid2metaid[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        data_dict = self.dataset_map_fn(video_objects_infos, select_k=self.sampled_frames)

        assert 'images' in data_dict.keys()
        pixel_values = []
        extra_pixel_values = []
        num_video_tokens = None
        num_frame_tokens = None
        if data_dict.get('images', None) is not None:
            frames_files = data_dict['images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size
                if self.extra_image_processor is not None:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)

                if self.preprocessor is not None:
                    pass
                else:
                    frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)

            if self.preprocessor is not None:
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    num_frames = _data_dict['image_grid_thw'].shape[0]
                    num_video_tokens = num_frame_tokens * num_frames
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                else:
                    raise NotImplementedError
                data_dict.update(_data_dict)
            else:
                pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
                data_dict['pixel_values'] = pixel_values
            if self.extra_image_processor is not None:
                data_dict['g_pixel_values'] = extra_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['video_masks'], image_size=(ori_height, ori_width))
            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))
            data_dict['masks'] = masks
        else:
            data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            data_dict['masks'] = None

        if num_video_tokens is not None:
            assert self.patch_token == 1
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][0]['input'] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        # for fast branch
        if self.use_fast:
            fast_pixel_values = []
            frames_files = data_dict['fast_images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['fast_video_masks'], image_size=(ori_height, ori_width))

            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))

            data_dict['fast_exists'] = masks.to(dtype=torch.int).sum(dim=(-2, -1)).ge(self.exist_thr).unsqueeze(-1)


            del data_dict['fast_video_masks']
        data_dict['type'] = 'video'
        return data_dict

    def visualization_debug(self, data_dict):
        save_folder = os.path.join(self.save_folder, 'sample_{}'.format(self.cur_number))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.cur_number += 1

        # images

        show_images = []

        pixel_values = data_dict['pixel_values']
        save_folder_image = os.path.join(save_folder, 'image')
        if not os.path.exists(save_folder_image):
            os.mkdir(save_folder_image)
        for i_image, image_pixel_value in enumerate(pixel_values):
            # print(image_pixel_value.shape)
            image_pixel_value[0] = image_pixel_value[0] * 0.2686
            image_pixel_value[1] = image_pixel_value[1] * 0.2613
            image_pixel_value[2] = image_pixel_value[2] * 0.2757
            image_pixel_value[0] = image_pixel_value[0] + 0.4814
            image_pixel_value[1] = image_pixel_value[1] + 0.4578
            image_pixel_value[2] = image_pixel_value[2] + 0.4082
            image_pixel_value = image_pixel_value * 255
            image_pixel_value = image_pixel_value.permute(1, 2, 0)
            image_pixel_value = image_pixel_value.to(torch.uint8).numpy()
            # print(os.path.join(save_folder_image, '{}.jpg'.format(i_image)))
            # print(image_pixel_value.shape)
            show_images.append(image_pixel_value)
            cv2.imwrite(os.path.join(save_folder_image, '{}.jpg'.format(i_image)), image_pixel_value)

        # text
        input_text = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=False)
        with open(os.path.join(save_folder, 'text.json'), 'w') as f:
            json.dump([input_text], f)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_frames = len(pixel_values)
        masks = data_dict['masks']
        _, h, w = masks.shape
        masks = masks.reshape(-1, n_frames, h, w)
        for i_obj, obj_masks in enumerate(masks):
            save_folder_mask_obj_folder = os.path.join(save_folder_mask, 'obj_{}'.format(i_obj))
            if not os.path.exists(save_folder_mask_obj_folder):
                os.mkdir(save_folder_mask_obj_folder)
            for i_frame, f_mask in enumerate(obj_masks):
                f_mask = f_mask.numpy()
                f_mask = f_mask * 255
                f_mask = np.stack([f_mask * 1, f_mask * 0, f_mask * 0], axis=2)
                f_mask = show_images[i_frame] * 0.3 + 0.7 * f_mask
                f_mask = f_mask.astype(np.uint8)
                cv2.imwrite(os.path.join(save_folder_mask_obj_folder, '{}.png'.format(i_frame)), f_mask)
        return
