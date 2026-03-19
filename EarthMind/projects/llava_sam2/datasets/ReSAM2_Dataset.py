import logging
import os
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import copy
from .encode_fn import video_lisa_encode_fn
import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

SEG_QUESTIONS = [
    "Please segment the object according to the description: {class_name}",
]

SEG_QUESTIONS_SHORT = [
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

class VideoSAM2Dataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    def __init__(self,
                 sam2_folder,
                 expression_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 select_number=5,
                 sampled_frames=5,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=8196,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 use_fast=False,
                 n_fast_images=50,
                 fast_pool_size=4,
                 mode='long',
                 frame_contiguous_sample=False,
    ):
        assert mode in ['long', 'long_short', 'short']
        self.mode = mode
        self.cur_mode = mode
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

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            video_ids, anno_dict = self.json_file_preprocess(expression_file)
            if self.lazy:
                self.video_ids = video_ids
                self.anno_dict = anno_dict
            else:
                raise NotImplementedError

        self.sam2_folder = sam2_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = repeats

        self._system = ''

        self.downsample_ratio = 0.5
        self.image_size = 448
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

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        self.frame_contiguous_sample = frame_contiguous_sample

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        print("Video res dataset (ref-sam2), include {} items.".format(len(self.video_ids)))

    def __len__(self):
        return len(self.video_ids) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.video_ids:
            cur_len = 20000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.video_ids)

    def json_file_preprocess(self, expression_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)

        video_ids = list(expression_datas.keys())
        return video_ids, expression_datas

    def dataset_map_fn(self, objects_expression_infos, n_frames, n_fast_frames=0):
        # prepare text
        if self.mode == 'long':
            expressions = [object_info['formated'] for object_info in objects_expression_infos]
            self.cur_mode = self.mode
        elif self.mode == 'short':
            expressions = [object_info['short_caps'][random.randint(0, len(object_info['short_caps'])-1)] for object_info in objects_expression_infos]
            self.cur_mode = self.mode
        else:
            if random.random() < 0.5:
                expressions = [object_info['formated'] for object_info in objects_expression_infos]
                self.cur_mode = 'long'
            else:
                expressions = [object_info['short_caps'][random.randint(0, len(object_info['short_caps']) - 1)] for
                               object_info in objects_expression_infos]
                self.cur_mode = 'short'
        text_dict = self.prepare_text(n_frames, expressions, num_image_tokens=self.patch_token,
                                      n_fast_frames=n_fast_frames)
        ret = {'conversation': text_dict['conversation']}
        return ret

    def prepare_text(self, n_frames, expressions, num_image_tokens=256, n_fast_frames=0):

        if self.use_fast:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_frames * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        questions = []
        answers = []
        for i, exp in enumerate(expressions):
            if self.cur_mode == 'short':
                question_template = random.choice(SEG_QUESTIONS_SHORT)
                exp = exp.replace("A ", '')
            else:
                question_template = random.choice(SEG_QUESTIONS)
            questions.append(question_template.format(class_name=exp))
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
                    {'from': 'human', 'value': frame_tokens + question}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': question}
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
        video_id = self.video_ids[index]
        expression_dict = self.anno_dict[video_id]
        object_ids = list(expression_dict['objects'].keys())

        video_path = os.path.join(self.sam2_folder, expression_dict['video_path'])
        anno_path = os.path.join(self.sam2_folder, expression_dict['anno_path'])

        video_frames = get_video_frames(video_path)

        if self.use_fast:
            # sample fast branch
            fast_interval = len(video_frames) / (self.n_fast_images + 1e-4)
            sampled_fast_frame_idxs = [min(int(i * fast_interval), len(video_frames) - 1) for i in range(self.n_fast_images)]
            fast_video_frames = [video_frames[_idx] for _idx in sampled_fast_frame_idxs]
        else:
            fast_video_frames = None

        video_frames = video_frames[::4]

        # mask annotation
        with open(anno_path, 'r') as f:
            mask_data = json.load(f)
        masklents = decode_masklet(mask_data['masklet'])

        n_frames = len(masklents)
        n_objects = len(object_ids)

        # sample object
        if n_objects > self.select_number:
            selected_indexes = np.random.choice(n_objects, self.select_number)
        else:
            selected_indexes = np.random.choice(n_objects, self.select_number, replace=True)

        selected_object_ids = [object_ids[_idx] for _idx in selected_indexes]
        objects_expression_infos = [expression_dict['objects'][_idx] for _idx in selected_object_ids]
        _masklents = []
        for _mask in masklents:
            _mask_selected = []
            for _idx in selected_object_ids:
                _mask_selected.append(_mask[:, :, int(_idx)])
            _mask_selected = np.stack(_mask_selected, axis=2)
            _masklents.append(_mask_selected)
        masklents = _masklents

        # sample video frames
        # prepare images, random select k frames
        if n_frames > self.sampled_frames + 1:
            if self.frame_contiguous_sample and random.random() < 0.5:
                # do contiguous sample
                selected_start_frame = np.random.choice(n_frames - self.sampled_frames, 1, replace=False)
                selected_frame_indexes = [selected_start_frame[0] + _i for _i in range(self.sampled_frames)]
            else:
                selected_frame_indexes = np.random.choice(n_frames, self.sampled_frames, replace=False)
        else:
            selected_frame_indexes = np.random.choice(n_frames, self.sampled_frames, replace=True)
        selected_frame_indexes.sort()

        video_frames = [video_frames[_idx] for _idx in selected_frame_indexes]
        masklents = [masklents[_idx] for _idx in selected_frame_indexes]

        data_dict = self.dataset_map_fn(objects_expression_infos, len(video_frames), n_fast_frames=self.n_fast_images)
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length, with_image_token=True)
        data_dict.update(result)

        pixel_values = []
        extra_pixel_values = []
        for frame in video_frames:
            frame = frame[:, :, ::-1]
            frame_image = Image.fromarray(frame).convert('RGB')
            ori_width, ori_height = frame_image.size
            if self.extra_image_processor is not None:
                g_image = np.array(frame_image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)

            frame_image = self.transformer(frame_image)
            pixel_values.append(frame_image)

        pixel_values = torch.stack(pixel_values, dim=0)  # (n_f, 3, h, w)
        data_dict['pixel_values'] = pixel_values
        if self.extra_image_processor is not None:
            data_dict['g_pixel_values'] = extra_pixel_values

        # for fast branch
        if self.use_fast:
            fast_pixel_values = []
            for frame_image in fast_video_frames:
                frame = frame_image[:, :, ::-1]
                frame_image = Image.fromarray(frame).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values

        # process and get masks
        masklents = np.stack(masklents, axis=0)  # (n_frames, h, w, n_obj)
        masklents = torch.from_numpy(masklents).permute(3, 0, 1, 2)
        masklents = masklents.flatten(0, 1)
        # print('sam2-mask_shape:', masklents.shape)
        # print('sam2-pixel_values:', data_dict['pixel_values'].shape)
        # print('sam2-g_pixel_values:', len(data_dict['g_pixel_values']), ', ', data_dict['g_pixel_values'][0].shape)
        data_dict['masks'] = masklents
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

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        frame_id += 1

    cap.release()
    return frames


def images_to_video(frames, video_name, fps=6):
    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    # cv2.destroyAllWindows()
    video.release()
    return

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = maskUtils.decode(_rle)
        masks.append(mask)
    return masks

def draw_mask(image, mask):
    obj_mask = mask * 255
    obj_mask = np.stack([obj_mask * 1, obj_mask * 0, obj_mask * 0], axis=2)
    obj_mask = obj_mask * 0.5 + copy.deepcopy(image) * 0.5
    obj_mask = obj_mask.astype(np.uint8)
    return obj_mask

def add_mask2images(frames, masklets):
    show_videos = []
    for i_frames, (frame, masks) in enumerate(zip(frames, masklets)):
        if i_frames == 0:
            n_obj = masks.shape[-1]
            for i_obj in range(n_obj):
                show_videos.append([])

        n_obj = masks.shape[-1]
        for i_obj in range(n_obj):
            show_videos[i_obj].append(draw_mask(copy.deepcopy(frame), masks[:, :, i_obj]))
    return show_videos