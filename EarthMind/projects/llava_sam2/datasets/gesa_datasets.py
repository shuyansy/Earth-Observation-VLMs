"""Datasets for cross-sensor EarthMind training with GeoSync Attention (GeSA).

Every sample emits ONE contiguous visual block, in the order required by the paper
(Appendix B): optical tokens (RGB, or K multispectral pseudo-frames), followed by SAR tokens,
followed by the text instruction:

    <|user|>\n<img>[opt frame 1 .. opt frame K][SAR frame]</img>\n<question> ...

together with `vis_layout = [[modality, tokens_per_frame, n_frames], ...]` that the model uses
to build the GeSA attention mask (Eq. 3) and the shared position ids (Eq. 4).

* `modality`  : 'fused' (optical + SAR), 'optical' or 'sar'  (single-/cross-sensor protocols,
                Table III "Single Modality" baseline, Table VIII).
* `fusion_layout`: 'concat' -> P + N tokens (concatenation / GeSA / external fusion),
                   'avg'    -> N tokens; the model averages co-located tokens (Table III).
* SAR is converted to a pseudo-RGB frame with zero padding (paper default, Table VI).
* All frames of a sample are tiled with the SAME dynamic-resolution grid, so token j of every
  frame covers the same ground location.
"""
import copy
import json
import os
import random
from typing import Literal

import numpy as np
import torch
from mmengine import print_log
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from xtuner.registry import BUILDER

from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
from projects.llava_sam2.hf.models.gesa_utils import (
    MOD_OPT, MOD_SAR, sar_to_pseudo_rgb, ms_folder_to_pseudo_frames, tile_coregistered,
    tiles_layout_to_token_layout, layout_num_tokens)
from .vqa_dataset import Multi_LLaVADataset
from .RefCOCO_Dataset import Multi_ReferSegmDataset

MODALITIES = ('fused', 'optical', 'sar')


def _select_frames(modality, opt_frames, sar_frame):
    if modality == 'fused':
        return [(MOD_OPT, opt_frames), (MOD_SAR, [sar_frame])]
    if modality == 'optical':
        return [(MOD_OPT, opt_frames)]
    if modality == 'sar':
        return [(MOD_SAR, [sar_frame])]
    raise ValueError(modality)


class _GeSAImageMixin:
    """Shared tiling / tokenisation for multi-frame co-registered inputs."""

    def _gesa_init(self, modality='fused', fusion_layout='concat', sar_pad_mode='zero',
                   max_dynamic_patch=12):
        assert modality in MODALITIES, modality
        assert fusion_layout in ('concat', 'avg'), fusion_layout
        self.modality = modality
        self.fusion_layout = fusion_layout
        self.sar_pad_mode = sar_pad_mode
        self.max_dynamic_patch = max_dynamic_patch

    def _encode_frames(self, frames_by_modality):
        tiles, tile_layout, _, _ = tile_coregistered(
            frames_by_modality, self.min_dynamic_patch, self.max_dynamic_patch,
            self.image_size, self.use_thumbnail)
        pixel_values = torch.stack([self.transformer(t) for t in tiles])
        layout = tiles_layout_to_token_layout(tile_layout, self.patch_token)
        if self.fusion_layout == 'avg' and len(layout) > 1:
            n_tokens = layout[0][1]          # co-located tokens are averaged by the model
        else:
            n_tokens = layout_num_tokens(layout)
        image_token_str = f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * n_tokens}{self.IMG_END_TOKEN}'
        return pixel_values, layout, image_token_str


class GeSAPairedVQADataset(_GeSAImageMixin, Multi_LLaVADataset):
    """FusionEO paired optical-SAR instruction data (LLaVA json format)."""

    def __init__(self, *args, modality='fused', fusion_layout='concat', sar_pad_mode='zero',
                 max_dynamic_patch=12, repeats=1, **kwargs):
        self.repeats = repeats
        super().__init__(*args, **kwargs)
        self._gesa_init(modality, fusion_layout, sar_pad_mode, max_dynamic_patch)
        self._n = len(self.data)

    def __len__(self):
        return self._n * self.repeats

    @property
    def modality_length(self):
        return [100] * len(self)

    def prepare_data(self, index):
        data_dict = copy.deepcopy(self.data[index % self._n])
        out = {}
        if data_dict.get('image', None) is None:
            out.update(self.get_inputid_labels(data_dict['conversations'], None))
            out['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)
            out['vis_layout'] = []
            return out
        rgb_path = os.path.join(self.rgb_folder, data_dict['image'])
        sar_path = os.path.join(self.sar_folder, data_dict['image'])
        try:
            opt = Image.open(rgb_path).convert('RGB') if self.modality != 'sar' else None
            sar = sar_to_pseudo_rgb(Image.open(sar_path), self.sar_pad_mode) if self.modality != 'optical' else None
        except Exception as e:  # broken / missing image -> refetch
            print_log(f'Error: {e}', logger='current')
            return None
        if opt is not None and sar is not None and sar.size != opt.size:
            sar = sar.resize(opt.size, resample=Image.BILINEAR)
        pixel_values, layout, image_token_str = self._encode_frames(
            _select_frames(self.modality, [opt] if opt is not None else None, sar))
        out['pixel_values'] = pixel_values
        out['vis_layout'] = layout
        out.update(self.get_inputid_labels(data_dict['conversations'], image_token_str))
        return out


class GeSAPairedRefSegDataset(_GeSAImageMixin, Multi_ReferSegmDataset):
    """Paired optical-SAR referring expression segmentation (RefCOCO-style annotations).

    `grounding_modality` selects the image fed to the SAM2 grounding encoder; the released
    EarthMind pipeline uses the SAR image ('sar').
    """

    def __init__(self, *args, modality='fused', fusion_layout='concat', sar_pad_mode='zero',
                 max_dynamic_patch=12, grounding_modality='sar', repeats=1, **kwargs):
        self.repeats = repeats
        super().__init__(*args, **kwargs)
        self._gesa_init(modality, fusion_layout, sar_pad_mode, max_dynamic_patch)
        if modality == 'optical':
            grounding_modality = 'optical'
        elif modality == 'sar':
            grounding_modality = 'sar'
        self.grounding_modality = grounding_modality

    def __len__(self):
        return super().__len__() * getattr(self, 'repeats', 1)

    @property
    def modality_length(self):
        return [100] * len(self)

    def prepare_data(self, index):
        index = index % super().__len__()
        # --- annotations (same as Multi_ReferSegmDataset._parse_annotations, without image IO) ---
        data_dict = copy.deepcopy(self.get_data_info(index))
        rgb_path, sar_path = data_dict['rgb_img_path'], data_dict['sar_img_path']
        try:
            opt = Image.open(rgb_path).convert('RGB')
            sar = sar_to_pseudo_rgb(Image.open(sar_path), self.sar_pad_mode)
        except Exception as e:
            print_log(f'Error: {e}', logger='current')
            return None
        width, height = opt.size
        if sar.size != opt.size:
            sar = sar.resize(opt.size, resample=Image.BILINEAR)
        instances, text = data_dict['instances'], data_dict['text']
        if len(instances) == 0:
            return None
        masks, phrases = [], []
        for idx in np.random.choice(range(len(instances)), self.num_classes_per_sample, replace=True):
            inst = instances[idx]
            phrase = text[idx].lower()
            if phrase.endswith('.'):
                phrase = phrase[:-1]
            phrases.append(phrase)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in inst['mask']:
                rles = mask_utils.frPyObjects([seg], height, width)
                binary_mask += mask_utils.decode(rles).astype(np.uint8).squeeze()
            masks.append(binary_mask)
        conversation = []
        for i, phrase in enumerate(phrases):
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})

        out = {'masks': torch.stack([torch.from_numpy(m) for m in masks], 0)}
        g_img = sar if self.grounding_modality == 'sar' else opt
        g = self.extra_image_processor.apply_image(np.array(g_img))
        out['g_pixel_values'] = torch.from_numpy(g).permute(2, 0, 1).contiguous()
        pixel_values, layout, image_token_str = self._encode_frames(
            _select_frames(self.modality, [opt], sar))
        out['pixel_values'] = pixel_values
        out['vis_layout'] = layout
        out.update(self.get_inputid_labels(conversation, image_token_str))
        return out


class GeSAMultiSpectralDataset(_GeSAImageMixin, Multi_LLaVADataset):
    """Multispectral (Sentinel-2) instruction data: 12 bands -> K pseudo-frames by three-band
    grouping (frame 0 = true colour).  If `sar_folder` is given (e.g. BigEarthNet-MM S1 or the
    EOFuse-Bench MS-SAR split) the SAR frame is appended after the K optical frames.

    json entries: {"id": <patch id>, "image": ..., "conversations": [...]}; band files are read
    from <ms_folder>/<id>/<id>_Bxx.tif (BigEarthNet layout) or <ms_folder>/<id>_Bxx.tif.
    """

    def __init__(self, tokenizer, data_path, prompt_template, special_tokens=None,
                 ms_folder=None, sar_folder=None, modality='optical', group_size=3,
                 fusion_layout='concat', sar_pad_mode='zero', max_dynamic_patch=12,
                 max_length=8192, repeats=1, **kwargs):
        self.repeats = repeats
        super().__init__(tokenizer=tokenizer, data_path=data_path, prompt_template=prompt_template,
                         special_tokens=special_tokens, data_root=ms_folder,
                         data_prefix=dict(rgb_path='', sar_path=''), max_length=max_length, **kwargs)
        self._gesa_init(modality, fusion_layout, sar_pad_mode, max_dynamic_patch)
        self.ms_folder = ms_folder
        self.ms_sar_folder = sar_folder
        self.group_size = group_size
        self._n = len(self.data)

    def __len__(self):
        return self._n * self.repeats

    @property
    def modality_length(self):
        return [100] * len(self)

    def _load_ms(self, pid):
        sub = os.path.join(self.ms_folder, pid)
        if os.path.isdir(sub):
            return ms_folder_to_pseudo_frames(sub, group_size=self.group_size)
        return ms_folder_to_pseudo_frames(self.ms_folder, prefix=pid, group_size=self.group_size)

    def prepare_data(self, index):
        data_dict = copy.deepcopy(self.data[index % self._n])
        pid = data_dict.get('id') or os.path.splitext(data_dict['image'])[0]
        try:
            opt_frames = self._load_ms(pid) if self.modality != 'sar' else None
            sar = None
            if self.modality != 'optical':
                assert self.ms_sar_folder is not None, 'SAR folder required for fused/sar modality'
                sar = sar_to_pseudo_rgb(Image.open(os.path.join(self.ms_sar_folder, pid + '.png')),
                                        self.sar_pad_mode)
        except Exception as e:
            print_log(f'Error: {e}', logger='current')
            return None
        pixel_values, layout, image_token_str = self._encode_frames(
            _select_frames(self.modality, opt_frames, sar))
        out = {'pixel_values': pixel_values, 'vis_layout': layout}
        out.update(self.get_inputid_labels(data_dict['conversations'], image_token_str))
        return out
