from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parts.mmdet.models.losses import CrossEntropyLoss

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from .lisa import LisaModel
import torch
import torch.nn.functional as F
from copy import deepcopy
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.terrascope.models.preprocess.image_resize import DirectResize
from typing import List, Tuple

import numpy as np

from .internvl import InternVL_Slowfast
from .utils import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools import mask as _mask

from types import MethodType

from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
import re

def edge_loss(pred_masks, gt_masks):
    """
    SobelLoss - 0-1
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred_masks.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred_masks.device)
    
    pred_sig = pred_masks.sigmoid() if pred_masks.requires_grad else pred_masks
    pred_edge_x = F.conv2d(pred_sig.unsqueeze(1), sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_sig.unsqueeze(1), sobel_y, padding=1)
    pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
    
    gt_edge_x = F.conv2d(gt_masks.float().unsqueeze(1), sobel_x, padding=1)
    gt_edge_y = F.conv2d(gt_masks.float().unsqueeze(1), sobel_y, padding=1)
    gt_edge = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-6)
    
    pred_edge = pred_edge / (pred_edge.max() + 1e-6)
    gt_edge = gt_edge / (gt_edge.max() + 1e-6)
    
    return F.l1_loss(pred_edge, gt_edge)



class VideoLLaVASAMModel(LisaModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 use_fast_supervision=False,
                 # for inference
                 phi3=True,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # for inference large model
                 split_model=False,
                 # ext
                 preprocessor=None,
                 # bs
                 bs:int=0,
                 ):
        super(LisaModel, self).__init__()
        self.split_model = split_model
        if split_model:
            mllm.model_split = split_model
        if special_tokens is None:
            special_tokens = ['[SEG]']
        self.special_tokens = special_tokens
        if 'special_tokens' not in mllm.keys():
            mllm.special_tokens = special_tokens
        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size
        if hasattr(self.mllm, '_post_init'):
            self.mllm._post_init(
                fast_pool_size=self.fast_pool_size,
                fast_pool=self.fast_pool
            )
        else:
            print("No _post_init() in mllm !!!")

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type == 'qwen':
                self.mllm.model.model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.get_output_embeddings().weight.requires_grad_(True)
            elif self.arch_type == 'llava':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            # self.mllm.model.language_model.base_model.model.lm_head.requires_grad_(True)
            # self.mllm.model.language_model.base_model.model.model.embed_tokens.requires_grad_(True)

        if self.arch_type == 'intern_vl':
            in_dim = self.mllm.model.config.llm_config.hidden_size
        elif self.arch_type == 'qwen':
            in_dim = self.mllm.model.config.hidden_size
        elif self.arch_type == 'llava':
            # for llava, the hidden size is in language model
            in_dim = self.mllm.model.language_model.config.hidden_size
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        if use_fast_supervision:
            self.text_exist_fcs = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
                nn.Linear(in_dim, 1), nn.Dropout(0.0)
            )

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        if use_fast_supervision:
            self.loss_exists = BUILDER.build(dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0)
            )

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        if fast_pool:
            self.fast_token_idx = self.tokenizer("<FAST_IMG_CONTEXT>", add_special_tokens=False).input_ids[0]
        else:
            self.fast_token_idx = None
        self.use_fast_supervision = use_fast_supervision

        self.phi3 = phi3
        self.template = template

        if preprocessor is None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = BUILDER.build(preprocessor)

        self.bs = bs

    def _merge_lora(self):
        # print('pre merge lora: ', self.mllm.model.language_model.base_model.model.get_input_embeddings().weight.shape)
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        # print('after merge lora: ', self.mllm.model.language_model.get_input_embeddings().weight.shape)
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        return state_dict

    def activation_checkpointing_disable(self):
        if self.arch_type == 'qwen':
            self.mllm.model.model.gradient_checkpointing_disable()
        else:
            self.mllm.model.language_model.gradient_checkpointing_disable()


    def _add_special_tokens(self):
        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        # if not isinstance(self.mllm.model.language_model.get_output_embeddings(), nn.Linear):
        #     print("Change the lm_head to nn.Linear !!!")
        #     transposed = False
        #     old_lm_head = self.mllm.model.language_model.get_output_embeddings()
        #     old_num_tokens, old_lm_head_dim = (
        #         old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        #     )
        #     new_lm_head_shape = (old_lm_head_dim, len(tokenizer)) if not transposed else (
        #     len(tokenizer), old_lm_head_dim)
        #     has_new_lm_head_bias = old_lm_head.bias is not None
        #     new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias).to(self.device)
        #     new_lm_head.weight = old_lm_head.weight
        #     new_lm_head.bias = old_lm_head.bias
        #     self.mllm.model.language_model.set_output_embeddings(new_lm_head)

        # this is already done in mllm
        # if num_new_tokens > 0:
        #     self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))

        # assert isinstance(self.mllm, InternVL_Slowfast)
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
            elif self.arch_type == 'qwen':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
                )
            elif self.arch_type == 'llava':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
            raise NotImplementedError
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})

        # Step 4. mask decoder of grounding model (SAM/SAM2)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mask_decoder' in k})

        # Step 5. others (fcs)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_exist_fcs.' in k}
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k or 'output' in k and 'sam2_model' not in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k or 'tok_embeddings' in k})
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        assert self.bs > 0
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        fast_exists = data.pop('fast_exists', None)
        region_info=data['region_info']
        print("11111",region_info)
        # if self.arch_type == 'llava' and data.get('pixel_values', None) is not None:
        #     data['pixel_values'] = data['pixel_values'].to(self.torch_dtype)
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            print("2")
            output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError

        if self.use_fast_supervision and fast_exists is not None:
            # gt_exists = []
            # for id_x, _fast_exists in enumerate(fast_exists):
            #     num_tot = _fast_exists.shape[0]
            #     num_conv = gt_masks[id_x].shape[0] // frames_per_batch[id_x]
            #     assert num_tot % num_conv == 0
            #     gt_exists.append(_fast_exists.reshape(num_conv, num_tot // num_conv))
            fast_flag = input_ids == self.fast_token_idx
            fast_tokens = output.hidden_states[-1][fast_flag]
            exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
            gt_exists = torch.cat(fast_exists)
            loss_exists = self.loss_exists(exists_logit, gt_exists)
        else:
            loss_exists = None

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        if loss_exists is not None:
            loss_dict['loss_exists'] = loss_exists
        return loss_dict

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch):
            success = True
        else:
            success = False
            print("len(pred_embeddings_list):{} is not equal to len(frames_per_batch):{} !!!".format(len(pred_embeddings_list), len(frames_per_batch)))
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video, success

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        if self.template is None:
            self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.mllm.to(self.torch_dtype)
        self.text_hidden_fcs.to(self.torch_dtype)
        # if getattr(self, 'text_exist_fcs', None) is not None:
        #     self.text_exist_fcs.to(self.torch_dtype)

        # for sam image processor
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        if 'max_dynamic_patch' in metainfo.keys():
            self.max_dynamic_patch = metainfo['max_dynamic_patch']
        else:
            self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if self.preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            # self.preprocessor = BUILDER.build(self.preprocessor)

        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.phi3:
            self.mllm.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.mllm.model.language_model)
        return

    def predict_video(self, pixel_values, text_prompts, **kwargs):
        ori_h, ori_w = kwargs['ori_height'], kwargs['ori_width']

        _input_ids = kwargs['input_ids']

        g_pixel_values = kwargs.pop('g_pixel_values', None)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])

        fast_pixel_values = kwargs.pop('fast_pixel_values', None)
        if fast_pixel_values is None:
            fast_token_idx = None
        else:
            fast_token_idx = self.fast_token_idx

        predictions = []
        pred_masks = []
        is_exists_list = []
        for input_ids in _input_ids:
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            pixel_values = pixel_values.to(dtype=self.torch_dtype)
            if fast_pixel_values is not None:
                fast_pixel_values = fast_pixel_values.to(dtype=self.torch_dtype)
            mm_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'fast_pixel_values': fast_pixel_values,
                'fast_token_idx': fast_token_idx,
            }
            if kwargs.get('image_grid_thw', None) is not None:
                mm_inputs['image_grid_thw'] = kwargs['image_grid_thw']

            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()

            # input_text = self.tokenizer.decode(mm_inputs['input_ids'][0], skip_special_tokens=False)
            # print(input_text, generate_output.sequences[0], '\n', predict, self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0])

            predictions.append(predict)

            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states, generate_output.sequences[0][:-1],
                seg_id=self.seg_token_idx
            )

            if len(seg_hidden_states) == 0:
                print("Warning, no [SEG] tokens !!!")
                pred_masks.append(torch.zeros((g_pixel_values.shape[0], ori_h, ori_w), dtype=torch.int))
                continue
            elif len(seg_hidden_states) > 1:
                print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                seg_hidden_states = seg_hidden_states[:1]
            seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

            seg_hidden_states = seg_hidden_states.to(dtype=torch.float32)

            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            # TODO: change 5
            if len(pixel_values) < 5:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * pixel_values.shape[0])
            else:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * 5)
            pred_mask = F.interpolate(
                pred_mask,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False,
            )
            pred_mask = pred_mask[:, 0]
            pred_mask = pred_mask.sigmoid() > 0.5
            pred_mask = pred_mask.int()
            # supervision
            if self.use_fast_supervision and (input_ids == self.fast_token_idx).sum() > 0:
                fast_flag = input_ids.squeeze(0) == self.fast_token_idx
                len_out = generate_output.sequences[0][:-1].shape[0]
                fast_tokens = last_hidden_states[:-len_out][fast_flag].to(dtype=torch.float32)
                exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
                is_exists = exists_logit.squeeze(-1).sigmoid() > 0.5
                is_exists_list.append(is_exists)
                not_exists = torch.logical_not(is_exists)
                if torch.any(not_exists):
                    pred_mask[not_exists] = pred_mask[not_exists] * 0

            pred_masks.append(pred_mask)
        assert len(pred_masks) == len(text_prompts)
        ret_dict = {
            'prediction': predictions,
            'prediction_masks': [mask_to_rle(_item.cpu().numpy()) for _item in pred_masks],
        }
        if 'id' in kwargs.keys():
            ret_dict['id'] = kwargs['id']

        if len(is_exists_list) > 0:
            ret_dict['is_exists'] = is_exists_list
        return ret_dict


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs


class VideoLLaVASAMModel_zero3(VideoLLaVASAMModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=['[SEG]', ],
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 arch_type='intern_vl',
                 # zero3
                 bs=1,
                 ):
        super(VideoLLaVASAMModel_zero3, self).__init__(
            mllm=mllm,
            tokenizer=tokenizer,
            grounding_encoder=grounding_encoder,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            torch_dtype=torch_dtype,
            pretrained_pth=pretrained_pth,
            frozen_sam2_decoder=frozen_sam2_decoder,
            special_tokens=special_tokens,
            loss_sample_points=loss_sample_points,
            num_points=num_points,
            # for slow fast arch
            fast_pool=fast_pool,
            fast_pool_size=fast_pool_size,
            arch_type=arch_type,
        )
        self.bs = bs
        # self.loss_boundary = BoundaryLoss()

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks
    

    import torch
    import torch.nn.functional as F
    from typing import List, Dict, Tuple

    import torch


    import torch
    import torch.nn.functional as F

    # def forward(self, data, data_samples=None, mode='loss', use_grpo=True, num_samples=4):
    #     """
        
    #     """
        
    #     if not use_grpo:
    #         return self._original_forward(data, data_samples, mode)
    #     else:
    #         return self._forward_grpo(data, data_samples, mode, num_samples)


    # def _original_forward(self, data, data_samples=None, mode='loss'):
    #     """
    #     """
    #     g_pixel_values = data.pop('g_pixel_values', None)
    #     gt_masks = data.pop('masks', None)
    #     frames_per_batch = data.pop('frames_per_batch', None)
    #     input_ids = data['input_ids']
    #     region_info = data["region_info"]

    #     if self.fast_pool:
    #         output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
    #     else:
    #         output = self.mllm(data, data_samples, mode)

    #     if gt_masks is None:
    #         seg_valid = False
    #         g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
    #             dtype=self.torch_dtype,
    #             device=input_ids.device,
    #         )
    #     else:
    #         seg_valid = True

    #     assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        
    #     ori_size_list = []
    #     for i_bs, mask in enumerate(gt_masks):
    #         mask_shape = mask.shape[-2:]
    #         ori_size_list += [mask_shape] * frames_per_batch[i_bs]

    #     seg_token_mask = input_ids == self.seg_token_idx

    #     hidden_states = output.hidden_states
    #     hidden_states = self.text_hidden_fcs(hidden_states[-1])

    #     _zero = hidden_states.mean() * 0.0
    #     if seg_valid:
    #         pred_embeddings = hidden_states[seg_token_mask] + _zero
    #     else:
    #         pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

    #     seg_token_counts = seg_token_mask.int().sum(-1)
    #     if not seg_valid:
    #         seg_token_counts += 5

    #     pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
    #     pred_embeddings_list = []
    #     for item in pred_embeddings_list_:
    #         if len(item) != 0:
    #             pred_embeddings_list.append(item)
    #     pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
    #         pred_embeddings_list, frames_per_batch)
    #     if not success:
    #         raise NotImplementedError

    #     gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
    #     pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
    #         pred_embeddings_list_video, gt_masks_video
    #     )
    #     g_pixel_values = torch.stack([
    #         self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
    #     ])
        
    #     num_objs = pred_embeddings_list_video[0].shape[0]
    #     num_frames = len(pred_embeddings_list_video)
    #     language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        
    #     sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
    #     pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

    #     gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
    #     gt_masks = torch.cat(gt_masks, dim=0)
    #     pred_masks = pred_masks.flatten(0, 1)

    #     bs = len(pred_masks)
    #     loss_mask, loss_dice = 0, 0
    #     if len(pred_masks) != len(gt_masks):
    #         print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
    #         min_num = min(len(pred_masks), len(gt_masks))
    #         pred_masks = pred_masks[:min_num]
    #         gt_masks = gt_masks[:min_num]
    #         seg_valid = False

    #     if self.loss_sample_points:
    #         sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
    #         sam_loss_dice = self.loss_dice(
    #             sampled_pred_mask,
    #             sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
    #         sam_loss_mask = self.loss_mask(
    #             sampled_pred_mask.reshape(-1),
    #             sampled_gt_mask.reshape(-1),
    #             avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
    #     else:
    #         sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
    #         sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
    #     loss_mask += sam_loss_mask
    #     loss_dice += sam_loss_dice

    #     if not seg_valid:
    #         _scale = 0.0
    #     else:
    #         _scale = 1.0
    #     loss_mask = loss_mask * _scale
    #     loss_dice = loss_dice * _scale

    #     loss_dict = {
    #         'loss_mask': loss_mask,
    #         'loss_dice': loss_dice,
    #         'llm_loss': output.loss,
    #     }
    #     return loss_dict






    def extract_answer_from_text(text):
        """
        A/B/C/D
        """
        patterns = [
            r'answer is ([A-D])',           # "The answer is D"
            r'?([A-D])',
            r'?([A-D])',
            r'\b([A-D])\s*[。.]?\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None


    def extract_ground_truth_answer(data_sample):
        """
        data_sample
        ground truth answer
        """
        if hasattr(data_sample, 'answer'):
            return data_sample.answer
        
        if hasattr(data_sample, 'conversations'):
            for conv in data_sample.conversations:
                if conv.get('from') == 'gpt':
                    text = conv.get('value', '')
                    return extract_answer_from_text(text)
        
        if hasattr(data_sample, 'metainfo'):
            return data_sample.metainfo.get('ground_truth_answer', None)
        
        return None


    # import torch
    # import torch.nn.functional as F

    # import torch
    # import torch.nn.functional as F
    # from typing import List, Tuple

    # import torch
    # import torch.nn.functional as F
    # from typing import List, Tuple, Optional

    # def _forward_grpo(self, data, data_samples, mode, num_samples=64):
    #     """
    #     """
    #     g_pixel_values_orig = data.get('g_pixel_values', None)
    #     gt_masks_orig = data.get('masks', None)
    #     frames_per_batch_orig = data.get('frames_per_batch', None)
    #     input_ids_orig = data['input_ids'].clone()
    #     labels = data.get('labels', None)
    #     if gt_masks_orig is None or labels is None:
    #         return self._original_forward(data, data_samples, mode)

    #     device = input_ids_orig.device
    #     B = input_ids_orig.shape[0]

    #     gt_answers = data.get('gt_answers', None)
    #     if gt_answers is None or len(gt_answers) != B:
    #         gt_answers = [None] * B

    #     use_top_p = 0.9                # nucleus sampling


    #     alpha = 0.4    # seg

    #     eps_std_warn = 1e-3
    #     eps_std_min = 5e-4

    #     # ----------------- Step1: teacher-forced logits -----------------
    #     data_for_logits = {
    #         'input_ids': input_ids_orig.clone(),
    #         'pixel_values': data.get('pixel_values', None),
    #         'attention_mask': data.get('attention_mask', None),
    #         'position_ids': data.get('position_ids', None),
    #         'region_info': data.get('region_info', None),
    #         'labels': labels,
    #     }
    #     if 'fast_pixel_values' in data:
    #         data_for_logits['fast_pixel_values'] = data['fast_pixel_values']

    #     if getattr(self, 'fast_pool', False):
    #         output = self.mllm(data_for_logits, data_samples, mode, fast_token_idx=self.fast_token_idx)
    #     else:
    #         output = self.mllm(data_for_logits, data_samples, mode)
    #     logits = output.logits  # [B, T, V]

    #     answer_mask = (labels != -100)  # [B, T]

    #     region_token_id = getattr(self.mllm.model, 'region_context_token_id', None)
    #     seg_token_id = getattr(self, 'seg_token_idx', None)
    #     img_context_token_id = getattr(self.mllm.model, 'img_context_token_id', None)

    #     def find_contiguous_spans(bool_tensor_1d: torch.Tensor) -> List[Tuple[int, int]]:
    #         idx = torch.nonzero(bool_tensor_1d, as_tuple=True)[0]
    #         spans: List[Tuple[int, int]] = []
    #         if idx.numel() == 0:
    #             return spans
    #         s = idx[0].item()
    #         prev = s
    #         for k in idx[1:].tolist():
    #             if k != prev + 1:
    #                 spans.append((s, prev + 1))
    #                 s = k
    #             prev = k
    #         spans.append((s, prev + 1))
    #         return spans

    #     answer_spans_list: List[List[Tuple[int, int]]] = []
    #     special_token_pos_sets: List[set] = []
    #     for b in range(B):
    #         spans = find_contiguous_spans(answer_mask[b])
    #         answer_spans_list.append(spans)
    #         special_positions = set()
    #         if spans:
    #             iid = input_ids_orig[b]
    #             for tid in [region_token_id, seg_token_id, img_context_token_id]:
    #                 if tid is None: 
    #                     continue
    #                 pos = torch.nonzero(iid == tid, as_tuple=True)[0]
    #                 for p in pos.tolist():
    #                     if any(s <= p < e for (s, e) in spans):
    #                         special_positions.add(p)
    #         special_token_pos_sets.append(special_positions)

    #     all_sampled_input_ids = []
    #     all_logprobs = []
    #     entropy_acc = torch.zeros((), device=device)
    #     entropy_steps = 0

    #     for si in range(num_samples):
    #         sampled_input_ids = input_ids_orig.clone()
    #         batch_logprobs = torch.zeros(B, device=device)

    #         temperature = max(0.7, base_temperature + temperature_jitter * ((si % 3) - 1))

    #         for b in range(B):
    #             spans = answer_spans_list[b]
    #             if not spans:
    #                 continue
    #             special_positions = special_token_pos_sets[b]

    #             total_logprob = torch.zeros((), device=device)
    #             valid_token_count = 0

    #             for (s, e) in spans:
    #                 L = e - s
    #                 if L <= 0:
    #                     continue

    #                 if s == 0:
    #                     use_start = 0
    #                     use_end = max(0, e - 1)
    #                     local_start = 1
    #                 else:
    #                     use_start = s - 1
    #                     use_end = e - 1
    #                     local_start = 0
    #                 if use_end <= use_start:
    #                     continue

    #                 seg_logits = logits[b, use_start:use_end]  # [L', V]

    #                 if gumbel_noise_scale and gumbel_noise_scale > 0:
    #                     u = torch.rand_like(seg_logits).clamp_(1e-12, 1 - 1e-12)
    #                     g = -torch.log(-torch.log(u))
    #                     seg_logits = seg_logits + gumbel_noise_scale * g

    #                 # softmax
    #                 probs = F.softmax(seg_logits / temperature, dim=-1)  # [L', V]

    #                 entropy_acc = entropy_acc + ent
    #                 entropy_steps += 1

    #                 # --- nucleus sampling ---
    #                 if use_top_p is not None:
    #                     sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    #                     cumsum = torch.cumsum(sorted_probs, dim=-1)
    #                     cutoff = (cumsum > use_top_p).float()
    #                     cutoff = torch.roll(cutoff, shifts=1, dims=-1)
    #                     cutoff[:, 0] = 0.0
    #                     keep = 1.0 - cutoff
    #                     filt_probs = sorted_probs * keep
    #                     filt_probs = filt_probs / (filt_probs.sum(dim=-1, keepdim=True) + 1e-12)
    #                     sampled_in_sorted = torch.multinomial(filt_probs, num_samples=1).squeeze(-1)  # [L']
    #                     sampled_tok = sorted_idx.gather(-1, sampled_in_sorted.unsqueeze(-1)).squeeze(-1)
    #                 else:
    #                     tk = min(int(use_top_k or probs.shape[-1]), probs.shape[-1])
    #                     topk_prob, topk_idx = torch.topk(probs, k=tk, dim=-1)
    #                     topk_prob = topk_prob / (topk_prob.sum(dim=-1, keepdim=True) + 1e-12)
    #                     sampled_idx_in_topk = torch.multinomial(topk_prob, num_samples=1).squeeze(-1)
    #                     sampled_tok = topk_idx.gather(-1, sampled_idx_in_topk.unsqueeze(-1)).squeeze(-1)

    #                 if epsilon_random and epsilon_random > 0:
    #                     rand_mask = (torch.rand(sampled_tok.shape, device=device) < epsilon_random)
    #                     if rand_mask.any():
    #                         V = probs.size(-1)
    #                         random_tok = torch.randint(low=0, high=V, size=sampled_tok.shape, device=device)
    #                         sampled_tok = torch.where(rand_mask, random_tok, sampled_tok)

    #                 for li in range(local_start, L):
    #                     gp = s + li
    #                     if gp in special_positions:
    #                         continue
    #                     row = li - 1 if s == 0 else li
    #                     if row < 0 or row >= sampled_tok.shape[0]:
    #                         continue

    #                     new_token = sampled_tok[row]
    #                     sampled_input_ids[b, gp] = new_token

    #                     lp_row = F.log_softmax(seg_logits[row] / temperature, dim=-1)
    #                     token_logprob = lp_row[new_token]
    #                     total_logprob = total_logprob + token_logprob
    #                     valid_token_count += 1

    #             if valid_token_count > 0:
    #                 batch_logprobs[b] = total_logprob / float(valid_token_count)

    #         all_sampled_input_ids.append(sampled_input_ids)
    #         all_logprobs.append(batch_logprobs)

    #     all_seg_rewards = []
    #     all_answer_rewards = []
    #     all_total_rewards = []

    #     base_data_keys = set(data.keys()) - {'input_ids', 'g_pixel_values', 'masks', 'frames_per_batch', 'labels'}

    #     letter_ids: Optional[List[int]] = None
    #     try:
    #         tok = self.mllm.tokenizer
    #         letter_ids = []
    #         for ch in ['A', 'B', 'C', 'D']:
    #             ids = tok.encode(ch, add_special_tokens=False)
    #             letter_ids.append(ids[0] if len(ids) >= 1 else None)
    #         if any(x is None for x in letter_ids):
    #             letter_ids = None
    #     except Exception:
    #         letter_ids = None

    #     def _extract_letter_ad(text: str):
    #         if text is None:
    #             return None
    #         s = str(text)
    #         import re
    #         for p in [
    #         ]:
    #             m_iter = list(re.finditer(p, s, flags=re.IGNORECASE))
    #             if m_iter:
    #                 return m_iter[-1].group(1).upper()
    #         m_iter = list(re.finditer(r'\b([A-D])\b', s, flags=re.IGNORECASE))
    #         if m_iter:
    #             return m_iter[-1].group(1).upper()
    #         return None

    #     valid_samples = []
    #     for si, sampled_ids in enumerate(all_sampled_input_ids):
    #         try:
    #             sampled_data = {
    #                 'input_ids': sampled_ids,
    #                 'g_pixel_values': g_pixel_values_orig,
    #                 'masks': gt_masks_orig,
    #                 'frames_per_batch': frames_per_batch_orig,
    #             }
    #             for k in base_data_keys:
    #                 sampled_data[k] = data[k]

    #             with torch.no_grad():
    #                 out = self._original_forward(sampled_data, data_samples, mode='loss')

    #             loss_mask = out['loss_mask'].detach()
    #             loss_dice = out['loss_dice'].detach()
    #             total_loss = loss_mask + loss_dice
    #             seg_reward = 1.0 / (1.0 + total_loss)
    #             if seg_reward.dim() == 0:
    #                 seg_reward = seg_reward.unsqueeze(0).expand(B)

    #             ans_reward = torch.zeros(B, device=device)
    #             for b in range(B):
    #                 gt = gt_answers[b]
    #                 if gt is None:
    #                     ans_reward[b] = 0.0
    #                     continue
    #                 spans = answer_spans_list[b]
    #                 if spans:
    #                     ids_list = []
    #                     row_ids = sampled_ids[b]
    #                     for (s, e) in spans:
    #                         ids_list.extend(row_ids[s:e].tolist())
    #                     gen_ids = ids_list
    #                 else:
    #                     gen_ids = sampled_ids[b].tolist()

    #                 used_soft = False
    #                 if letter_ids is not None and spans:
    #                     s_last, e_last = spans[-1]
    #                     if s_last == 0:
    #                     else:
    #                         t_row = e_last - 1
    #                     if 0 <= t_row < logits.size(1):
    #                         cand_logits = logits[b, t_row, letter_ids]  # [4]
    #                         cand_probs = F.softmax(cand_logits, dim=-1)
    #                         gt_letter = str(gt).strip().upper()
    #                         idx = "ABCD".find(gt_letter)
    #                         if idx != -1:
    #                             ans_reward[b] = cand_probs[idx].detach()
    #                             used_soft = True

    #                 if not used_soft:
    #                     gen_text = self.mllm.tokenizer.decode(gen_ids, skip_special_tokens=True)
    #                     pred_letter = _extract_letter_ad(gen_text)
    #                     gt_letter = str(gt).strip().upper()
    #                     if pred_letter is None or len(str(pred_letter).strip()) == 0:
    #                         ans_reward[b] = -0.5
    #                     else:
    #                         ans_reward[b] = 1.0 if pred_letter == gt_letter else 0.0

    #             total_reward = alpha * seg_reward + beta * ans_reward
    #             all_seg_rewards.append(seg_reward)
    #             all_answer_rewards.append(ans_reward)
    #             all_total_rewards.append(total_reward)
    #             valid_samples.append(si)

    #         except Exception as e:
    #             print(f"⚠️ Sample {si} failed: {e}")
    #             import traceback; traceback.print_exc()
    #             continue

    #     if len(valid_samples) < 1:
    #         print("⚠️ GRPO: no valid samples this step; fallback original forward")
    #         return self._original_forward(data, data_samples, mode)

    #     all_total_rewards = torch.stack(all_total_rewards)                 # [S', B]
    #     logprobs = torch.stack([all_logprobs[i] for i in valid_samples])  # [S', B]

    #     mean_reward = all_total_rewards.mean(dim=0)  # [B]
    #     std_reward = all_total_rewards.std(dim=0)    # [B]

    #     if torch.isfinite(std_reward.mean()) and std_reward.mean() < eps_std_warn:

    #     low_var_mask = (std_reward < eps_std_min)
    #     advantages = (all_total_rewards - mean_reward) / (std_reward.clamp_min(eps_std_min))
    #     advantages[:, low_var_mask] = (all_total_rewards[:, low_var_mask] - mean_reward[low_var_mask])

    #     advantages = torch.nan_to_num(advantages, nan=0.0, posinf=5.0, neginf=-5.0)
    #     advantages = torch.clamp(advantages, -5.0, 5.0)

    #     # policy loss
    #     policy_loss = -(advantages * logprobs).mean()

    #     if entropy_steps > 0 and entropy_bonus_weight > 0:
    #         avg_entropy = entropy_acc / float(entropy_steps)
    #         policy_loss = policy_loss - entropy_bonus_weight * avg_entropy

    #     llm_loss = output.loss if getattr(output, 'loss', None) is not None else torch.tensor(0.0, device=device)
    #     llm_weight = 0.5 if llm_loss < 2.0 else 0.1

    #     total_loss = policy_loss + llm_weight * llm_loss

    #     def _stat_tensor(x, device):
    #         if isinstance(x, torch.Tensor):
    #             return x.detach().to(device=device, dtype=torch.float32)
    #         if isinstance(x, (int, float)):
    #             return torch.tensor(x, device=device, dtype=torch.float32)
    #         try:
    #             return torch.tensor(float(x), device=device, dtype=torch.float32)
    #         except Exception:
    #             return torch.tensor(0.0, device=device, dtype=torch.float32)

    #     seg_rewards_tensor     = torch.stack(all_seg_rewards)          # [S', B]
    #     answer_rewards_tensor  = torch.stack(all_answer_rewards)       # [S', B]
    #     num_valid_samples_t    = _stat_tensor(len(valid_samples), device)
    #     low_var_batches_t      = low_var_mask.sum().to(device=device, dtype=torch.float32)

    #     loss_dict = {
    #         'loss': total_loss,


    #         'mean_reward':     _stat_tensor(mean_reward.mean(), device),
    #         'std_reward':      _stat_tensor(std_reward.mean(), device),
    #         'max_reward':      _stat_tensor(all_total_rewards.max(), device),
    #         'min_reward':      _stat_tensor(all_total_rewards.min(), device),

    #         'mean_seg_reward': _stat_tensor(seg_rewards_tensor.mean(), device),
    #         'mean_ans_reward': _stat_tensor(answer_rewards_tensor.mean(), device),
    #         'max_seg_reward':  _stat_tensor(seg_rewards_tensor.max(), device),
    #         'max_ans_reward':  _stat_tensor(answer_rewards_tensor.max(), device),

    #         'num_valid_samples':   num_valid_samples_t,
    #         'low_variance_batches': low_var_batches_t,
    #     }
    #     return loss_dict






    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']

        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError
            # return {'llm_loss': output.loss, 'loss_mask': output.loss * 0.0, 'loss_dice': output.loss * 0.0}

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        # print("2222",pred_embeddings_list_video[0].shape[0])
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        # print(f"Done, {g_pixel_values.device} !!!\n\n")
        num_objs = pred_embeddings_list_video[0].shape[0]
        # print("11111",num_objs)
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        # print(f"Done, {g_pixel_values.device} !!! {num_frames}---{num_objs}, {language_embeddings.shape}\n\n")
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)
        # pred_masks = torch.cat(pred_masks, dim=0)


        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        # boundary_loss = self.loss_boundary(pred_masks.sigmoid(), gt_masks.float())
        # boundary_loss = edge_loss(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale



        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            # 'loss_boundary': boundary_loss,  
            'llm_loss': output.loss,
        }
        return loss_dict
