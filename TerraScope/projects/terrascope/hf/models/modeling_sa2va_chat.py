# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria

from .configuration_sa2va_chat import Sa2VAChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from .sam2 import SAM2
from .templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F

try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)


import torch

def build_region_embeds_from_indices(
    vit_embeds_flat: torch.Tensor,   # [N_ctx, C]
    region_info: dict,               # {'visual_token_indices': [...], 'num_tokens': K}
    *, 
    batchify: bool = True
) -> torch.Tensor:
    """
     region_info  'visual_token_indices' vit_embeds_flat 
    -  & 
    - /
    """
    if vit_embeds_flat is None:
        raise ValueError("vit_embeds_flat is None")
    if 'visual_token_indices' not in region_info:
        raise KeyError("region_info  'visual_token_indices'")

    idx_list = region_info['visual_token_indices']
    K = region_info.get('num_tokens', len(idx_list))

    if len(idx_list) != K:
        raise ValueError(f"num_tokens({K})  indices ({len(idx_list)}) ")

    idx = torch.as_tensor(idx_list, dtype=torch.long, device=vit_embeds_flat.device)

    N_ctx, C = vit_embeds_flat.shape
    if (idx < 0).any() or (idx >= N_ctx).any():
        bad = idx[(idx < 0) | (idx >= N_ctx)]
        raise IndexError(f": {bad.tolist()} 0..{N_ctx-1}")

    region_embeds = vit_embeds_flat.index_select(0, idx)   # [K, C]

    if batchify:
        region_embeds = region_embeds.unsqueeze(0)         # [1, K, C]
    return region_embeds



def simplify_text_for_print(text, max_repeats=3):
    """token"""
    import re
    
    text = re.sub(
        r'(<IMG_CONTEXT>){4,}',
        lambda m: f'<IMG_CONTEXT>×{len(m.group(0))//13}',
        text
    )
    
    text = re.sub(
        r'(<REGION>){4,}',
        lambda m: f'<REGION>×{len(m.group(0))//8}',
        text
    )
    
    return text

def simplify_ids_for_print(ids, special_tokens_to_compress=None):
    """token"""
    if special_tokens_to_compress is None:
        special_tokens_to_compress = [151667, 151677]  # IMG_CONTEXT, <REGION>
    
    result = []
    i = 0
    while i < len(ids):
        token = ids[i]
        
        if token in special_tokens_to_compress:
            count = 1
            while i + count < len(ids) and ids[i + count] == token:
                count += 1
            
            if count > 5:
                result.extend([token] * 3)
                result.append(f"...×{count-6}...")
                result.extend([token] * 3)
            else:
                result.extend([token] * count)
            
            i += count
        else:
            result.append(token)
            i += 1
    
    return result



def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModel(PreTrainedModel):
    config_class = Sa2VAChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: Sa2VAChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.template = self.template.replace('-', '_')
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                print("0000000")
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.init_prediction_config = False

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        if 'vp_overall_mask' not in data.keys():
            vp_overall_mask = None
        else:
            vp_overall_mask = data['vp_overall_mask']

        if 'prompt_masks' in data.keys():
            prompt_masks = data['prompt_masks']
        else:
            prompt_masks = None

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            vp_overall_mask=None,
            prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
        fast_vit_embeds = None

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token]
        else:
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    # @torch.no_grad()
    # def generate(
    #         self,
    #         pixel_values: Optional[torch.FloatTensor] = None,
    #         input_ids: Optional[torch.FloatTensor] = None,
    #         attention_mask: Optional[torch.LongTensor] = None,
    #         visual_features: Optional[torch.FloatTensor] = None,
    #         generation_config: Optional[GenerationConfig] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    #         prompt_masks=None,
    #         vp_overall_mask=None,
    #         **generate_kwargs,
    # ) -> torch.LongTensor:
    #     device = self.device
    #     assert self.img_context_token_id is not None

    #     if pixel_values is not None:
    #         if visual_features is not None:
    #             vit_embeds = visual_features
    #         else:
    #             if type(pixel_values) is list or pixel_values.ndim == 5:
    #                 if type(pixel_values) is list:
    #                     pixel_values = [
    #                         x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
    #                     ]
    #                 # b*n, c, h, w
    #                 pixel_values = torch.cat(
    #                     [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)

    #             vit_embeds = self.extract_feature(pixel_values.to(device))
    #         image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
    #         image_flags = image_flags.long()
    #         vit_embeds = vit_embeds[image_flags == 1]

    #         input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
    #         B, N, C = input_embeds.shape
    #         input_embeds = input_embeds.reshape(B * N, C)

    #         if vp_overall_mask is not None and prompt_masks is not None:
    #             vp_embeds = []
    #             vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
    #             prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

    #             vp_overall_mask = vp_overall_mask[image_flags == 1]
    #             overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

    #             i_vp_img = 0
    #             for i_img in range(len(vit_embeds)):
    #                 vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
    #                 if vp_overall_mask[i_img]:
    #                     tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
    #                     objects_prompt_masks = prompt_masks[i_vp_img]
    #                     n_obj = len(objects_prompt_masks)
    #                     tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
    #                     objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
    #                     vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
    #                     i_vp_img += 1

    #             vp_embeds = torch.cat(vp_embeds, dim=0)
    #         else:
    #             vp_embeds = None

    #         input_ids = input_ids.reshape(B * N)
    #         selected = (input_ids == self.img_context_token_id)
    #         assert selected.sum() != 0
    #         if vp_embeds is None:
    #             input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
    #         else:
    #             if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
    #                 print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
    #                       .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
    #                 min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
    #                 input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
    #             else:
    #                 input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

    #         input_embeds = input_embeds.reshape(B, N, C)
    #     else:
    #         input_embeds = self.language_model.get_input_embeddings()(input_ids)

    #     outputs = self.language_model.generate(
    #         inputs_embeds=input_embeds,
    #         attention_mask=attention_mask.to(device),
    #         generation_config=generation_config,
    #         output_hidden_states=output_hidden_states,
    #         # return_dict=return_dict,
    #         use_cache=True,
    #         **generate_kwargs,
    #     )
    #     return outputs

    def build_inputs_embeds_for_prefill(self, input_ids, vit_embeds_flat, img_context_token_id):
        device = input_ids.device
        tok_emb = self.language_model.get_input_embeddings()
        inputs_embeds = tok_emb(input_ids.to(device))   # [B, L, C]

        if vit_embeds_flat is None:
            return inputs_embeds

        B, L, C = inputs_embeds.shape
        flat_embeds = inputs_embeds.view(B*L, C)
        flat_ids    = input_ids.view(B*L)

        mask = (flat_ids == img_context_token_id)       # True at IMG_CONTEXT positions
        k = mask.sum().item()
        if k != vit_embeds_flat.shape[0]:
            raise ValueError(f"IMG_CONTEXT ...: tokens={k}, vit_embeds={vit_embeds_flat.shape[0]}")

        flat_embeds = flat_embeds.clone()
        flat_embeds[mask] = vit_embeds_flat.to(flat_embeds)
        return flat_embeds.view(B, L, C)



    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            position_ids=None,
            cache_position=None,
            cached_vit_embeds_flat: Optional[torch.FloatTensor] = None,
            cached_vit_embeds_3d: Optional[torch.FloatTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prompt_masks=None,
            vp_overall_mask=None,
            region_info=None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        print("position_dis",position_ids)
        print("cache_position",cache_position)

        device = self.device
        assert self.img_context_token_id is not None
        
        vit_embeds_3d, vit_embeds_flat = None, None

        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values.to(device))
            vit_embeds_3d = vit_embeds
            vit_embeds_flat = vit_embeds_3d.reshape(-1, vit_embeds_3d.shape[-1])
        elif cached_vit_embeds_flat is not None:
            vit_embeds_flat = cached_vit_embeds_flat
            vit_embeds_3d = cached_vit_embeds_3d
        
        input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        
        if vit_embeds_flat is not None:
            selected = (input_ids_flat == self.img_context_token_id)
            if selected.any():
                input_embeds[selected] = vit_embeds_flat.to(input_embeds.device)
        
        if region_info is not None and len(region_info) > 0 and vit_embeds_flat is not None:
            region_token_id = self.tokenizer.convert_tokens_to_ids('<REGION>')
            
            for batch_idx in range(B):
                if batch_idx >= len(region_info):
                    continue
                
                sample_region_info = region_info[batch_idx]
                if sample_region_info is None or len(sample_region_info) == 0:
                    continue
                
                sample_start_idx = batch_idx * N
                sample_end_idx = sample_start_idx + N
                sample_input_ids = input_ids_flat[sample_start_idx:sample_end_idx]
                
                region_positions = (sample_input_ids == region_token_id).nonzero(as_tuple=True)[0]
                
                if len(region_positions) == 0:
                    continue
                
                region_idx = 0
                
                # print("sample",sample_region_info)
                for obj_idx, obj_info in enumerate(sample_region_info):
                    num_tokens = obj_info['num_tokens']
                    visual_indices = obj_info['visual_token_indices']
                    
                    if region_idx + num_tokens > len(region_positions):
                        break
                    
                    try:
                        region_features = vit_embeds_flat[visual_indices]
                        
                        for i in range(num_tokens):
                            global_pos = sample_start_idx + region_positions[region_idx + i]
                            input_embeds[global_pos] = region_features[i]
                        
                        region_idx += num_tokens
                        
                    except Exception as e:
                        print(f"⚠️  Error processing region info: {e}")
                        continue
        
        input_embeds = input_embeds.reshape(B, N, C)
        
        if past_key_values is not None and attention_mask is not None:
            past_length = past_key_values[0][0].shape[2]
            print("past_length",past_length)
            if attention_mask.shape[1] == N:
                past_mask = torch.ones((B, past_length), dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([past_mask, attention_mask], dim=1)
                print("new_generated_mask",attention_mask.shape)

        use_cache = generate_kwargs.pop('use_cache', True)
        return_dict_in_generate = generate_kwargs.pop('return_dict_in_generate', True)

        if output_hidden_states is None:
            output_hidden_states = generate_kwargs.pop('output_hidden_states', False)

        custom_params = {
            'g_pixel_values', 
            'region_info', 
            'prompt_masks', 
            'vp_overall_mask', 
            'cached_vit_embeds_flat',
            'cached_vit_embeds_3d',
            'past_key_values',
        }

        filtered_kwargs = {
            k: v for k, v in generate_kwargs.items() 
            if k not in custom_params
        }

        print("@@@@@@@@@",input_embeds.shape,attention_mask.shape)
        if past_key_values:
            attention_mask = None
            print("!!!!!!!!!!!")
            print(len(past_key_values))
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **filtered_kwargs,
        )
        
        return outputs, vit_embeds_3d, vit_embeds_flat


    def _spatial_uniform_sample(
        self,
        token_indices,
        mask_resized,
        total_tokens_width,
        total_tokens_height,
        tokens_per_side,
        n_patch_cols,
        target_count
    ):
        """
        Spatially uniform sampling from selected tokens
        
        Strategy: divide mask region into grid, sample from each grid cell
        """
        token_positions = []
        for token_idx in token_indices:
            patch_idx = token_idx // (tokens_per_side * tokens_per_side)
            token_in_patch = token_idx % (tokens_per_side * tokens_per_side)
            
            patch_row = patch_idx // n_patch_cols
            patch_col = patch_idx % n_patch_cols
            
            local_row = token_in_patch // tokens_per_side
            local_col = token_in_patch % tokens_per_side
            
            global_row = patch_row * tokens_per_side + local_row
            global_col = patch_col * tokens_per_side + local_col
            
            token_positions.append((global_row, global_col, token_idx))
        
        mask_rows, mask_cols = np.where(mask_resized)
        if len(mask_rows) == 0:
            return token_indices[:target_count]
        
        min_row, max_row = mask_rows.min(), mask_rows.max()
        min_col, max_col = mask_cols.min(), mask_cols.max()
        
        grid_size = int(np.ceil(np.sqrt(target_count)))
        row_step = max(1, (max_row - min_row + 1) / grid_size)
        col_step = max(1, (max_col - min_col + 1) / grid_size)
        
        sampled = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell_row_min = min_row + i * row_step
                cell_row_max = min_row + (i + 1) * row_step
                cell_col_min = min_col + j * col_step
                cell_col_max = min_col + (j + 1) * col_step
                
                cell_tokens = [
                    (r, c, idx) for r, c, idx in token_positions
                    if cell_row_min <= r < cell_row_max and cell_col_min <= c < cell_col_max
                ]
                
                if cell_tokens:
                    cell_center_r = (cell_row_min + cell_row_max) / 2
                    cell_center_c = (cell_col_min + cell_col_max) / 2
                    
                    closest = min(cell_tokens, 
                                key=lambda x: (x[0] - cell_center_r)**2 + (x[1] - cell_center_c)**2)
                    sampled.append(closest[2])
                
                if len(sampled) >= target_count:
                    break
            if len(sampled) >= target_count:
                break
        
        return sampled[:target_count]

    def _extract_region_info_from_mask(
        self,
        mask,
        vit_embeds_flat,
        pixel_values,
        ori_height,
        ori_width,
        max_tokens=64
    ):
        """
        maskvisual token indices
        🎯 mask
        """
        visual_indices = self._get_visual_tokens_from_mask_accurate(
            mask=mask,
            ori_height=ori_height,
            ori_width=ori_width,
            pixel_values=pixel_values,
            max_tokens_per_object=max_tokens
        )
        
        return {
            'visual_token_indices': visual_indices,
            'num_tokens': len(visual_indices)
        }
        

    def _get_visual_tokens_from_mask_multi_simple(
        self,
        mask,
        ori_height,
        ori_width,
        max_tokens_per_object=64
    ):
        """
        Simple version for multi-image inference:
        - Image is resized to 448x448
        - Single patch (no dynamic preprocessing)
        - For multi-image inference scenarios
        """
        tokens_per_side = int(np.sqrt(self.patch_token))

        # Step 1: Resize mask to 448x448 (same as image preprocessing)
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        mask_resized = np.array(
            Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size),
                Image.NEAREST
            )
        ) > 127

        # Step 2: Map mask to token grid
        mask_on_token_grid = np.array(
            Image.fromarray((mask_resized * 255).astype(np.uint8)).resize(
                (tokens_per_side, tokens_per_side),
                Image.NEAREST
            )
        ) > 127

        # Step 3: Find selected tokens
        selected_tokens = []
        for row in range(tokens_per_side):
            for col in range(tokens_per_side):
                if mask_on_token_grid[row, col]:
                    token_idx = row * tokens_per_side + col
                    selected_tokens.append(token_idx)

        # Step 4: Sample if too many tokens
        if len(selected_tokens) > max_tokens_per_object:
            selected_tokens = self._spatial_uniform_sample(
                selected_tokens,
                mask_on_token_grid,
                tokens_per_side,
                tokens_per_side,
                tokens_per_side,
                1,
                max_tokens_per_object
            )

        # Step 5: Ensure at least 1 token
        if len(selected_tokens) == 0:
            y_indices, x_indices = np.where(mask_on_token_grid)
            if len(y_indices) > 0:
                center_y = int(y_indices.mean())
                center_x = int(x_indices.mean())
                token_idx = center_y * tokens_per_side + center_x
                selected_tokens = [token_idx]
            else:
                selected_tokens = [0]

        return selected_tokens


    def _extract_region_info_from_mask_multi(
        self,
        masks,
        ori_heights,
        ori_widths,
        pixel_values,
        max_tokens=64
    ):
        """
        masksvisual token indices - 

        Args:
            masks: (2, H, W) - 2masks (prepost)
            ori_heights: [pre_height, post_height]
            ori_widths: [pre_width, post_width]
            pixel_values: (2, 3, 448, 448) - 
            max_tokens: int - masktoken

        Returns:
            dict: {
                'visual_token_indices': List[int] - token
                'num_tokens': int - token
            }
        """
        num_objects = masks.shape[0]

        num_pre_masks = num_objects // 2
        num_post_masks = num_objects - num_pre_masks

        tokens_per_image = self.patch_token

        all_visual_indices = []

        for obj_idx in range(num_pre_masks):
            mask = masks[obj_idx]

            pre_visual_token_indices = self._get_visual_tokens_from_mask_multi_simple(
                mask=mask,
                ori_height=ori_heights[0],
                ori_width=ori_widths[0],
                max_tokens_per_object=max_tokens
            )

            all_visual_indices.extend(pre_visual_token_indices)

        for obj_idx in range(num_post_masks):
            mask = masks[num_pre_masks + obj_idx]

            post_visual_token_indices = self._get_visual_tokens_from_mask_multi_simple(
                mask=mask,
                ori_height=ori_heights[1],
                ori_width=ori_widths[1],
                max_tokens_per_object=max_tokens
            )

            post_visual_token_indices_global = [
                idx + tokens_per_image for idx in post_visual_token_indices
            ]

            all_visual_indices.extend(post_visual_token_indices_global)

        return {
            'visual_token_indices': all_visual_indices,
            'num_tokens': len(all_visual_indices)
        }


    def _get_visual_tokens_from_mask_accurate(
        self,
        mask,
        ori_height,
        ori_width,
        pixel_values,
        max_tokens_per_object=64
    ):
        """
        InternVL
        
        1. resizetarget_width x target_height
        2. blocks
        3. maskresize
        """
        tokens_per_patch = self.patch_token
        tokens_per_side = int(np.sqrt(tokens_per_patch))
        
        patch_layout, target_aspect_ratio, target_width, target_height = \
            self._get_patch_layout_accurate(
                pixel_values.shape[0], 
                ori_height, 
                ori_width,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size
            )
        
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        
        mask_resized_to_target = np.array(
            Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (target_width, target_height), 
                Image.NEAREST
            )
        ) > 127
        
        n_cols = target_aspect_ratio[0]
        n_rows = target_aspect_ratio[1]
        
        total_tokens_width = n_cols * tokens_per_side
        total_tokens_height = n_rows * tokens_per_side
        
        mask_on_token_grid = np.array(
            Image.fromarray((mask_resized_to_target * 255).astype(np.uint8)).resize(
                (total_tokens_width, total_tokens_height),
                Image.NEAREST
            )
        ) > 127
        
        selected_tokens = []
        
        for patch_info in patch_layout:
            patch_idx = patch_info['patch_idx']
            
            patch_col = patch_idx % n_cols
            patch_row = patch_idx // n_cols
            
            token_col_start = patch_col * tokens_per_side
            token_col_end = (patch_col + 1) * tokens_per_side
            token_row_start = patch_row * tokens_per_side
            token_row_end = (patch_row + 1) * tokens_per_side
            
            patch_start_token = patch_idx * tokens_per_patch
            
            for local_row in range(tokens_per_side):
                for local_col in range(tokens_per_side):
                    global_token_row = token_row_start + local_row
                    global_token_col = token_col_start + local_col
                    
                    if (global_token_row < total_tokens_height and 
                        global_token_col < total_tokens_width and
                        mask_on_token_grid[global_token_row, global_token_col]):
                        
                        token_idx_in_patch = local_row * tokens_per_side + local_col
                        global_token_idx = patch_start_token + token_idx_in_patch
                        selected_tokens.append(global_token_idx)
        
        if len(selected_tokens) > max_tokens_per_object:
            selected_tokens = self._spatial_uniform_sample(
                selected_tokens,
                mask_on_token_grid,
                total_tokens_width,
                total_tokens_height,
                tokens_per_side,
                n_cols,
                max_tokens_per_object
            )
        
        if len(selected_tokens) == 0:
            y_indices, x_indices = np.where(mask_on_token_grid)
            if len(y_indices) > 0:
                center_y = int(y_indices.mean())
                center_x = int(x_indices.mean())
                
                patch_row = center_y // tokens_per_side
                patch_col = center_x // tokens_per_side
                patch_idx = patch_row * n_cols + patch_col
                
                if patch_idx < len(patch_layout):
                    local_row = center_y % tokens_per_side
                    local_col = center_x % tokens_per_side
                    token_idx = patch_idx * tokens_per_patch + local_row * tokens_per_side + local_col
                    selected_tokens = [token_idx]
                else:
                    selected_tokens = [0]
            else:
                selected_tokens = [0]
        
        return selected_tokens

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.bot_name = 'BOT'
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
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

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            self.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.language_model)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.img_context_token_id = img_context_token_id
        self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
        return
    



    @torch.no_grad()
    def predict_forward_with_grounding(
        self,
        image=None,
        video=None,
        text=None,
        past_text='',
        mask_prompts=None,
        tokenizer=None,
        max_tokens_per_seg=64,
        max_iterations=20,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        device = torch.device("cuda")
        self.gen_config.max_length = 8192

        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = self.template['INSTRUCTION'].format(input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = torch.tensor(self.tokenizer.encode(input_text), device=device).unsqueeze(0)
            attn = torch.ones_like(ids, dtype=torch.long)

            # prefill
            L0 = ids.shape[1]
            cache_pos0 = torch.arange(0, L0, device=device)
            out = self.language_model(
                input_ids=ids, attention_mask=attn,
                past_key_values=None, use_cache=True,
                cache_position=cache_pos0,
                output_hidden_states=True, return_dict=True,
            )
            past_kv = out.past_key_values
            past_len = L0
            all_gen = []

            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
            while True:
                step_attn = torch.ones_like(next_id, dtype=torch.long)
                step_pos  = torch.arange(past_len, past_len + 1, device=device)
                out = self.language_model(
                    input_ids=next_id, attention_mask=step_attn,
                    past_key_values=past_kv, use_cache=True,
                    cache_position=step_pos,
                    output_hidden_states=True, return_dict=True,
                )
                past_kv = out.past_key_values
                past_len += 1

                tok = next_id.item()
                all_gen.append(tok)
                if tok == self.tokenizer.eos_token_id or len(all_gen) >= self.gen_config.max_length:
                    break
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            predict = self.tokenizer.decode(all_gen, skip_special_tokens=False).strip()
            return {'prediction': predict, 'prediction_masks': []}

        if video is not None:
            pixel_values, extra_pixel_values = [], []
            ori_image_size = video[0].size
            for frame_idx, frame_image in enumerate(video):
                assert ori_image_size == frame_image.size
                g_image = np.array(frame_image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_image)
                if frame_idx < 5:
                    img = self.transformer(frame_image)
                    pixel_values.append(img)

            pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype).to(device)
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(self.torch_dtype).to(device)

            num_image_tokens = self.patch_token
            num_frames = len(pixel_values)
            ori_height, ori_width = ori_image_size[1], ori_image_size[0]
        else:
            ori_image_size = image.size
            ori_width, ori_height = ori_image_size

            g_image = np.array(image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype).to(device)
            extra_pixel_values = [g_pixel_values]
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(self.torch_dtype).to(device)

            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)

            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(self.torch_dtype).to(device)
            num_image_tokens = pixel_values.shape[0] * self.patch_token
            num_frames = 1

        image_token_str = (
            f'{self.IMG_START_TOKEN}'
            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}'
            f'{self.IMG_END_TOKEN}'
        )
        image_token_str = (image_token_str + '\n') * num_frames
        image_token_str = image_token_str.strip()

        text = text.replace('<image>', image_token_str)
        input_text = self.template['INSTRUCTION'].format(input=text, round=1, bot_name=self.bot_name)
        input_text = past_text + input_text
        initial_ids = torch.tensor(self.tokenizer.encode(input_text), device=device).unsqueeze(0)  # [1, L0]

        past_kv = None
        past_len = 0
        ret_masks = []
        all_region_infos = []
        seg_token_id = self.seg_token_idx
        region_token_id = self.tokenizer.convert_tokens_to_ids('<REGION>')
        eos_token_id = self.tokenizer.eos_token_id
        all_generated_ids = []
        interleave_cache = {}

     
        with torch.no_grad():
            vit_embeds = self.extract_feature(pixel_values.to(device))
            vit_embeds_3d = vit_embeds
            vit_embeds_flat = vit_embeds_3d.reshape(-1, vit_embeds_3d.shape[-1])

        print("\nStarting iterative generation with KV cache...")

        for iteration in range(1, max_iterations + 1):
            print(f"\nIteration {iteration}:")

            if iteration == 1:
                inputs_embeds0 = self.build_inputs_embeds_for_prefill(
                initial_ids, vit_embeds_flat, self.img_context_token_id
                )  # [1, L0, C]
                L0 = initial_ids.shape[1]
                attn0 = torch.ones((1, L0), dtype=torch.long, device=device)
                cache_pos0 = torch.arange(0, L0, device=device)
                prefill_out = self.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds0,
                    attention_mask=attn0,
                    past_key_values=None,
                    use_cache=True,
                    cache_position=cache_pos0,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_kv = prefill_out.past_key_values
                past_len = L0

                next_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
            else:
                if next_id is None:
                    logits_last = last_out.logits[:, -1, :]
                    next_id = logits_last.argmax(dim=-1, keepdim=True)

            seg_hit = False
            while True:
                step_attn = torch.ones_like(next_id, dtype=torch.long)          # [1,1]
                step_pos  = torch.arange(past_len, past_len + 1, device=device) # [1]
                out = self.language_model(
                    input_ids=next_id,
                    attention_mask=step_attn,
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_position=step_pos,
                    # interleave_inf=True,
                    # interleave_cache=interleave_cache,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
                past_len += 1

                tok = next_id.item()
                all_generated_ids.append(tok)

                if tok in self.tokenizer.all_special_ids:
                    pass
                    # print(f"{self.tokenizer.convert_ids_to_tokens(tok)}")
                else:
                    piece = self.tokenizer.decode([tok], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    # print(f"{repr(piece)}", end="", flush=True)

                if tok == eos_token_id or tok==151682 :
                    print("✅ EOS detected, generation complete")
                    final_prediction = self.tokenizer.decode(all_generated_ids, skip_special_tokens=False).strip()
                    print(f"\n✅ Generation complete:\n   Total iterations: {iteration}\n   Total [SEG] masks: {len(ret_masks)}\n   Total tokens: {len(all_generated_ids)}")
                    return {'prediction': final_prediction, 'prediction_masks': ret_masks}

                if tok == seg_token_id:
                    print("🎭 [SEG] detected, predicting mask...")
                    last_layer_h = out.hidden_states[-1]      # [B, L, C]
                    seg_hidden   = last_layer_h[0, -1, :].unsqueeze(0)     # [C]
                    seg_hidden = self.text_hidden_fcs(seg_hidden)

                    with torch.no_grad():
                        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
                        pred_masks = self.grounding_encoder.language_embd_inference(
                            sam_states, [seg_hidden] * num_frames
                        )
                        masks = F.interpolate(pred_masks, size=(ori_height, ori_width),
                                            mode='bilinear', align_corners=False)
                        masks = masks[:, 0].sigmoid()
                        predicted_mask = (masks > 0.5).squeeze(0)

                    coverage = predicted_mask.sum().item() / predicted_mask.numel() * 100
                    # print(f"   Mask coverage: {coverage:.2f}%")
                    ret_masks.append(predicted_mask.cpu().numpy())

                    if getattr(self, "grounding_encoder", None) is None:
                        cached_vit_embeds_flat = None
                    else:
                        cached_vit_embeds_flat = None

                    region_info = self._extract_region_info_from_mask(
                        mask=predicted_mask,
                        vit_embeds_flat=cached_vit_embeds_flat,
                        pixel_values=pixel_values,
                        ori_height=ori_height,
                        ori_width=ori_width,
                        max_tokens=8,
                    )
                    all_region_infos.append(region_info)
                    num_tokens = region_info['num_tokens']
                    # print(f"   ✅ Sampled {num_tokens} visual tokens")
                    # print(region_info)

                    region_ids = torch.full((1, num_tokens), region_token_id, dtype=torch.long, device=device)
                    region_attn = torch.ones(1, past_len + num_tokens, dtype=torch.long, device=device)
                    region_pos  = torch.arange(past_len, past_len + num_tokens, device=device)

                    region_embeds = build_region_embeds_from_indices(
                        vit_embeds_flat=vit_embeds_flat, 
                        region_info=region_info, 
                        batchify=True,
                    )

                    out_region = self.language_model(
                        input_ids=None,
                        inputs_embeds=region_embeds,
                        attention_mask=region_attn,
                        past_key_values=past_kv,
                        use_cache=True,
                        cache_position=region_pos,
                        # interleave_inf=True,
                        # interleave_cache=interleave_cache,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    past_kv = out_region.past_key_values
                    past_len += num_tokens
                    all_generated_ids.extend(region_ids[0].tolist())
                    # print(f"   ✅ Injected {num_tokens} <REGION> tokens")

                    last_logits = out_region.logits[:, -1, :]
                    next_id = last_logits.argmax(dim=-1, keepdim=True)
                    seg_hit = True
                    break

                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                if len(all_generated_ids) >= self.gen_config.max_length:
                    print(f"⚠️ Reached max_length ({self.gen_config.max_length}), stopping")
                    final_prediction = self.tokenizer.decode(all_generated_ids, skip_special_tokens=False).strip()
                    return {'prediction': final_prediction, 'prediction_masks': ret_masks}

            if not seg_hit:
                break

        final_prediction = self.tokenizer.decode(all_generated_ids, skip_special_tokens=False).strip()
        print(f"\n✅ Generation complete:\n   Total iterations: {iteration}\n   Total [SEG] masks: {len(ret_masks)}\n   Total tokens: {len(all_generated_ids)}")
        return {
            'prediction': final_prediction,
            'prediction_masks': ret_masks,
        }



    @torch.no_grad()
    def predict_forward_with_grounding_multi(
        self,
        image_list=None,
        video=None,
        text=None,
        past_text='',
        mask_prompts=None,
        tokenizer=None,
        max_tokens_per_seg=64,
        max_iterations=20,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        device = torch.device("cuda")
        self.gen_config.max_length = 8192

        if image_list is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = self.template['INSTRUCTION'].format(input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = torch.tensor(self.tokenizer.encode(input_text), device=device).unsqueeze(0)
            attn = torch.ones_like(ids, dtype=torch.long)

            L0 = ids.shape[1]
            cache_pos0 = torch.arange(0, L0, device=device)
            out = self.language_model(
                input_ids=ids, attention_mask=attn,
                past_key_values=None, use_cache=True,
                cache_position=cache_pos0,
                output_hidden_states=True, return_dict=True,
            )
            past_kv = out.past_key_values
            past_len = L0
            all_gen = []

            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            while True:
                step_attn = torch.ones_like(next_id, dtype=torch.long)
                step_pos  = torch.arange(past_len, past_len + 1, device=device)
                out = self.language_model(
                    input_ids=next_id, attention_mask=step_attn,
                    past_key_values=past_kv, use_cache=True,
                    cache_position=step_pos,
                    output_hidden_states=True, return_dict=True,
                )
                past_kv = out.past_key_values
                past_len += 1

                tok = next_id.item()
                all_gen.append(tok)
                if tok == self.tokenizer.eos_token_id or len(all_gen) >= self.gen_config.max_length:
                    break
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            predict = self.tokenizer.decode(all_gen, skip_special_tokens=False).strip()
            return {'prediction': predict, 'prediction_masks': []}

        if video is not None:
            pixel_values, extra_pixel_values = [], []
            ori_image_size = video[0].size
            for frame_idx, frame_image in enumerate(video):
                assert ori_image_size == frame_image.size
                g_image = np.array(frame_image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_image)
                if frame_idx < 5:
                    img = self.transformer(frame_image)
                    pixel_values.append(img)

            pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype).to(device)
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(self.torch_dtype).to(device)

            num_image_tokens = self.patch_token
            num_frames = len(pixel_values)
            ori_height, ori_width = ori_image_size[1], ori_image_size[0]
        else:
            extra_pixel_values = []
            ori_sizes = []
            for image in image_list:
                ori_width, ori_height = image.size
                ori_sizes.append((ori_height, ori_width))
                g_image = np.array(image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(self.torch_dtype).to(device)

            pixel_values = [self.transformer(image) for image in image_list]
            pixel_values = torch.stack(pixel_values).to(self.torch_dtype).to(device)

            num_image_tokens = pixel_values.shape[0] * self.patch_token

            num_frames = len(image_list)
            ori_height, ori_width = ori_sizes[0]

        num_images = pixel_values.shape[0]
        frame_tokens_list = []
        for i in range(num_images):
            frame_token_str = (
                f'{self.IMG_START_TOKEN}'
                f'{self.IMG_CONTEXT_TOKEN * self.patch_token}'
                f'{self.IMG_END_TOKEN}'
            )
            frame_tokens_list.append(frame_token_str)
        image_token_str = '\n'.join(frame_tokens_list)

        text = text.replace('<image>', image_token_str)
        input_text = self.template['INSTRUCTION'].format(input=text, round=1, bot_name=self.bot_name)
        input_text = past_text + input_text
        initial_ids = torch.tensor(self.tokenizer.encode(input_text), device=device).unsqueeze(0)

        past_kv = None
        past_len = 0
        ret_masks = []
        all_region_infos = []
        seg_token_id = self.seg_token_idx
        region_token_id = self.tokenizer.convert_tokens_to_ids('<REGION>')
        eos_token_id = self.tokenizer.eos_token_id
        all_generated_ids = []
        interleave_cache = {}
        seg_count = 0

        with torch.no_grad():
            vit_embeds = self.extract_feature(pixel_values.to(device))
            vit_embeds_3d = vit_embeds
            vit_embeds_flat = vit_embeds_3d.reshape(-1, vit_embeds_3d.shape[-1])

        print("\nStarting iterative generation with KV cache...")

        for iteration in range(1, max_iterations + 1):
            print(f"\nIteration {iteration}:")

            if iteration == 1:
                inputs_embeds0 = self.build_inputs_embeds_for_prefill(
                    initial_ids, vit_embeds_flat, self.img_context_token_id
                )
                L0 = initial_ids.shape[1]
                attn0 = torch.ones((1, L0), dtype=torch.long, device=device)
                cache_pos0 = torch.arange(0, L0, device=device)
                prefill_out = self.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds0,
                    attention_mask=attn0,
                    past_key_values=None,
                    use_cache=True,
                    cache_position=cache_pos0,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_kv = prefill_out.past_key_values
                past_len = L0

                next_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                if next_id is None:
                    logits_last = last_out.logits[:, -1, :]
                    next_id = logits_last.argmax(dim=-1, keepdim=True)

            seg_hit = False
            while True:
                step_attn = torch.ones_like(next_id, dtype=torch.long)
                step_pos  = torch.arange(past_len, past_len + 1, device=device)
                out = self.language_model(
                    input_ids=next_id,
                    attention_mask=step_attn,
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_position=step_pos,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
                past_len += 1

                tok = next_id.item()
                all_generated_ids.append(tok)

                if tok in self.tokenizer.all_special_ids:
                    pass
                else:
                    piece = self.tokenizer.decode([tok], skip_special_tokens=False, clean_up_tokenization_spaces=False)

                if tok == eos_token_id or tok == 151682 or tok == 151684:
                    print("EOS detected, generation complete")
                    final_prediction = self.tokenizer.decode(all_generated_ids, skip_special_tokens=False).strip()
                    print(f"\nGeneration complete:\n   Total iterations: {iteration}\n   Total [SEG] masks: {len(ret_masks)}\n   Total tokens: {len(all_generated_ids)}")
                    return {'prediction': final_prediction, 'prediction_masks': ret_masks}

                if tok == seg_token_id:
                    print(f"[SEG] #{seg_count} detected, predicting mask...")
                    last_layer_h = out.hidden_states[-1]
                    seg_hidden = last_layer_h[0, -1, :].unsqueeze(0)
                    seg_hidden = self.text_hidden_fcs(seg_hidden)

                    with torch.no_grad():
                        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
                        pred_masks = self.grounding_encoder.language_embd_inference(
                            sam_states, [seg_hidden] * num_frames
                        )
                        masks = F.interpolate(pred_masks, size=(ori_height, ori_width),
                                              mode='bilinear', align_corners=False)
                        masks = masks[:, 0].sigmoid()
                        predicted_mask = (masks > 0.5).squeeze(0)

                    coverage = predicted_mask.sum().item() / predicted_mask.numel() * 100
                    ret_masks.append(predicted_mask.cpu().numpy())

                    img_idx = seg_count % num_frames
                    single_mask = predicted_mask[img_idx] if predicted_mask.dim() == 3 else predicted_mask
                    cur_ori_h, cur_ori_w = ori_sizes[img_idx]

                    single_region_indices = self._get_visual_tokens_from_mask_multi_simple(
                        mask=single_mask,
                        ori_height=cur_ori_h,
                        ori_width=cur_ori_w,
                        max_tokens_per_object=8,
                    )
                    token_offset = img_idx * self.patch_token
                    single_region_indices = [idx + token_offset for idx in single_region_indices]

                    region_info = {
                        'visual_token_indices': single_region_indices,
                        'num_tokens': len(single_region_indices),
                    }
                    seg_count += 1
                    all_region_infos.append(region_info)
                    num_tokens = region_info['num_tokens']

                    region_ids = torch.full((1, num_tokens), region_token_id, dtype=torch.long, device=device)
                    region_attn = torch.ones(1, past_len + num_tokens, dtype=torch.long, device=device)
                    region_pos  = torch.arange(past_len, past_len + num_tokens, device=device)

                    region_embeds = build_region_embeds_from_indices(
                        vit_embeds_flat=vit_embeds_flat,
                        region_info=region_info,
                        batchify=True,
                    )

                    out_region = self.language_model(
                        input_ids=None,
                        inputs_embeds=region_embeds,
                        attention_mask=region_attn,
                        past_key_values=past_kv,
                        use_cache=True,
                        cache_position=region_pos,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    past_kv = out_region.past_key_values
                    past_len += num_tokens
                    all_generated_ids.extend(region_ids[0].tolist())

                    last_logits = out_region.logits[:, -1, :]
                    next_id = last_logits.argmax(dim=-1, keepdim=True)
                    seg_hit = True
                    break

                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                if len(all_generated_ids) >= self.gen_config.max_length:
                    print(f"Reached max_length ({self.gen_config.max_length}), stopping")
                    final_prediction = self.tokenizer.decode(all_generated_ids, skip_special_tokens=False).strip()
                    return {'prediction': final_prediction, 'prediction_masks': ret_masks}

            if not seg_hit:
                break

        final_prediction = self.tokenizer.decode(all_generated_ids, skip_special_tokens=False).strip()
        print(f"\nGeneration complete:\n   Total iterations: {iteration}\n   Total [SEG] masks: {len(ret_masks)}\n   Total tokens: {len(all_generated_ids)}")
        return {
            'prediction': final_prediction,
            'prediction_masks': ret_masks,
        }


    # def predict_forward_with_grounding(
    #     self,
    #     image=None,
    #     video=None,
    #     text=None,
    #     past_text='',
    #     mask_prompts=None,
    #     tokenizer=None,
    #     max_tokens_per_seg=64,
    # ):

    #     if not self.init_prediction_config:
    #         assert tokenizer
    #         self.preparing_for_generation(tokenizer=tokenizer)
        
    #     self.gen_config.max_length = 8192

    #     # ============================================
    #     # ============================================
    #     if image is None and video is None and '<image>' not in past_text:
    #         text = text.replace('<image>', "")
    #         input_text = ''
    #         input_text += self.template['INSTRUCTION'].format(
    #             input=text, round=1, bot_name=self.bot_name)
    #         input_text = past_text + input_text
    #         ids = self.tokenizer.encode(input_text)
    #         ids = torch.tensor(ids).cuda().unsqueeze(0)

    #         attention_mask = torch.ones_like(ids, dtype=torch.bool)

    #         mm_inputs = {
    #             'pixel_values': None,
    #             'input_ids': ids,
    #             'attention_mask': attention_mask,
    #             'position_ids': None,
    #             'past_key_values': None,
    #             'labels': None,
    #             'prompt_masks': None,
    #             'vp_overall_mask': None,
    #         }
        
    #         generate_output = self.generate(
    #             **mm_inputs,
    #             generation_config=self.gen_config,
    #             streamer=None,
    #             bos_token_id=self.tokenizer.bos_token_id,
    #             stopping_criteria=self.stop_criteria,
    #         )
    #         predict = self.tokenizer.decode(
    #             generate_output.sequences[0], skip_special_tokens=False).strip()
            
    #         return {'prediction': predict, 'prediction_masks': []}

    #     # ============================================
    #     # ============================================
    #     input_dict = {}
    #     if video is not None:
    #         pixel_values = []
    #         extra_pixel_values = []
    #         ori_image_size = video[0].size
    #         for frame_idx, frame_image in enumerate(video):
    #             assert ori_image_size == frame_image.size
    #             g_image = np.array(frame_image)
    #             g_image = self.extra_image_processor.apply_image(g_image)
    #             g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
    #             extra_pixel_values.append(g_image)
    #             if frame_idx < 5:
    #                 img = self.transformer(frame_image)
    #                 pixel_values.append(img)

    #         pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)
    #         g_pixel_values = torch.stack([
    #             self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
    #         ]).to(self.torch_dtype)
    #         num_image_tokens = self.patch_token
    #         num_frames = len(pixel_values)
    #         ori_height, ori_width = ori_image_size[1], ori_image_size[0]
    #     else:
    #         ori_image_size = image.size
    #         ori_width, ori_height = ori_image_size

    #         g_image = np.array(image)
    #         g_image = self.extra_image_processor.apply_image(g_image)
    #         g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
    #         extra_pixel_values = [g_pixel_values]
    #         g_pixel_values = torch.stack([
    #             self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
    #         ]).to(self.torch_dtype)

    #         images = dynamic_preprocess(image, self.min_dynamic_patch,
    #                                 self.max_dynamic_patch,
    #                                 self.image_size, self.use_thumbnail)

    #         pixel_values = [self.transformer(image) for image in images]
    #         pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
    #         num_image_tokens = pixel_values.shape[0] * self.patch_token
    #         num_frames = 1
        
    #     # ============================================
    #     # ============================================
    #     image_token_str = f'{self.IMG_START_TOKEN}' \
    #                     f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
    #                     f'{self.IMG_END_TOKEN}'
    #     image_token_str = image_token_str + '\n'
    #     image_token_str = image_token_str * num_frames
    #     image_token_str = image_token_str.strip()

    #     text = text.replace('<image>', image_token_str)
    #     input_text = ''
    #     input_text += self.template['INSTRUCTION'].format(
    #         input=text, round=1, bot_name=self.bot_name)
    #     input_text = past_text + input_text
        
    #     initial_ids = torch.tensor(self.tokenizer.encode(input_text)).cuda().unsqueeze(0)

    #     # ============================================
    #     # ============================================
    #     past_key_values = None
    #     cached_vit_embeds_3d = None
    #     cached_vit_embeds_flat = None
    #     all_generated_ids = []
    #     past_length=0
    #     ret_masks = []
    #     all_region_infos = []
    #     seg_token_id = self.seg_token_idx
    #     region_token_id = self.tokenizer.convert_tokens_to_ids('<REGION>')
    #     eos_token_id = self.tokenizer.eos_token_id
    #     total_length=0
        
    #     with torch.no_grad():
    #         sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
        
    #     print(f"\n🚀 Starting iterative generation with KV cache...")
        
    #     # ============================================
    #     # ============================================
    #     for iteration in range(1, max_iterations + 1):
    #         print(f"\n📝 Iteration {iteration}:")
            
    #         if len(all_generated_ids) >= self.gen_config.max_length:
    #             print(f"⚠️ Reached max_length ({self.gen_config.max_length}), stopping")
    #             break
            
    #         # ============================================
    #         # ============================================
    #         if iteration == 1:
    #             current_input_ids = initial_ids
    #             current_pixel_values = pixel_values
    #             current_g_pixel_values = g_pixel_values
    #             current_attention_mask = torch.ones_like(current_input_ids)
    #             position_ids=None
    #             cache_position=None
    #             current_region_info = None
    #             use_cache = False
    #             print(f"   First iteration: full input ({current_input_ids.shape[1]} tokens)")
    #         else:
    #             num_new_tokens=len(new_tokens)
    #             current_input_ids = new_tokens.unsqueeze(0)
    #             current_pixel_values = None
    #             current_g_pixel_values = None

    #             current_region_info = [all_region_infos[-1]] if all_region_infos else None
    #             position_ids = torch.arange(
    #                 past_length,
    #                 past_length + num_new_tokens,
    #                 device='cuda'
    #             ).unsqueeze(0)
    #             cache_position = torch.arange(
    #                 past_length,
    #                 past_length + num_new_tokens,
    #                 device='cuda'
    #             )

    #             use_cache = True
    #             print(f"   Incremental input: {current_input_ids.shape[1]} new tokens")
            
    #         current_attention_mask = torch.ones_like(current_input_ids, dtype=torch.bool)
            
    #         # ============================================
    #         # ============================================
    #         mm_inputs = {
    #             'pixel_values': current_pixel_values,
    #             'g_pixel_values': current_g_pixel_values,
    #             'input_ids': current_input_ids,
    #             'attention_mask': current_attention_mask,
    #             'past_key_values': past_key_values if use_cache else None,
    #             'region_info': current_region_info,
    #             'position_ids': position_ids,
    #             'cache_position': cache_position,
    #             'labels': None,
    #             'prompt_masks': None,
    #             'vp_overall_mask': None,
    #         }

    
    #         remaining_tokens = self.gen_config.max_length - len(all_generated_ids)
    #         max_new_tokens = min(200, remaining_tokens)
            
    #         generate_output,vit_embeds_3d, vit_embeds_flat  = self.generate(
    #             **mm_inputs,
    #             generation_config=self.gen_config,
    #             max_new_tokens=max_new_tokens,
    #             eos_token_id=[seg_token_id, eos_token_id],
    #             bos_token_id=self.tokenizer.bos_token_id,
    #             stopping_criteria=self.stop_criteria,
    #             output_hidden_states=True,
    #             return_dict_in_generate=True,
    #         )
            
    #         # ============================================
    #         # ============================================
            
    #         if hasattr(generate_output, 'past_key_values'):
    #             past_key_values = generate_output.past_key_values
    #             print(f"   ✅ Updated KV cache")
            
    #         new_generated = generate_output.sequences[0]
            
    #         print(f"   Generated {len(new_generated)} new tokens")
            
    #         all_generated_ids.extend(new_generated.tolist())
    #         if iteration == 1:
    #             past_length = len(generate_output.sequences[0])
    #         else:
    #             past_length += len(new_generated) + len(current_input_ids[0])
            
    #         if iteration == 1:
    #             if hasattr(generate_output, 'vit_embeds_flat'):
    #                 cached_vit_embeds_flat = vit_embeds_flat
    #                 cached_vit_embeds_3d = vit_embeds_3d
    #                 print(f"   ✅ Cached visual embeddings: {cached_vit_embeds_flat.shape}")
            
    #         # ============================================
    #         # ============================================
    #         if len(new_generated) == 0:
    #             print(f"⚠️ No new tokens generated, stopping")
    #             break
            
    #         last_token = new_generated[-1].item()
            
    #         if last_token == eos_token_id:
    #             print(f"✅ EOS detected, generation complete")
    #             break
    #         elif last_token != seg_token_id:
    #             print(f"⚠️ Neither [SEG] nor EOS detected, continuing...")
    #             new_tokens = torch.tensor([], dtype=torch.long, device='cuda')
    #             continue
            
    #         # ============================================
    #         # ============================================
    #         print(f"🎭 [SEG] detected, predicting mask...")

    #         hidden_states = generate_output.hidden_states
    #         last_hidden_states = [item[-1][0] for item in hidden_states]
    #         last_hidden_states = torch.cat(last_hidden_states, dim=0)
            
    #         seg_hidden_state = get_seg_hidden_states(
    #             last_hidden_states, 
    #             seg_id=self.seg_token_idx
    #         )
        
    #         seg_hidden_state = self.text_hidden_fcs(seg_hidden_state)

    #         with torch.no_grad():
    #             pred_masks = self.grounding_encoder.language_embd_inference(
    #                 sam_states, 
    #                 [seg_hidden_state] * num_frames
    #             )
                
    #             masks = F.interpolate(pred_masks, size=(ori_height, ori_width), 
    #                                 mode='bilinear', align_corners=False)
    #             masks = masks[:, 0]
    #             masks_sigmoid = masks.sigmoid()
    #             predicted_mask = (masks_sigmoid > 0.5).squeeze(0)
            
    #         coverage = predicted_mask.sum().item() / predicted_mask.numel() * 100
    #         print(f"   Mask coverage: {coverage:.2f}%")
    #         ret_masks.append(predicted_mask.cpu().numpy())
            
    #         # ============================================
    #         # ============================================
    #         print(f"📦 Sampling visual tokens from mask...")
            
    #         if cached_vit_embeds_flat is None:
    #             print(f"⚠️ Warning: No cached visual embeddings, need to extract")
            
    #         region_info = self._extract_region_info_from_mask(
    #             mask=predicted_mask,
    #             vit_embeds_flat=cached_vit_embeds_flat,
    #             pixel_values=pixel_values,
    #             ori_height=ori_height,
    #             ori_width=ori_width,
    #             max_tokens=max_tokens_per_seg
    #         )
            
    #         num_tokens = region_info['num_tokens']
    #         print(f"   ✅ Sampled {num_tokens} visual tokens")
            
    #         all_region_infos.append(region_info)
            
    #         # ============================================
    #         # ============================================
    #         new_tokens = torch.full(
    #             (num_tokens,),
    #             region_token_id,
    #             dtype=torch.long,
    #             device='cuda'
    #         )
            
    #         # all_generated_ids = torch.cat([all_generated_ids, new_tokens], dim=0)
    #         print(f"   ✅ Prepared {num_tokens} <REGION> tokens for next iteration")
        
    #     # ============================================
    #     # ============================================
    #     final_prediction = self.tokenizer.decode(all_generated_ids, skip_special_tokens=False).strip()
        
    #     print(f"\n✅ Generation complete:")
    #     print(f"   Total iterations: {iteration}")
    #     print(f"   Total [SEG] masks: {len(ret_masks)}")
    #     print(f"   Total tokens: {len(all_generated_ids)}")
        
    #     return {
    #         'prediction': final_prediction,
    #         'prediction_masks': ret_masks,
    #     }
        



    def _get_patch_layout_accurate(self, image_size_tuple, ori_height, ori_width, min_num=1, max_num=6, image_size=448):
        """
        InternVLdynamic_preprocesspatch layout
        
        Args:
            image_size_tuple: (num_patches,) num_patches
            ori_height, ori_width: 
            min_num, max_num, image_size: dynamic_preprocess
        
        Returns:
            patch_layout: list of dictpatch**resize**
            target_aspect_ratio: (rows, cols) grid
            resized_width, resized_height: resize
        """
        aspect_ratio = ori_width / ori_height
        
        target_ratios = {(i, j)
                        for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num}
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = ori_width * ori_height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        target_aspect_ratio = best_ratio
        
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        patch_layout = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            patch_layout.append({
                'patch_idx': i,
                'x_start': box[0],
                'x_end': box[2],
                'y_start': box[1],
                'y_end': box[3],
            })
        
        return patch_layout, target_aspect_ratio, target_width, target_height

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': None,
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
            }
            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx < 5:
                        img = self.transformer(frame_image)
                        pixel_values.append(img)

                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)
                num_image_tokens = self.patch_token
                num_frames = len(pixel_values)

                input_dict['vp_overall_mask'] = None
            else:
                ori_image_size = image.size

                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)

                if mask_prompts is not None:
                    vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
                    input_dict['vp_overall_mask'] = vp_overall_mask
                else:
                    input_dict['vp_overall_mask'] = None

                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
                num_image_tokens = pixel_values.shape[0] * self.patch_token
                num_frames = 1
            input_dict['g_pixel_values'] = g_pixel_values
            input_dict['pixel_values'] = pixel_values

            if mask_prompts is not None:
                # reshape mask prompts to feature size
                mask_prompts = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
                mask_prompts = [F.interpolate(
                    item.unsqueeze(0),
                    size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                          int(self.image_size // self.patch_size * self.downsample_ratio)),
                    mode='nearest').squeeze(0) for item in mask_prompts]
                region_pixels = []
                for mask_prompt in mask_prompts[0]:
                    region_pixels.append(mask_prompt.bool().to(torch.int64).sum())

                vp_token_str = '\nThere are {} part regions in the picture: '.format(len(mask_prompts[0]))
                for i in range(len(mask_prompts[0])):
                    vp_token_str = vp_token_str + \
                                   f"region{i + 1}" + self.VP_START_TOKEN + \
                                   self.IMG_CONTEXT_TOKEN * region_pixels[i] + \
                                   self.VP_END_TOKEN
                    if i == len(mask_prompts[0]) - 1:
                        vp_token_str = vp_token_str + '.\n'
                    else:
                        vp_token_str = vp_token_str + ', '
            else:
                vp_token_str = ''

            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = image_token_str + '\n'
            image_token_str = image_token_str * num_frames
            image_token_str = image_token_str.strip()

            ret_masks = []

            if '<image>' in text or mask_prompts is not None:
                assert past_text is None or len(past_text) == 0
            text = text.replace('<image>', image_token_str + vp_token_str)
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': input_dict['pixel_values'],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': mask_prompts,
                'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}




def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation_phi3(
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

