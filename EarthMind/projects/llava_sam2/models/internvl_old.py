import torch
from xtuner.model import InternVL_V1_5
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
import torch.nn as nn

from mmengine import print_log
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BitsAndBytesConfig)
from xtuner.model.utils import (find_all_linear_names, get_peft_model_state_dict,
                    guess_load_checkpoint, make_inputs_require_grad)
import os



def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

# This function is used to split large model
def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    return device_map


# def extract_feature_with_projector_toggle(self, pixel_values, use_projector=True):
#     if self.select_layer == -1:
#         vit_embeds = self.vision_model(
#             pixel_values=pixel_values,
#             output_hidden_states=False,
#             return_dict=True).last_hidden_state
#     else:
#         vit_embeds = self.vision_model(
#             pixel_values=pixel_values,
#             output_hidden_states=True,
#             return_dict=True).hidden_states[self.select_layer]

#     vit_embeds = vit_embeds[:, 1:, :]
#     h = w = int(vit_embeds.shape[1] ** 0.5)
#     vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
#     vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
#     vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

#     if use_projector:
#         vit_embeds = self.mlp1(vit_embeds)

#     return vit_embeds



class InternVL_Slowfast(InternVL_V1_5):

    def __init__(self,
                 model_path,
                 freeze_llm=True,
                 freeze_visual_encoder=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 special_tokens=None,
                 model_split=False,
                 ):
        print_log('Start to load InternVL_V1_5 model.', logger='current')
        super(InternVL_V1_5, self).__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('language_model')

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            quantization_clazz = quantization_config.pop('type')
            quantization = quantization_clazz(**quantization_config)

        if model_split:
            # print("\n\nDone Model Split !!!!!!!!!!!\n\n")
            device_map = split_model("InternVL2-26B")
            # print(device_map)
            self.device = 'cuda'
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map).eval()

        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization,
                config=config,
                trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.tokenizer = tokenizer

        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, 'enable_input_require_grads'):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0
        print_log(self, logger='current')
        print_log('InternVL_V1_5 construction is complete', logger='current')

        self.transfer_to_hf = False

 
        # import types
        # self.model.extract_feature = types.MethodType(extract_feature_with_projector_toggle, self.model)


    def _add_special_tokens(self, special_tokens):
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)

        if num_new_tokens > 0:
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))

    def _post_init(self, fast_pool_size=4, fast_pool=True):
        if fast_pool:
            self.fast_pool = nn.AdaptiveAvgPool2d((fast_pool_size, fast_pool_size))
        return

    def forward(self, data, data_samples=None, mode='loss', fast_token_idx=None):
        if 'fast_pixel_values' in data.keys():
            assert fast_token_idx is not None
            fast_pixel_values = data['fast_pixel_values']
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                fast_concat_images = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)
            else:
                raise NotImplementedError()
        else:
            fast_pixel_values = None
            fast_concat_images = None

        # Load SAR as main visual input (default)
        pixel_values = data['pixel_values']
        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        # Load RGB image as auxiliary input (for contrastive learning etc.)
        if 'rgb_pixel_values' in data:
            rgb_values = data['rgb_pixel_values']
            if type(rgb_values) is list or rgb_values.ndim == 5:
                if type(rgb_values) is list:
                    rgb_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in rgb_values
                    ]
                rgb_concat_images = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in rgb_values], dim=0)
            else:
             
                raise NotImplementedError()
        else:
            rgb_concat_images = None

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']

        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        vp_overall_mask = data.get('vp_overall_mask', None)
        prompt_masks = data.get('prompt_masks', None)
        # print("1111111",concat_images.shape,rgb_concat_images.shape)

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,  # SAR
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            fast_pixel_values=fast_concat_images,
            fast_token_idx=fast_token_idx,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
            rgb_pixel_values=rgb_concat_images,  # ✅ 新增
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
        fast_pixel_values=None,
        fast_token_idx=None,
        vp_overall_mask=None,
        prompt_masks=None,
        rgb_pixel_values=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids).clone()

        if fast_pixel_values is not None:
            n_fast_images = fast_pixel_values.shape[0]
            whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
            vit_embeds = self.extract_feature(whole_pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
            _size = int(fast_vit_embeds.shape[1] ** 0.5)
            fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size, fast_vit_embeds.shape[-1])
            # pooling
            fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
            fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
            fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
            vit_embeds = vit_embeds[n_fast_images:]
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?

            rgb_vit_embeds = self.model.extract_feature(rgb_pixel_values)
            rgb_vit_embeds = rgb_vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = None

            constrastive_pair=(vit_embeds,rgb_vit_embeds)
        
        
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
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask] # (n_img, hw, c)

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
        selected = (input_ids == self.model.img_context_token_id)

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

        if fast_vit_embeds is not None:
            selected = (input_ids == fast_token_idx)
            selected_tot = selected.sum().item()
            if selected_tot > fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]:
                assert selected_tot % (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]) == 0
                repeat_times = selected_tot / (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1])
                fast_vit_embeds = fast_vit_embeds.repeat(int(repeat_times), 1, 1)
            try:
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C)
            except Exception as e:
                fast_vit_embeds = fast_vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[fast_selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'fast_vit_embeds.shape={fast_vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = fast_vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
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
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ),constrastive_pair

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_token_idx=None,
        fast_pixel_values=None,
        prompt_masks=None,
        vp_overall_mask=None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.model.device
        assert self.model.img_context_token_id is not None

        if fast_pixel_values is not None:
            assert fast_token_idx is not None
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                # b*n, c, h, w
                fast_pixel_values = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)

                if fast_pixel_values is not None:
                    n_fast_images = fast_pixel_values.shape[0]
                    whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
                    vit_embeds = self.model.extract_feature(whole_pixel_values.to(device))
                    # vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
                    fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
                    _size = int(fast_vit_embeds.shape[1] ** 0.5)
                    fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size,
                                                              fast_vit_embeds.shape[-1])
                    # pooling
                    fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
                    fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
                    fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
                    vit_embeds = vit_embeds[n_fast_images:]
                else:
                    fast_vit_embeds = None
                    vit_embeds = self.model.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]
            
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

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
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!"\
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            if fast_vit_embeds is not None:
                selected = (input_ids == fast_token_idx)
                # FIXME, add repeat.
                assert selected.sum() != 0
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def state_dict(self, *args, **kwargs):
        if self.transfer_to_hf:
            state_dict = super(InternVL_V1_5, self).state_dict(*args, **kwargs)
            return state_dict
        else:
            return super().state_dict(*args, **kwargs)


