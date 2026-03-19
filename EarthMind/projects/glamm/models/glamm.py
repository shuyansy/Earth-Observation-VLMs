import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from xtuner.model.utils import LoadWoInit, guess_load_checkpoint
from xtuner.model.llava import LLaVAModel

from mmengine.model import BaseModel
from mmengine import print_log

from projects.glamm.utils import prepare_inputs_labels_for_multimodal
from projects.glamm.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class GLaMM(LLaVAModel):
    def __init__(self,
                 use_activation_checkpointing=True,
                 tokenizer=None,
                 grounding_encoder=None,
                 region_encoder=None,
                 loss_mask=None,
                 loss_dice=None,
                 *args, **kwargs):
        super(GLaMM, self).__init__(
            *args, use_activation_checkpointing=use_activation_checkpointing, **kwargs)

        self.use_activation_checkpointing = use_activation_checkpointing
        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()

        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        self.grounding_encoder.mask_decoder.requires_grad_(True)

        if region_encoder is not None:
            self.region_encoder = BUILDER.build(region_encoder)

        in_dim = self.config.hidden_size
        out_dim = self.grounding_encoder.mask_decoder.transformer_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

    def _add_special_tokens(self):
        reg_tokens = ['<im_start>', '<im_end>', '<bbox>', '<point>']
        segmentation_tokens = ['[SEG]']
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            input_embeddings = self.llm.get_input_embeddings().weight.data
            output_embeddings = self.llm.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.bop_token_idx = self.tokenizer("<p>", add_special_tokens=False).input_ids[0]
        self.eop_token_idx = self.tokenizer("</p>", add_special_tokens=False).input_ids[0]
        self.bbox_token_idx = self.tokenizer("<bbox>", add_special_tokens=False).input_ids[0]

        if self.use_activation_checkpointing or self.use_llm_lora or not self.freeze_llm:
            self.llm.enable_input_require_grads()

    def forward(self, data, data_samples=None, mode='loss'):
        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'].to(self.visual_encoder.dtype),
                output_hidden_states=True)
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            data['pixel_values'] = pixel_values
            bboxes = data.pop('bboxes', None)
            if bboxes is not None:
                select_hidden_state_layer = -2
                num_level_reg_features = 4
                mlvl_reg_features = visual_outputs.hidden_states[select_hidden_state_layer::-3]
                mlvl_reg_features = mlvl_reg_features[::-1]
                mlvl_reg_features = mlvl_reg_features[-num_level_reg_features:]
                mlvl_reg_features = [item[:, 1:] for item in mlvl_reg_features]
                mlvl_reg_features = self.region_encoder(mlvl_reg_features, bboxes)
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
            
            if bboxes is not None:
                inputs_embeds = data['inputs_embeds']
                for i, reg_feat in enumerate(mlvl_reg_features):
                    reg_mask = data['new_input_ids'][i] == self.bbox_token_idx
                    inputs_embeds[i][reg_mask] = reg_feat
                data['inputs_embeds'] = inputs_embeds

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def compute_loss(self, data, data_samples=None):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        new_input_ids = data.pop('new_input_ids', None)

        output = self.llm(output_hidden_states=True, **data)
        if gt_masks is None:
            return {'llm_loss': output.loss}

        resize_list = [pixel.shape[-2:] for pixel in g_pixel_values]
        ori_size_list = [mask.shape[-2:] for mask in gt_masks]
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess(pixel) for pixel in g_pixel_values
        ])
        image_embeddings = self.grounding_encoder.image_encoder(g_pixel_values)

        seg_token_mask = new_input_ids == self.seg_token_idx
        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        pred_embeddings = hidden_states[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        
        pred_masks = self._generate_and_postprocess_masks(
            pred_embeddings_list, image_embeddings, resize_list, ori_size_list)
        
        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        for i in range(bs):
            pred_mask = pred_masks[i]
            gt_mask = gt_masks[i]

            sam_loss_mask = self.loss_mask(pred_mask, gt_mask)
            sam_loss_dice = self.loss_dice(pred_mask, gt_mask)
            accuracy = torch.eq((pred_mask.sigmoid() > 0.5), gt_mask).to(pred_mask).mean()
            loss_mask += sam_loss_mask
            loss_dice += sam_loss_dice


        loss_dict = {
            'loss_mask': loss_mask / bs,
            'loss_dice': loss_dice / bs,
            'accuracy': accuracy,
            'llm_loss': output.loss,
        }
        return loss_dict

  
    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list=None, orig_size_list=None, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            
            pred_mask = self.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size_list[i], )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks
    
    def predict(self, data):
        pass

    def _forward(self, data, dta_samples=None):
        outputs = self.llm(**data)
        return outputs
