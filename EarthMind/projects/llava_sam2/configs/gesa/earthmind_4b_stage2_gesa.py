# EarthMind stage 2 (EO adaptation) with GeoSync Attention (GeSA): single-sensor EO data + FusionEO
# cross-sensor pairs; init = EarthMind-Pretrain (stage 1, pretrained/iter_112588.pth); hyper-parameters of
# Table IV. Global batch 64 = #GPUs x batch_size (2) x accumulative_counts (8 for 4 GPUs, 4 for 8 GPUs).
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts_multi.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.llava_sam2.models.internvl import InternVL_Slowfast
from projects.llava_sam2.models import VideoLLaVASAMModel_zero3, SAM2TrainRunner
from projects.llava_sam2.datasets import (video_lisa_collate_fn, LLaVADataset, ReferSegmDataset,
                                          OspreyDescriptionDataset, GeSAPairedVQADataset,
                                          GeSAPairedRefSegDataset, GeSAMultiSpectralDataset)
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# ---- cross-sensor fusion variant ----
fusion_mode = 'concat'      # concat | avg | external
attn_mode = 'full'          # causal | intra | full   (GeSA: full, Eq. 3)
pos_mode = 'shared'            # seq | shared            (GeSA: shared, Eq. 4)
modality = 'fused'            # fused | optical | sar   (training-time sensor input)
fusion_layout = 'avg' if fusion_mode == 'avg' else 'concat'
sar_pad_mode = 'zero'              # Table VI default

# Model
path = 'pretrained/InternVL2_5-4B'
pretrained_pth = 'pretrained/iter_112588.pth'

# Data
template = 'phi3_chat'
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Scheduler & Optimizer (Table IV): 4 GPUs x 2 x 8 accumulation = global batch 64
batch_size = 2  # per_device
accumulative_counts = 8
dataloader_num_workers = 8
max_epochs = 1
optim_type = AdamW            # weight_decay = 0  ->  Adam
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.0
max_norm = 1  # grad clip
warmup_ratio = 0.3

# Save
save_steps = 2000
save_total_limit = 2

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
        special_tokens=special_tokens,
        fusion_mode=fusion_mode,
        attn_mode=attn_mode,
        pos_mode=pos_mode,
        llm_attn_implementation='sdpa',
        external_fusion_layers=5,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
# ---- single-sensor EO data (released EarthMind stage-2 recipe) ----
def _res(root, ann='instances.json'):
    return dict(
        type=ReferSegmDataset,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        data_root=root,
        data_prefix=dict(img_path='images/'),
        ann_file=ann,
        split_file='refs(unc).p',
        prompt_template=prompt_template,
        num_classes_per_sample=5,
        max_length=max_length,
    )

rrsisd_segm_dataset = _res('data/Refer_Segm/rrsisd', 'instances_new.json')
refsegrs_segm_dataset = _res('data/Refer_Segm/refsegrs')
risbench_segm_dataset = _res('data/Refer_Segm/risbench')
vrsbench_segm_dataset = _res('data/vrsbench_seg')

def _vqa(path, folder):
    return dict(type=LLaVADataset, tokenizer=tokenizer, data_path=path, prompt_template=prompt_template,
                special_tokens=special_tokens, image_folder=folder)

earthgpt_vqa_dataset = _vqa('data/earthgptvqa/annotations/train.json', 'data/earthgptvqa/images')
vrs_vqa_dataset = _vqa('data/vrsbenchvqa/annotations/vqatrain.json', 'data/vrsbenchvqa/Images_train/')
geochat_vqa_dataset = _vqa('data/geochatdata/GeoChatvqa.json', 'data/GeoChat_finetuning/final_images_llava')

def _region(path, folder):
    return dict(type=OspreyDescriptionDataset, image_folder=folder, data_path=path, tokenizer=tokenizer,
                template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
                max_length=max_length, lazy=True, repeats=1, special_tokens=special_tokens)

rsvg_dataset = _region('data/region_caption/rsvg_train.json', 'data/region_caption/rsvg/images')
georegion_dataset = _region('data/geochatdata/GeoChatregion_train_cleaned.json',  # as in the released config
                            'data/GeoChat_finetuning/final_images_llava')

# multispectral (BigEarthNet S2, 12 bands -> 4 pseudo-frames by three-band grouping)
ms_vqa_dataset = dict(
    type=GeSAMultiSpectralDataset,
    tokenizer=tokenizer,
    data_path='data/multi-spectrum/ms_train.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    ms_folder='data/multi-spectrum/images',
    modality='optical',
    group_size=3,
    max_length=max_length,
)
# ---- FusionEO: co-registered optical-SAR instruction pairs ----
paired_vqa_dataset = dict(
    type=GeSAPairedVQADataset,
    tokenizer=tokenizer,
    data_root='data/pair_data/train',
    data_path='data/pair_data/train/final_all.json',
    data_prefix=dict(rgb_path='rgb/img', sar_path='sar/img'),
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    max_length=max_length,
    modality=modality,
    fusion_layout=fusion_layout,
    sar_pad_mode=sar_pad_mode,
)

paired_segm_dataset = dict(
    type=GeSAPairedRefSegDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='data/Refer_Segm/sarnew',
    data_prefix=dict(rgb_path='rgb/', sar_path='sar/', img_path='sar/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    modality=modality,
    fusion_layout=fusion_layout,
    sar_pad_mode=sar_pad_mode,
    grounding_modality='sar',
)
# a sample that fails to load is replaced by a random other one instead of ending the run
from projects.llava_sam2.datasets.safe_concat import SafeConcatDataset

train_dataset = dict(
    type=SafeConcatDataset, datasets=[
        rrsisd_segm_dataset, refsegrs_segm_dataset, risbench_segm_dataset,
        rrsisd_segm_dataset, refsegrs_segm_dataset, risbench_segm_dataset,
        rrsisd_segm_dataset, refsegrs_segm_dataset, risbench_segm_dataset,
        vrs_vqa_dataset,
        earthgpt_vqa_dataset,
        geochat_vqa_dataset,
        ms_vqa_dataset,
        rsvg_dataset, rsvg_dataset, rsvg_dataset, rsvg_dataset, rsvg_dataset,
        georegion_dataset,
        vrsbench_segm_dataset,
        # cross-sensor FusionEO pairs (trained with GeSA)
        paired_segm_dataset,
        paired_vqa_dataset,
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = []

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=True,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)
