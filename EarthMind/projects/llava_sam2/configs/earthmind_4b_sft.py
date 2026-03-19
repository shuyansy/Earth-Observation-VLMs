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

from projects.llava_sam2.models import VideoLLaVASAMModel, SAM2TrainRunner, VideoLLaVASAMModel_zero3
from projects.llava_sam2.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset
from projects.llava_sam2.datasets import VideoChatUniViDataset
from projects.llava_sam2.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset, OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset
from projects.llava_sam2.datasets import LLaVADataset,  Multi_LLaVADataset, MS_LLaVADataset
from projects.llava_sam2.datasets import Multi_ReferSegmDataset,ReferSegmDataset
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = 'pretrained/InternVL2_5-4B'
pretrained_pth = "pretrained/iter_112588.pth"

# Data
template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

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
    # loss_sample_points=False,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
geochat_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='data/geochatdata/GeoChatvqa.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder='data/GeoChat_finetuning/final_images_llava',
)


earthgpt_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='data/earthgptvqa/annotations/train.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder='data/earthgptvqa/images',
)

# ms_vqa_dataset = dict(
#     type=MS_LLaVADataset,
#     tokenizer=tokenizer,
#     data_path='data/multi-spectrum/ms_train.json',
#     prompt_template=prompt_template,
#     special_tokens=special_tokens,
#     image_folder='data/multi-spectrum/rgb_images',
# )

ms_vqa_dataset = dict(
    type=MS_LLaVADataset,
    tokenizer=tokenizer,
    data_path='/data/ms_train_new_filter.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder='/data/BigEarthNet-S2',
)


vrs_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='data/vrsbenchvqa/annotations/vqatrain.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder='data/vrsbenchvqa/Images_train/',
)

refsegrs_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root="data/Refer_Segm/refsegrs",
    data_prefix=dict(img_path='images/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

risbench_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root="data/Refer_Segm/risbench",
    data_prefix=dict(img_path='images/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

rrsisd_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root="data/Refer_Segm/rrsisd",
    data_prefix=dict(img_path='images/'),
    ann_file='instances_new.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)


vrsbench_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root="data/vrsbench_seg",
    data_prefix=dict(img_path='images/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)


sar_segm_dataset=dict(
    type=Multi_ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root="data/Refer_Segm/sarnew",
    data_prefix=dict(
        rgb_path='rgb/',
        sar_path='sar/',
        img_path='sar/',
    ),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

sar_vqa_dataset = dict(
    type=Multi_LLaVADataset,
    tokenizer=tokenizer,
    data_root="data/pair_data/train",
    data_path='data/pair_data/train/final_all.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    data_prefix=dict(
        rgb_path='rgb/img',
        sar_path='sar/img',
        img_path='sar/img',
    ),
)



rsvg_dataset = dict(
    type=OspreyDescriptionDataset,
    image_folder="data/region_caption/rsvg/images",
    data_path="data/region_caption/rsvg_train.json",
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)


georegion_dataset = dict(
    type=OspreyDescriptionDataset,
    image_folder="data/GeoChat_finetuning/final_images_llava",
    data_path="data/geochatdata/GeoChatregion_train_cleaned.json",
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)



train_dataset = dict(
    type=ConcatDataset, datasets=[
        rrsisd_segm_dataset,refsegrs_segm_dataset,risbench_segm_dataset,
        rrsisd_segm_dataset,refsegrs_segm_dataset,risbench_segm_dataset,
        rrsisd_segm_dataset,refsegrs_segm_dataset,risbench_segm_dataset,
        vrs_vqa_dataset,
        earthgpt_vqa_dataset,
        geochat_vqa_dataset,
        ms_vqa_dataset,
        rsvg_dataset,
        rsvg_dataset,
        rsvg_dataset,
        rsvg_dataset,
        rsvg_dataset,
        georegion_dataset, 
        vrsbench_segm_dataset
        # sar_segm_dataset,
        # sar_vqa_dataset,
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
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
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

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
