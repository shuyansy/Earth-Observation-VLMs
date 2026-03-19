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

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.terrascope.models.internvl import InternVL_Slowfast

from projects.terrascope.models import VideoLLaVASAMModel, SAM2TrainRunner, VideoLLaVASAMModel_zero3
from projects.terrascope.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset
from projects.terrascope.datasets import VideoChatUniViDataset,VideoCoTDataset
from projects.terrascope.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset, OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset
from projects.terrascope.datasets import LLaVADataset
from projects.terrascope.datasets import ReferSegmDataset
from projects.terrascope.datasets import SARRGBDataset
from projects.terrascope.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = 'pretrained/InternVL3-8B'
pretrained_pth = "pretrained/iter_15000.pth"

# template = "phi3_chat"
# prompt_template = PROMPT_TEMPLATE.phi3_chat
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4
dataloader_num_workers = 16
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 5000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['<think>','</think>','<answer>','</answer>','[SEG]','<REGION>','<p>', '</p>', '<vp>', '</vp>','<|end|>']

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
        loss_weight=2),   #2
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

rs_dataset = dict(
    type=GranDfGCGDataset,
    image_folder="data/chatearthnet/s2_images",
    data_path="data/train_json/chatearthnet_vqa.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_dataset_caption = dict(
    type=GranDfGCGDataset,
    image_folder="data/chatearthnet/s2_images",
    data_path="data/train_json/training_data_caption.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_dataset_be = dict(
    type=GranDfGCGDataset,
    image_folder="data/bigearthnet/BigEarthNet-RGB",
    data_path="data/train_json/bigearthnet_cls.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_dataset_cloud = dict(
    type=GranDfGCGDataset,
    image_folder="data/bigearthnet/BigEarthNet-RGB",
    data_path="data/train_json/cloudvqa.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_dataset_vqa = dict(
    type=GranDfGCGDataset,
    image_folder="data/bigearthnet/BigEarthNet-RGB",
    data_path="data/train_json/bigearthnet_vqa.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_dataset_tem = dict(
    type=VideoCoTDataset,
    image_folder="data/xbd/train/images",
    json_file="data/train_json/disaster.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_dataset_neg_vqa  = dict(
    type=GranDfGCGDataset,
    image_folder="data/bigearthnet/BigEarthNet-RGB",
    data_path="data/train_json/bigearthnet_neg_vqa.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)


rs_vqa_lr  = dict(
    type=GranDfGCGDataset,
    image_folder="data/rsvqa/LR_png",
    data_path="data/train_json/rsvqa_lr_new_new.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)


road_vqa  = dict(
    type=GranDfGCGDataset,
    image_folder="data/road/deepglobe/train",
    data_path="data/train_json/visualcot_road_ratio.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)


road_tem = dict(
    type=VideoCoTDataset,
    image_folder="data/road/deepglobe/train",
    json_file="data/train_json/road_temporal.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_vqa_val  = dict(
    type=GranDfGCGDataset,
    image_folder="data/rsvqa/LR_png",
    data_path="data/train_json/rsvqa_val_new_new.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)


rs_dataset_urban = dict(
    type=GranDfGCGDataset,
    image_folder="data/bigearthnet/BigEarthNet-RGB",
    data_path="data/train_json/urbanvqa.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)


rs_dataset_winter = dict(
    type=GranDfGCGDataset,
    image_folder="data/bigearthnet/BigEarthNet-RGB",
    data_path="data/train_json/wintervqa.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)


# SAR+RGB paired datasets (ChatEarthNet S2+S1)
rs_dataset_sar_rgb_vqa = dict(
    type=SARRGBDataset,
    rgb_image_folder="data/chatearthnet/s2_images",
    sar_image_folder="data/chatearthnet/s1_images",
    data_path="data/train_json/chatearthnet_vqa_sar_rgb.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

rs_dataset_sar_rgb_caption = dict(
    type=SARRGBDataset,
    rgb_image_folder="data/chatearthnet/s2_images",
    sar_image_folder="data/chatearthnet/s1_images",
    data_path="data/train_json/training_data_caption_sar_rgb.json",
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)


train_dataset = dict(
    type=ConcatDataset, datasets=[
        rs_dataset,
        rs_dataset_caption,
        rs_dataset_be,
        rs_dataset_cloud,
        rs_dataset_vqa,
        rs_dataset_tem,
        rs_dataset_tem,
        rs_dataset_neg_vqa,
        rs_vqa_lr,
        road_vqa,
        road_tem,
        road_tem,
        rs_vqa_val,
        rs_dataset_urban,
        rs_dataset_winter,
        rs_dataset_sar_rgb_vqa,
        rs_dataset_sar_rgb_caption,
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
