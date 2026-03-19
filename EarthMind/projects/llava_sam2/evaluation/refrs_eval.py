import argparse
import copy
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random
import cv2
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('--model_path',default="/data/new2/train_finalnew", help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refsegrs',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='test',
        help='Specify a split')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

DATASETS_ATTRIBUTES = {
    'refsegrs': {'splitBy': "unc", 'dataset_name': 'refsegrs'},
    'rrsisd': {'splitBy': "unc", 'dataset_name': 'rrsisd'},
    'risbench': {'splitBy': "unc", 'dataset_name': 'risbench'},
    'sarnew': {'splitBy': "unc", 'dataset_name': 'sarnew'}
}

args = parse_args()
DATA_PATH = 'data/Refer_Segm'
# Dynamically construct image folder path
IMAGE_FOLDER = os.path.join(DATA_PATH, args.dataset, 'images')


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    dataset = RESDataset(
        image_folder=IMAGE_FOLDER,
        dataset_name=dataset_info['dataset_name'],
        data_path=DATA_PATH,
        split=args.split,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))


    negative = 0
    results = []

    for idx in tqdm.tqdm(per_rank_ids):
        print("idx", idx)
        data_batch = dataset[idx]

        # 保存 GT 信息
        img_id = data_batch['img_id']
        gt_masks = data_batch['gt_masks']      # shape: (N, H, W)
        texts = data_batch['text']             # length: N（与 gt_masks 对应）

        assert gt_masks.shape[0] == len(texts), f"GT/Text mismatch: {gt_masks.shape[0]} vs {len(texts)}"

        prediction = {
            'img_id': img_id,
            'gt_masks': mask_to_rle(gt_masks.cpu().numpy())
        }

        # 删除以免被 model.forward 使用
        del data_batch['img_id'], data_batch['gt_masks'], data_batch['text']

        pred_masks = []

        for i, text in enumerate(texts):
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text

            pred_mask_list = model.predict_forward(**_data_batch, tokenizer=tokenizer)['prediction_masks']

            if len(pred_mask_list) == 0:
                print("No seg pred !!!")
                pred_masks.append(None)
                continue

            pred_mask = pred_mask_list[0]
            # print("predict shape:", pred_mask.shape)

            gt_mask_i = gt_masks[i:i+1]  # (1, H, W)
            gt_shape = gt_mask_i.shape
            # print("gt",gt_shape)

            if pred_mask.shape != gt_shape:
                print(f"Shape mismatch. Resizing from {pred_mask.shape} to {gt_shape}")
                negative += 1

                pred_mask = pred_mask.astype(np.uint8)
                resized = cv2.resize(
                    pred_mask[0] if pred_mask.ndim == 3 else pred_mask,
                    (gt_shape[-1], gt_shape[-2]),
                    interpolation=cv2.INTER_NEAREST
                )
                pred_mask = resized[np.newaxis, ...]  # 变为 (1, H, W)
                print("new", pred_mask.shape)

            pred_mask_rle = mask_to_rle(pred_mask)
            pred_masks.append(pred_mask_rle)

        prediction['prediction_masks'] = pred_masks
        prediction['texts'] = texts  # 可选，便于后续分析
        results.append(prediction)

    print("Total shape mismatch corrected:", negative)

    # 后处理保存
    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)

    if get_rank() == 0:
        metric = dataset.evaluate(results, './work_dirs')
        print(metric)
    # negative=0
    # for idx in tqdm.tqdm(per_rank_ids):
    #     print("idx", idx)
    #     data_batch = dataset[idx]
        
    #     # 保存 GT 信息
    #     img_id = data_batch['img_id']
    #     gt_mask = data_batch['gt_masks']
    #     gt_shape = gt_mask.shape  # e.g., (1, 800, 800)
    #     print("gt",gt_shape)

    #     prediction = {
    #         'img_id': img_id,
    #         'gt_masks': mask_to_rle(gt_mask.cpu().numpy())
    #     }

    #     del data_batch['img_id'], data_batch['gt_masks']
        
    #     texts = data_batch['text']
    #     del data_batch['text']
    #     pred_masks = []

    #     for text in texts:
    #         _data_batch = copy.deepcopy(data_batch)
    #         _data_batch['text'] = text

    #         pred_mask_list = model.predict_forward(**_data_batch, tokenizer=tokenizer)['prediction_masks']
            
    #         if len(pred_mask_list) == 0:
    #             print("No seg pred !!!")
    #             pred_masks.append(None)
    #         else:
    #             pred_mask = pred_mask_list[0]
    #             print("predict shape:", pred_mask.shape)

    #             # 调整预测 mask 尺寸
    #             if pred_mask.shape != gt_shape:
    #                 print(f"Shape mismatch. Resizing from {pred_mask.shape} to {gt_shape}")
    #                 negative += 1

    #                 # Resize to GT shape using cv2
    #                 pred_mask = pred_mask.astype(np.uint8)
    #                 resized = cv2.resize(pred_mask[0] if pred_mask.ndim == 3 else pred_mask,
    #                                     (gt_shape[-1], gt_shape[-2]),
    #                                     interpolation=cv2.INTER_NEAREST)
    #                 pred_mask = resized[np.newaxis, ...]  # 保持 (1, H, W)
    #                 print("new",pred_mask.shape)

    #             pred_mask_rle = mask_to_rle(pred_mask)
    #             pred_masks.append(pred_mask_rle)

    #     prediction['prediction_masks'] = pred_masks
    #     results.append(prediction)

    # print("Total shape mismatch corrected:", negative)

    
    # tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    # results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    # if get_rank() == 0:
    #     metric = dataset.evaluate(results, './work_dirs')
    #     print(metric)
    
    

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

if __name__ == '__main__':
    main()
