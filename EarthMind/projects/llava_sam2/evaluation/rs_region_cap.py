import argparse
import re
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import json

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocog region caption')
    parser.add_argument('--model_path',default="/data/Sa2VA-main/train_weight_newnew", help='hf model path.')
    parser.add_argument(
        '--output-path',
        default='region_cap_pred.json',
        help='save path of the prediction')
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

class RegionCapInferenceDataset:
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 ):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def decode_mask(self, annotation, image_info):
        flag = False
        masks = []

        for ann_id in range(1):

            ann = {"segmentation": annotation}

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        return masks

    def get_questions(self):
        # question = "<image>\nPlease give me a short description of the region in the picture marked by region1. Please response in a word."
        question = "<image>\nPlease give me a short description of the region in the picture marked by region1."
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.image_dict[image_id]['file_name']

        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        masks = self.ann_dict[image_id]['segmentation']
        image_info = self.image_dict[image_id]
        masks = self.decode_mask(masks, image_info)

        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = image_id
        data_dict['mask_prompts'] = [masks]

       
        caption=self.image_dict[image_id]["caption"]

        return data_dict,caption


ANNOTATION_FILE = 'data/region_caption/rsvg_test.json'
IMAGE_FOLDER = 'data/region_caption/rsvg/images'

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

    dataset = RegionCapInferenceDataset(
        image_folder=IMAGE_FOLDER,
        annotation_file=ANNOTATION_FILE,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch,caption = dataset[idx]
  
        result_dict = {'image_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        prediction = model.predict_forward(**data_batch, tokenizer=tokenizer)['prediction']

        text_output = prediction.replace("<s>", "").replace("\n", "") \
            .replace("region1", '').replace("Region1", '').replace("The region marked by", "").replace("The region marked as", "").replace("The region marked", "") \
            .replace("is", "").replace("shows", "").replace(':', '').replace("   ", " ").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]
        cleaned_str = re.sub(r'<.*?>', '', text_output)
        cleaned_str = cleaned_str.replace('[SEG]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()

        result_dict["caption"] = caption
        result_dict["prediction"] = cleaned_str
        print("#####", caption)
        print("@@@@@",cleaned_str)
        results.append(result_dict)

    tmpdir = './dist_test_temp_regioncap_' + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        with open(args.output_path, 'w') as json_file:
            json.dump(results, json_file, indent=2)

if __name__ == '__main__':
    main()
