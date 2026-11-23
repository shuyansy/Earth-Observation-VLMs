"""EOFuse-Bench evaluation of EarthMind checkpoints (GRSM paper, Table I).

Three input protocols (`--setting`): optical | sar | fused, on the strictly co-registered pairs.
Tasks (`--tasks`):
  mcq : Scene Classification (SC), Object Existence (OE), Hallucination Detection (HD),
        Object Counting (OC), Directional Reasoning (DR)                     -> accuracy
  oe  : Image Caption, Disaster Forecasting, Route Planning, Urban Assessment -> predictions
        (score them with the GPT judge in gpt_judge/: make_inputs.py, then judge.py run)
  seg : Referring Expression Segmentation (RS)                                -> mIoU

Sharded inference (one process per GPU), then merge:
  python eval_eofuse.py --model_path M --setting fused --out_dir O --shard 0 --num_shards 4 &
  ...
  python eval_eofuse.py --model_path M --setting fused --out_dir O --merge
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

MCQ_FILES = {
    'SC': 'scene_classification_unmatched.json',
    'OE': 'object_existence_all_unmatched.json',
    'HD': 'hallucination_detection_all_unmatched.json',
    'OC': 'object_counting_all_unmatched.json',
    'DR': 'spatial_relationship_all_unmatched.json',
}
OE_FILES = {
    'caption': 'caption_all_unmatched.json',
    'disaster': 'disaster_all_unmatched.json',
    'route': 'Navigation_all_unmatched.json',
    'urban': 'urban_all_unmatched.json',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--setting', choices=['optical', 'sar', 'fused'], default='fused')
    p.add_argument('--tasks', default='mcq,seg')
    p.add_argument('--bench_root', default='data/EOFuse-Bench')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--max_per_task', type=int, default=0, help='debug: limit items per task')
    p.add_argument('--attn_mode', default=None, help='override (test-time ablation)')
    p.add_argument('--pos_mode', default=None)
    p.add_argument('--sar_pad', default=None, help='zero | replicate (default: model config)')
    p.add_argument('--legacy_fusion', action='store_true',
                   help='released checkpoints: use predict_forward_multi (weighted token fusion)')
    p.add_argument('--grounding', default='auto', help='SAM2 input for fused RS: auto|sar|optical')
    p.add_argument('--return_mag', action='store_true', help='also record Eq. (1) MAG for MCQ items')
    p.add_argument('--ms_qa_file', default=None,
                   help='multispectral scene-classification QA json (ms_scene_classification_qa.json, '
                        'not released); images from --ms_root/{ms,sar}')
    p.add_argument('--ms_root', default='data/pair_data/test/ms_sar')
    return p.parse_args()


# ------------------------------------------------------------------------------------------
def load_model(path, sar_pad=None):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    legacy = getattr(cfg, 'gesa_fusion_mode', 'legacy') == 'legacy'
    if legacy:
        cfg.gesa_force_sdpa = True
        cfg.gesa_sar_pad_mode = sar_pad or 'replicate'    # released models: convert('RGB')
    elif sar_pad:
        cfg.gesa_sar_pad_mode = sar_pad
    model = AutoModel.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tok)
    return model, tok, legacy


def letter(pred):
    s = pred.replace('<|end|>', '').strip()
    m = re.match(r'^\(?([A-Da-d])(?:[\.\)\s:,]|$)', s)
    return m.group(1).upper() if m else (s[:1].upper() if s[:1].upper() in 'ABCD' else '')


def run_model(model, tok, args, legacy, rgb, sar, text):
    use_opt = args.setting in ('optical', 'fused')
    use_sar = args.setting in ('sar', 'fused')
    if legacy and args.legacy_fusion and args.setting == 'fused':
        return model.predict_forward_multi(image=sar.convert('RGB'), rgb_image=rgb, text=text, tokenizer=tok)
    return model.predict_forward_gesa(optical=rgb if use_opt else None, sar=sar if use_sar else None,
                                      text=text, tokenizer=tok, attn_mode=args.attn_mode,
                                      pos_mode=args.pos_mode, grounding=args.grounding,
                                      return_mag=args.return_mag)


def img_pair(root, file_name):
    stem = os.path.splitext(file_name)[0]
    rgb = Image.open(os.path.join(root, 'img/test/rgb/img', stem + '.png')).convert('RGB')
    sar = Image.open(os.path.join(root, 'img/test/sar/img', stem + '.png'))
    return rgb, sar


# ------------------------------------------------------------------------------------------
def build_items(args):
    items = []
    tasks = args.tasks.split(',')
    if 'mcq' in tasks:
        for t, f in MCQ_FILES.items():
            data = json.load(open(os.path.join(args.bench_root, 'json', f)))
            if args.max_per_task:
                data = data[:args.max_per_task]
            for i, d in enumerate(data):
                items.append(dict(kind='mcq', task=t, idx=i, d=d))
    if 'mcq' in tasks and args.ms_qa_file:
        data = json.load(open(args.ms_qa_file))
        if args.max_per_task:
            data = data[:args.max_per_task]
        for i, d in enumerate(data):
            items.append(dict(kind='mcq', task='SC-MS', idx=i, d=d))
    if 'oe' in tasks:
        for t, f in OE_FILES.items():
            data = json.load(open(os.path.join(args.bench_root, 'json', f)))
            if args.max_per_task:
                data = data[:args.max_per_task]
            for i, d in enumerate(data):
                items.append(dict(kind='oe', task=t, idx=i, d=d))
    if 'seg' in tasks:
        seg_root = os.path.join(args.bench_root, 'json', 'segmentation')
        refs = [r for r in pickle.load(open(os.path.join(seg_root, 'refs(unc).p'), 'rb')) if r['split'] == 'test']
        if args.max_per_task:
            refs = refs[:args.max_per_task]
        for i, r in enumerate(refs):
            for j, s in enumerate(r['sentences']):
                items.append(dict(kind='seg', task='RS', idx=f'{i}_{j}', d=dict(ref=r, sent=s['sent'])))
    return items


def seg_gt(ann, h, w):
    from pycocotools import mask as mask_utils
    seg = ann['segmentation']
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        m = mask_utils.decode(rles)
        m = m.sum(-1) if m.ndim == 3 else m
    else:
        m = mask_utils.decode(seg)
    return (m > 0).astype(np.uint8)


def infer(args):
    from pycocotools import mask as mask_utils
    model, tok, legacy = load_model(args.model_path, args.sar_pad)
    items = build_items(args)[args.shard::args.num_shards]
    anns = None
    if any(it['kind'] == 'seg' for it in items):
        ins = json.load(open(os.path.join(args.bench_root, 'json/segmentation/instances.json')))
        anns = {a['id']: a for a in ins['annotations']}
    out_path = os.path.join(args.out_dir, f'pred_{args.setting}_shard{args.shard}of{args.num_shards}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r['kind'], r['task'], str(r['idx'])))
    fout = open(out_path, 'a')
    t0 = time.time()
    for n, it in enumerate(items):
        key = (it['kind'], it['task'], str(it['idx']))
        if key in done:
            continue
        d = it['d']
        rec = dict(kind=it['kind'], task=it['task'], idx=it['idx'])
        try:
            if it['task'] == 'SC-MS':
                from projects.llava_sam2.hf.models.gesa_utils import ms_bands_to_pseudo_frames
                stem = os.path.splitext(d['file_name'])[0]
                bands = sorted(glob.glob(os.path.join(args.ms_root, 'ms', f'{stem}_*.tif*')) +
                               glob.glob(os.path.join(args.ms_root, 'ms', 'img', f'{stem}_*.tif*')))
                rgb = ms_bands_to_pseudo_frames(bands)            # K = 4 pseudo-frames
                sar_p = [q for q in (os.path.join(args.ms_root, 'sar', stem + '.png'),
                                     os.path.join(args.ms_root, 'sar', 'img', stem + '.png')) if os.path.exists(q)][0]
                sar = Image.open(sar_p)
                opts = d.get('candidate', [])
                text = '<image>' + d['question'] + ('\n' + '\n'.join(opts) if opts else '')
                res = run_model(model, tok, args, legacy, rgb, sar, text)
                pred = res['prediction'].replace('<|end|>', '').strip()
                rec.update(pred=pred, gt=d['answer'], question=d['question'], file_name=d['file_name'],
                           strict=pred == d['answer'], correct=pred == d['answer'])
            elif it['kind'] in ('mcq', 'oe'):
                rgb, sar = img_pair(args.bench_root, d['file_name'])
                if it['kind'] == 'mcq':
                    opts = d.get('candidate', [])
                    text = '<image>' + d['question'] + ('\n' + '\n'.join(opts) if opts else '') + \
                           '\nonly output the correct option.'
                else:
                    text = '<image>' + d['question']
                res = run_model(model, tok, args, legacy, rgb, sar, text)
                pred = res['prediction'].replace('<|end|>', '').strip()
                rec.update(pred=pred, gt=d['answer'], question=d['question'], file_name=d['file_name'])
                if it['kind'] == 'mcq':
                    rec.update(strict=pred == d['answer'], correct=letter(pred) == d['answer'].strip()[:1].upper())
                if 'mag' in res:
                    rec['mag'] = res['mag']
            else:
                ref, sent = d['ref'], d['sent']
                rgb, sar = img_pair(args.bench_root, 'SPACE6_' + ref['file_name'])
                w, h = rgb.size
                gt = seg_gt(anns[ref['ann_id']], h, w)
                res = run_model(model, tok, args, legacy, rgb, sar,
                                '<image>\n Please segment {} in this image.'.format(sent))
                masks = res['prediction_masks']
                if len(masks) == 0:
                    pred_m = None
                else:
                    pm = masks[0]
                    pred_m = (pm[0] if pm.ndim == 3 else pm).astype(np.uint8)
                    if pred_m.shape != gt.shape:
                        pred_m = np.array(Image.fromarray(pred_m).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
                inter_fg = int(((pred_m > 0) & (gt > 0)).sum()) if pred_m is not None else 0
                union_fg = int(((pred_m > 0) | (gt > 0)).sum()) if pred_m is not None else int(gt.sum())
                if pred_m is not None:
                    inter_bg = int(((pred_m == 0) & (gt == 0)).sum())
                    union_bg = int(((pred_m == 0) | (gt == 0)).sum())
                else:
                    inter_bg = union_bg = 0
                rec.update(sent=sent, pred_text=res['prediction'], has_mask=pred_m is not None,
                           inter_fg=inter_fg, union_fg=union_fg, inter_bg=inter_bg, union_bg=union_bg,
                           gt_area=int(gt.sum()))
        except Exception as e:  # keep going, record the failure
            rec.update(error=repr(e))
            print('ERROR', key, repr(e), file=sys.stderr)
        fout.write(json.dumps(rec) + '\n')
        fout.flush()
        if n % 50 == 0:
            print(f'[shard {args.shard}] {n}/{len(items)} {time.time() - t0:.0f}s', flush=True)
    fout.close()


# ------------------------------------------------------------------------------------------
def merge(args):
    recs = []
    for f in sorted(glob.glob(os.path.join(args.out_dir, f'pred_{args.setting}_shard*.jsonl'))):
        recs += [json.loads(l) for l in open(f)]
    res = dict(setting=args.setting, model_path=args.model_path, n_records=len(recs),
               n_errors=sum('error' in r for r in recs))
    mcq = {}
    for t in list(MCQ_FILES) + ['SC-MS']:
        rs = [r for r in recs if r['kind'] == 'mcq' and r['task'] == t and 'error' not in r]
        if rs:
            mcq[t] = dict(n=len(rs), acc=100 * np.mean([r['correct'] for r in rs]),
                          acc_strict=100 * np.mean([r['strict'] for r in rs]))
            mags = [r['mag']['MAG'] for r in rs if 'mag' in r]
            if mags:
                mcq[t]['MAG'] = float(np.nanmean(mags))
    if mcq:
        res['mcq'] = mcq
        res['M-Avg(5)'] = float(np.mean([v['acc'] for k, v in mcq.items() if k in MCQ_FILES]))
        if 'SC-MS' in mcq:   # paper: SC = RGB-SAR scene items + multispectral scene items (458)
            n1, n2 = mcq['SC']['n'], mcq['SC-MS']['n']
            mcq['SC+MS'] = dict(n=n1 + n2, acc=(mcq['SC']['acc'] * n1 + mcq['SC-MS']['acc'] * n2) / (n1 + n2))
    seg = [r for r in recs if r['kind'] == 'seg' and 'error' not in r]
    if seg:
        iou_fg = np.array([r['inter_fg'] / r['union_fg'] if r['union_fg'] > 0 else 1.0 for r in seg])
        has = np.array([r['has_mask'] for r in seg])
        iou_bg = np.array([r['inter_bg'] / r['union_bg'] if r['union_bg'] > 0 else 1.0 for r in seg])
        res['seg'] = dict(
            n=len(seg), no_seg_token=int((~has).sum()),
            # definition of the released evaluation code (RES.py): per-sample mean of background and
            # foreground IoU, samples without a [SEG] mask skipped
            mIoU_released=100 * float(np.mean(((iou_fg + iou_bg) / 2)[has])) if has.any() else 0.0,
            mIoU_fg=100 * float(iou_fg.mean()),                # foreground, missing mask = 0 IoU
            cIoU=100 * sum(r['inter_fg'] for r in seg) / max(1, sum(r['union_fg'] for r in seg)))
    oe = {}
    for t in OE_FILES:
        rs = [r for r in recs if r['kind'] == 'oe' and r['task'] == t and 'error' not in r]
        if rs:
            oe[t] = dict(n=len(rs))
    if oe:
        res['oe_generated'] = oe
    path = os.path.join(args.out_dir, f'metrics_{args.setting}.json')
    json.dump(res, open(path, 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    merge(a) if a.merge else infer(a)
