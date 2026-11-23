"""BigEarthNet-S2 multi-label classification (Table V, multispectral column).

Inference follows the released `evaluation/infer_ms.py`: prompt "<image>" + question, greedy
decoding. The 12 Sentinel-2 bands become K = 4 pseudo-RGB frames (one 448px tile each) placed in
one visual block.

  --grouping released : RGB | B05 B06 B07 | B8A B11 B12 | B01 B09 B09   (`MS_LLaVADataset`, used to
                        train the released EarthMind-4B; B08 unused, last group padded by repetition)
  --grouping paper    : RGB | B01 B05 B06 | B07 B08 B8A | B09 B11 B12   ("every three consecutive bands",
                        `gesa_utils.ms_bands_to_pseudo_frames`, used to train EarthMind-4B-GeSA; default)

Attention / positions: the checkpoint's own (legacy = causal + sequential; GeSA = config), or forced
with --attn_mode / --pos_mode.

  python ms_bigearthnet_eval.py --model_path M --out_dir O --grouping paper --shard i --num_shards n
  python ms_bigearthnet_eval.py --out_dir O --merge

Metrics (merge): token recall of the released `evaluation/ms_eval.py` (nltk word_tokenize; the
script's active metric), exact set match of option letters, sample-averaged F1, micro / macro F1.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '../../../..')))

LETTERS = [chr(c) for c in range(ord('A'), ord('S') + 1)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path')
    p.add_argument('--ann', default='data/multi-spectrum/ms_test.jsonl')
    p.add_argument('--img_dir', default='data/multi-spectrum/images')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--grouping', choices=['released', 'paper'], default='paper')
    p.add_argument('--attn_mode', default=None)
    p.add_argument('--pos_mode', default=None)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0, help='random subset of this size (seed 0); 0 = all')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tag', default='bigearthnet')
    p.add_argument('--merge', action='store_true')
    return p.parse_args()


def load_items(a):
    items = [json.loads(l) for l in open(a.ann)]
    if a.limit:
        random.Random(0).shuffle(items)
        items = items[:a.limit]
    return items


def sample_folder(img_dir, image):
    name = image[:-4] if image.lower().endswith(('.png', '.jpg', '.tif')) else image
    return os.path.join(img_dir, name)


class FramesDataset:
    def __init__(self, items, img_dir, grouping):
        self.items, self.img_dir, self.grouping = items, img_dir, grouping

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        folder = sample_folder(self.img_dir, self.items[i]['image'])
        try:
            if self.grouping == 'released':
                from projects.llava_sam2.datasets.vqa_dataset import ms_folder_to_rgb_and_extra_groups
                frames = ms_folder_to_rgb_and_extra_groups(folder, to_uint8=True, pad_mode='repeat')
            else:
                from projects.llava_sam2.hf.models.gesa_utils import ms_folder_to_pseudo_frames
                frames = ms_folder_to_pseudo_frames(folder, group_size=3)
        except Exception as e:  # missing / broken sample
            return i, None, repr(e)
        return i, frames, None


def run_shard(a):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    items = load_items(a)
    idx = list(range(a.shard, len(items), a.num_shards))
    out_path = os.path.join(a.out_dir, f'{a.tag}_shard{a.shard}of{a.num_shards}.jsonl')
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)['idx'] for l in open(out_path)}
    todo = [i for i in idx if i not in done]
    print(f'[shard {a.shard}] {len(idx)} items, {len(done)} done, {len(todo)} to go', flush=True)
    cfg = AutoConfig.from_pretrained(a.model_path, trust_remote_code=True)
    if getattr(cfg, 'gesa_fusion_mode', 'legacy') == 'legacy':
        cfg.gesa_force_sdpa = True     # FlashAttention-2 cannot take the 4D prefill mask
    model = AutoModelForCausalLM.from_pretrained(a.model_path, config=cfg, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).cuda().eval()
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    print(f'[shard {a.shard}] fusion_mode={getattr(model, "gesa_fusion_mode", "legacy")} '
          f'attn={a.attn_mode} pos={a.pos_mode} grouping={a.grouping}', flush=True)
    sub = [items[i] for i in todo]
    if a.grouping == 'released':   # heavy package import once, inherited by the forked workers
        from projects.llava_sam2.datasets.vqa_dataset import ms_folder_to_rgb_and_extra_groups  # noqa: F401
    loader = torch.utils.data.DataLoader(FramesDataset(sub, a.img_dir, a.grouping), batch_size=None,
                                         shuffle=False, num_workers=a.workers, prefetch_factor=8,
                                         collate_fn=lambda x: x)
    t0, n = time.time(), 0
    with open(out_path, 'a') as f:
        for j, frames, err in loader:
            it = sub[j]
            rec = dict(idx=todo[j], id=it['id'], answer=it['answer'])
            if frames is None:
                rec.update(pred=None, error=err)
            else:
                res = model.predict_forward_gesa(optical=frames, text='<image>' + it['question'], tokenizer=tok,
                                                 attn_mode=a.attn_mode, pos_mode=a.pos_mode)
                rec['pred'] = res['prediction'].replace('<|end|>', '').strip()
                rec['n_frames'] = len(frames)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
            if n % 500 == 0:
                f.flush()
                print(f'[shard {a.shard}] {n}/{len(todo)} {(time.time() - t0) / n:.3f}s/it', flush=True)
    print(f'[shard {a.shard}] finished {n} in {time.time() - t0:.0f}s', flush=True)


def letters(s):
    """Option letters of 'B. Arable land, D. Broad-leaved forest' or of a letter list 'B, D'.
    Comma-separated chunks that start with an option letter followed by '.' / ')' / end count;
    continuation chunks of class names that contain commas ('..., with significant areas') do not."""
    out = set()
    for chunk in (s or '').replace(';', ',').split(','):
        m = re.match(r'^\s*\(?([A-S])\s*(?:[.)]|$)', chunk)
        if m:
            out.add(m.group(1))
    return out


def token_recall(reference, candidate):
    """Released ms_eval.py: |set(ref tokens) & set(cand tokens)| / len(ref tokens)."""
    from nltk.tokenize import word_tokenize
    ref = word_tokenize(reference.strip().lower())
    cand = word_tokenize(candidate.strip().lower())
    common = set(ref) & set(cand)
    return len(common) / len(ref) if common else 0.0


def merge(a):
    recs = []
    for fp in sorted(glob.glob(os.path.join(a.out_dir, f'{a.tag}_shard*of*.jsonl'))):
        recs += [json.loads(l) for l in open(fp)]
    by_idx = {r['idx']: r for r in recs}
    recs = [by_idx[k] for k in sorted(by_idx)]
    n_err = sum(r['pred'] is None for r in recs)
    tp = np.zeros(len(LETTERS)); fp_ = np.zeros(len(LETTERS)); fn = np.zeros(len(LETTERS))
    recall_sum = recall_letters_sum = exact = f1_sum = srec_sum = sprec_sum = 0.0
    for r in recs:
        pred = r['pred'] or ''
        recall_sum += token_recall(r['answer'], pred)
        g, p = letters(r['answer']), letters(pred)
        # same token recall on letter-only renderings ("B, D, M, P") of answer and prediction
        recall_letters_sum += token_recall(', '.join(sorted(g)), ', '.join(sorted(p))) if g else 0.0
        exact += float(g == p)
        inter = len(g & p)
        f1_sum += 2 * inter / (len(g) + len(p)) if (g or p) else 1.0
        srec_sum += inter / len(g) if g else 1.0
        sprec_sum += inter / len(p) if p else 0.0
        for k, L in enumerate(LETTERS):
            tp[k] += (L in g) and (L in p)
            fp_[k] += (L not in g) and (L in p)
            fn[k] += (L in g) and (L not in p)
    n = max(1, len(recs))
    micro_f1 = 2 * tp.sum() / max(1, 2 * tp.sum() + fp_.sum() + fn.sum())
    per_cls = [2 * tp[k] / (2 * tp[k] + fp_[k] + fn[k]) for k in range(len(LETTERS)) if (tp[k] + fn[k]) > 0]
    res = dict(task=a.tag, n=len(recs), n_unreadable=n_err,
               token_recall_released=100 * recall_sum / n,
               token_recall_letter_strings=100 * recall_letters_sum / n,
               exact_set_match=100 * exact / n, sample_recall=100 * srec_sum / n,
               sample_precision=100 * sprec_sum / n, sample_f1=100 * f1_sum / n,
               micro_f1=100 * micro_f1, macro_f1=100 * float(np.mean(per_cls)),
               mean_n_gt=float(np.mean([len(letters(r['answer'])) for r in recs])),
               mean_n_pred=float(np.mean([len(letters(r['pred'])) for r in recs])))
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(a.out_dir, f'metrics_{a.tag}.json'), 'w'), indent=2)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.merge:
        merge(args)
    else:
        run_shard(args)
