"""Sharded image-level evaluation (Table II): UC-Merced classification (as in the released classify.py)
and VRSBench-VQA (as in the released infer_vrsbench_vqa.py). Prompts, the `predict_forward(video=[img])`
call (one 448px tile) and the answer normalisation are identical to the released scripts.

  python table2_image_eval.py --model_path M --task ucm --out_dir O --shard i --num_shards n
  python table2_image_eval.py --model_path M --task ucm --out_dir O --merge
VRSBench-VQA: --merge reports exact match. The released metric additionally asks a GPT judge about
non-identical answers: pack the predictions with gpt_judge/make_inputs.py and score them with judge.py.
"""
import argparse
import glob
import json
import os

TASKS = {
    'ucm': dict(ann='data/GeoChat-Bench/UCmerced.jsonl', img='data/UCMerced_LandUse/Images', fmt='jsonl'),
    'vrsbench_vqa': dict(ann='data/vrsbenchvqa/VRSBench_EVAL_vqa.json', img='data/vrsbenchvqa/Images_val',
                         fmt='json'),
}


def load_items(task):
    t = TASKS[task]
    if t['fmt'] == 'jsonl':
        return [json.loads(l) for l in open(t['ann'])]
    return json.load(open(t['ann']))


def infer(a):
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype='auto', device_map='cuda:0',
                                                 trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    items = load_items(a.task)[a.shard::a.num_shards]
    out = os.path.join(a.out_dir, f'{a.task}_shard{a.shard}of{a.num_shards}.jsonl')
    done = {json.loads(l)['k'] for l in open(out)} if os.path.exists(out) else set()
    f = open(out, 'a')
    for n, it in enumerate(items):
        if a.task == 'vrsbench_vqa':
            k = str(it['question_id'])
            img_file = it['image_id']
            text = '<image>' + it['question'] + 'Answer the question using a single word or phrase.'
        else:
            k = str(it['question_id'])
            img_file = it['image']
            text = '<image>' + it['text']
        if k in done:
            continue
        img = Image.open(os.path.join(TASKS[a.task]['img'], img_file)).convert('RGB')
        with torch.no_grad():
            pred = model.predict_forward(video=[img], text=text, tokenizer=tok)['prediction'].replace('<|end|>', '')
        rec = dict(k=k, answer=pred)
        if a.task == 'ucm':
            rec['gt'] = it['question_id'].split('/')[0]
        else:
            rec.update(gt=it['ground_truth'], type=it.get('type'), question=it['question'])
        f.write(json.dumps(rec) + '\n')
        f.flush()
        if n % 500 == 0:
            print(f'[{a.task} shard {a.shard}] {n}/{len(items)}', flush=True)


def merge(a):
    recs = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.out_dir, f'{a.task}_shard*.jsonl'))) for l in open(p)]
    res = dict(task=a.task, n=len(recs), model_path=a.model_path)
    if a.task == 'ucm':   # classify.py: answer.replace(' ','').lower().replace('.','') == gt.lower()
        res['acc'] = 100 * sum(r['answer'].replace(' ', '').lower().replace('.', '') == r['gt'].lower()
                               for r in recs) / max(1, len(recs))
    else:
        res['acc_exact_match'] = 100 * sum(r['answer'].strip().lower() == r['gt'].strip().lower() for r in recs) / max(1, len(recs))
    json.dump(res, open(os.path.join(a.out_dir, f'metrics_{a.task}.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--task', choices=list(TASKS), required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    merge(a) if a.merge else infer(a)
