#!/usr/bin/env python3
"""Collect the predictions that need a GPT judge into a self-contained package (judge.py + inputs/).

    python make_inputs.py --results results --out gpt_judge_run --run gesa:eofuse:table2:fused

--run  name:eofuse_dir:table2_dir:settings  (eofuse_dir / table2_dir relative to --results, may be empty)
Writes inputs/oe__<name>__<setting>.jsonl (EOFuse-Bench open-ended records of eval_eofuse.py) and
inputs/vrsvqa__<name>.jsonl (VRSBench-VQA records of table2_image_eval.py), copies judge.py next to them.
"""
import argparse
import glob
import json
import os
import shutil


def load(pattern):
    files = sorted(glob.glob(pattern))
    return [json.loads(l) for f in files for l in open(f, encoding='utf-8') if l.strip()]


def write(path, rows):
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate ids in %s' % path
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('%-60s %6d items' % (path, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', action='append', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'inputs'), exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'judge.py'), a.out)
    for spec in a.run:
        name, eo_dir, t2_dir, settings = spec.split(':')
        for s in filter(None, settings.split(',')):
            recs = [r for r in load(os.path.join(a.results, eo_dir, 'pred_%s_shard*.jsonl' % s))
                    if r.get('kind') == 'oe' and 'error' not in r]
            assert recs, 'no open-ended records for %s / %s' % (eo_dir, s)
            write(os.path.join(a.out, 'inputs', 'oe__%s__%s.jsonl' % (name, s)),
                  [dict(id='%s:%s' % (r['task'], r['idx']), task=r['task'], question=r['question'],
                        gt=r['gt'], pred=r['pred']) for r in recs])
        if t2_dir:
            recs = load(os.path.join(a.results, t2_dir, 'vrsbench_vqa_shard*.jsonl'))
            assert recs, 'no VRSBench-VQA records in %s' % t2_dir
            write(os.path.join(a.out, 'inputs', 'vrsvqa__%s.jsonl' % name),
                  [dict(id=str(r['k']), type=r.get('type'), question=r['question'], gt=r['gt'], pred=r['answer'])
                   for r in recs])


if __name__ == '__main__':
    main()
