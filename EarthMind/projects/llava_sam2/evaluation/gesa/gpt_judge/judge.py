#!/usr/bin/env python3
"""GPT-judge scoring for EarthMind (GeSA) evaluations. Python 3.6+, standard library only.

    export JUDGE_API_KEY=<key>          # key of an OpenAI-compatible endpoint (LiteLLM, OpenAI, ...)
    python3 judge.py models             # list the models the endpoint serves
    python3 judge.py run                # judge everything in inputs/ (resumable: just run it again)
    python3 judge.py run --model gpt-4o --workers 16
    python3 judge.py summary            # recompute outputs/summary.json from outputs/

Endpoint: --base_url or $JUDGE_BASE_URL (any OpenAI-compatible /v1 API), default
https://api.openai.com/v1. Proxy environment variables are ignored unless JUDGE_USE_PROXY=1.

Judges
  oe      EOFuse-Bench open-ended tasks (caption / disaster / route / urban) with the GPT-4 prompt of
          the paper (appendix, "GPT-4 Evaluation Prompt for Open-Ended Tasks"): accuracy and relevance
          scores 1-5. O-Avg (Table I) = mean accuracy score over the four tasks.
  vrsvqa  VRSBench-VQA with the rules of the released evaluate_vrsbench_vqa.py: correct if the ground
          truth is contained in the answer; yes/no/number answers must match exactly; every other
          answer is sent to the judge ("1" match / "0" no match). Accuracy over all questions.

Files: inputs/<judge>__<model>[__<setting>].jsonl -> outputs/<same name>.jsonl (one line per judged
item, the last line of an id wins) and outputs/summary.json.
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, 'inputs')
OUT_DIR = os.path.join(HERE, 'outputs')
DEFAULT_BASE = 'https://api.openai.com/v1'

# identical to evaluation/gesa/score_oe_gpt.py (the paper's prompt)
OE_PROMPT = """### Task Description:
You are required to evaluate a respondent's answer based on a provided question, some scoring points, and the respondent's answer. You should provide two scores. The first is the accuracy score, which should range from 1 to 5. The second is the relevance score, which should also range from 1 to 5. Below are the criteria for each scoring category.
### Scoring Criteria:
Please rate the similarity between the predicted caption and the ground truth based on the following criteria:
- 1 - Completely unrelated (content is very different)
- 2 - Slightly related, but most descriptions do not match
- 3 - Somewhat similar, with a few common details but also clear differences
- 4 - Mostly matching, only a few minor differences
- 5 - Highly consistent, both descriptions describe the same content in detail
### Question:
{question}
### Scoring points (ground truth):
{gt}
### Respondent's answer:
{pred}
## INSTRUCTION:
Output Scores in JSON Format: Present the scores in a strictly structured JSON format as follows:
{{ "accuracy_score": [1-5], "relevance_score": [1-5] }}"""

# identical to the released evaluation/evaluate_vrsbench_vqa.py (lower-cased ground truth / answer)
VQA_PROMPT = ("Question: {question}\nGround Truth Answer: {gt}\nPredicted Answer: {pred}\n"
              "Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. "
              "Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football "
              "and soccer, playground and ground track field, building and rooftop, pond and swimming pool. "
              "Do not explain the reason.\n")

# --model auto: first pattern with a hit wins (GPT-4o, as in the released VRSBench judge, then GPT-4)
PREFERRED = [r'gpt-4o(-\d{4}-\d{2}-\d{2})?', r'gpt-4o\b.*', r'gpt-4(-\d{4})?', r'gpt-4-turbo.*', r'gpt-4\b.*']
NOT_A_JUDGE = ('mini', 'nano', 'audio', 'realtime', 'transcribe', 'tts', 'search', 'image', 'embedding')


class Client(object):
    def __init__(self, base, key, timeout):
        self.base = base.rstrip('/')
        self.key = key
        self.timeout = timeout
        # direct connection unless JUDGE_USE_PROXY=1 (then http(s)_proxy from the environment are used)
        proxies = None if os.environ.get('JUDGE_USE_PROXY') == '1' else {}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        self.plain = False   # True once the model rejected temperature / max_tokens

    def _call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.base + path, data=data)
        req.add_header('Authorization', 'Bearer ' + self.key)
        if data is not None:
            req.add_header('Content-Type', 'application/json')
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'replace')
            raise HTTPFailure(e.code, body)

    def models(self):
        return sorted(m['id'] for m in self._call('/models').get('data', []))

    def chat(self, model, prompt, max_tokens):
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}]}
        if not self.plain:
            payload['temperature'] = 0.0
            payload['max_tokens'] = max_tokens
        try:
            out = self._call('/chat/completions', payload)
        except HTTPFailure as e:
            if e.code == 400 and not self.plain and re.search(r'temperature|max_tokens|unsupported', e.body, re.I):
                self.plain = True     # e.g. reasoning models: retry without sampling parameters
                return self.chat(model, prompt, max_tokens)
            raise
        return out['choices'][0]['message'].get('content') or ''


class HTTPFailure(Exception):
    def __init__(self, code, body):
        Exception.__init__(self, 'HTTP %d: %s' % (code, body[:300]))
        self.code, self.body = code, body


def parse_oe(txt):
    a = re.search(r'accuracy_score"?\s*:\s*\[?\s*"?([1-5])', txt)
    r = re.search(r'relevance_score"?\s*:\s*\[?\s*"?([1-5])', txt)
    return (int(a.group(1)), int(r.group(1))) if a and r else None


def parse_vqa(txt):
    s = txt.strip()
    if s[:1] in ('0', '1'):
        return int(s[0])
    m = re.search(r'\b([01])\b', s)
    return int(m.group(1)) if m else None


def vqa_rule(gt, pred):
    """Released rule part: 1 / 0, or None when the judge has to decide."""
    gt, pred = gt.lower(), pred.lower()
    if gt in pred:
        return 1
    if gt in ['yes', 'no'] + [str(i) for i in range(100)]:
        return int(gt == pred)
    return None


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def latest(path):
    return {r['id']: r for r in load_jsonl(path)} if os.path.exists(path) else {}


def judge_kind(name):
    return name.split('__')[0]


def ask(client, model, prompt, max_tokens, parse, tries):
    err = raw = None
    for t in range(tries):
        try:
            raw = client.chat(model, prompt, max_tokens)
            val = parse(raw)
            if val is not None:
                return val, raw, None
            err = 'unparsable answer'
        except HTTPFailure as e:
            err = str(e)
            if e.code in (401, 403):
                raise
        except Exception as e:      # timeouts, connection resets, ...
            err = repr(e)[:300]
        if t + 1 < tries:
            time.sleep(min(60, 2 ** (t + 1)))
    return None, raw, err


def pick_model(a, client):
    if a.model != 'auto':
        return a.model
    try:
        names = client.models()
    except HTTPFailure as e:
        if e.code in (401, 403):
            sys.exit('The endpoint rejected the API key (%s).' % e)
        sys.exit('Could not list the models (%s). Pass --model <name>.' % e)
    except Exception as e:
        sys.exit('Could not reach %s (%s).' % (client.base, e))
    for pat in PREFERRED:
        hits = [m for m in names if re.fullmatch(pat, m, re.I) and not any(w in m.lower() for w in NOT_A_JUDGE)]
        if hits:
            return sorted(hits, key=len)[0]
    sys.exit('No GPT-4o / GPT-4 model served here. Available: %s\nPass --model <name>.' % ', '.join(names))


def cmd_run(a, client):
    model = pick_model(a, client)
    try:   # fail fast on a wrong key / model name
        client.chat(model, 'Reply with the digit 1.', 5)
    except Exception as e:
        sys.exit('Test request to %s with model %s failed: %s' % (client.base, model, e))
    print('judge model: %s   endpoint: %s' % (model, client.base), flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = []
    for f in sorted(glob.glob(os.path.join(IN_DIR, '*.jsonl'))):
        name = os.path.basename(f)[:-len('.jsonl')]
        if a.only and not any(s in name for s in a.only.split(',')):
            continue
        out = os.path.join(OUT_DIR, name + '.jsonl')
        have = latest(out)
        todo = []
        for it in load_jsonl(f):
            prev = have.get(it['id'])
            if prev is not None and prev.get('error') is None:
                continue
            if judge_kind(name) == 'vrsvqa' and vqa_rule(it['gt'], it['pred']) is not None:
                continue
            todo.append(it)
        if a.limit:
            todo = todo[:a.limit]
        print('%-45s %6d to judge' % (name, len(todo)), flush=True)
        jobs += [(name, out, it) for it in todo]
    if not jobs:
        print('nothing to judge')
        return

    lock = threading.Lock()
    stats = dict(done=0, failed=0)
    t0 = time.time()

    def work(job):
        name, out, it = job
        if judge_kind(name) == 'oe':
            val, raw, err = ask(client, model, OE_PROMPT.format(question=it['question'], gt=it['gt'], pred=it['pred']),
                                200, parse_oe, a.tries)
            rec = dict(id=it['id'], task=it['task'], accuracy=val[0] if val else None,
                       relevance=val[1] if val else None)
        else:
            val, raw, err = ask(client, model, VQA_PROMPT.format(question=it['question'], gt=it['gt'].lower(),
                                                                 pred=it['pred'].lower()),
                                100, parse_vqa, a.tries)
            rec = dict(id=it['id'], type=it.get('type'), correct=val)
        rec.update(raw=raw, error=err, model=model)
        with lock:
            with open(out, 'a', encoding='utf-8') as fo:
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
            stats['done'] += 1
            stats['failed'] += err is not None
            n = stats['done']
            if n % 200 == 0 or n == len(jobs):
                el = time.time() - t0
                print('%d / %d judged, %d failed, %.0f s elapsed, ~%.0f min left'
                      % (n, len(jobs), stats['failed'], el, el / n * (len(jobs) - n) / 60), flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fu in as_completed(futs):
            try:
                fu.result()
            except HTTPFailure as e:        # 401 / 403: stop everything
                print('fatal: %s' % e, flush=True)
                os._exit(2)
    cmd_summary()
    if stats['failed']:
        print('%d items failed; run the same command again to retry them.' % stats['failed'])


def mean(v):
    v = list(v)
    return round(sum(v) / len(v), 4) if v else None


def cmd_summary(a=None, client=None):
    summ = {'oe': {}, 'vrsvqa': {}}
    for f in sorted(glob.glob(os.path.join(IN_DIR, '*.jsonl'))):
        name = os.path.basename(f)[:-len('.jsonl')]
        items = load_jsonl(f)
        res = latest(os.path.join(OUT_DIR, name + '.jsonl'))
        models = sorted({r['model'] for r in res.values() if r.get('model')})
        key = name.split('__', 1)[1]
        if judge_kind(name) == 'oe':
            by_task, missing = {}, 0
            for it in items:
                r = res.get(it['id'])
                if r is None or r.get('accuracy') is None:
                    missing += 1
                    by_task.setdefault(it['task'], [])
                    continue
                by_task.setdefault(it['task'], []).append((r['accuracy'], r['relevance']))
            tasks = {t: dict(n=len(v), accuracy=mean(x for x, _ in v), relevance=mean(y for _, y in v))
                     for t, v in sorted(by_task.items())}
            o_avg = mean(t['accuracy'] for t in tasks.values()) if all(t['n'] for t in tasks.values()) else None
            summ['oe'][key] = {'O-Avg': o_avg, 'tasks': tasks, 'n': len(items), 'missing': missing, 'judge': models}
        else:
            correct, n_gpt, unresolved, per_type = 0, 0, 0, {}
            for it in items:
                c = vqa_rule(it['gt'], it['pred'])
                if c is None:
                    n_gpt += 1
                    r = res.get(it['id'])
                    c = r.get('correct') if r else None
                    if c is None:
                        unresolved += 1
                        c = 0
                correct += c
                per_type.setdefault(it.get('type'), []).append(c)
            summ['vrsvqa'][key] = {
                'accuracy': round(100.0 * correct / len(items), 2), 'n': len(items), 'judged_by_gpt': n_gpt,
                'unresolved_counted_wrong': unresolved, 'judge': models,
                'per_type': {t: round(100.0 * sum(v) / len(v), 2) for t, v in sorted(per_type.items(), key=str)}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w', encoding='utf-8') as fo:
        json.dump(summ, fo, indent=2, ensure_ascii=False)
    for k, v in summ['oe'].items():
        print('OE      %-32s O-Avg %s  %s  missing %d' % (
            k, v['O-Avg'], ' '.join('%s %s' % (t, d['accuracy']) for t, d in v['tasks'].items()), v['missing']))
    for k, v in summ['vrsvqa'].items():
        print('VRS-VQA %-32s acc %.2f  (%d judged by GPT, %d unresolved)' % (
            k, v['accuracy'], v['judged_by_gpt'], v['unresolved_counted_wrong']))
    print('-> %s' % os.path.join(OUT_DIR, 'summary.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['models', 'run', 'summary'])
    ap.add_argument('--model', default='auto', help='judge model name (default: GPT-4o / GPT-4 served by the endpoint)')
    ap.add_argument('--workers', type=int, default=16, help='parallel requests')
    ap.add_argument('--only', default='', help='comma-separated substrings of input names, e.g. main_gesa')
    ap.add_argument('--limit', type=int, default=0, help='judge at most N items per input file (smoke test)')
    ap.add_argument('--tries', type=int, default=6)
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--base_url', default=os.environ.get('JUDGE_BASE_URL', DEFAULT_BASE))
    a = ap.parse_args()
    if a.cmd == 'summary':
        return cmd_summary()
    key = os.environ.get('JUDGE_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('Set JUDGE_API_KEY (or OPENAI_API_KEY) first.')
    client = Client(a.base_url, key, a.timeout)
    if a.cmd == 'models':
        print('\n'.join(client.models()))
    else:
        cmd_run(a, client)


if __name__ == '__main__':
    main()
