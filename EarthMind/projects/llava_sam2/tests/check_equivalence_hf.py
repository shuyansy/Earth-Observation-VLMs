"""Equivalence checks of the GeSA code path against the standard HF path (GPU).

1) causal mask + sequential positions via our 4D mask  ==  HF 2D-mask causal attention
   (fp32 + eager attention, i.e. no kernel differences), and the bf16/SDPA error magnitudes;
2) our greedy decoder (gesa_greedy_generate) == HF generate on a few EOFuse-Bench prompts.

PYTHONPATH=. python projects/llava_sam2/tests/check_equivalence_hf.py <hf_model_dir>
"""
import json
import sys

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, 'projects/llava_sam2/evaluation/gesa')
from eval_eofuse import img_pair  # noqa: E402
from projects.llava_sam2.hf.models.gesa_utils import gesa_greedy_generate  # noqa: E402


def load(path, impl, dtype):
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cfg.gesa_force_sdpa = impl == 'sdpa'
    m = AutoModel.from_pretrained(path, config=cfg, torch_dtype=dtype, trust_remote_code=True,
                                  use_flash_attn=False).eval().cuda()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    m.preparing_for_generation(tokenizer=tok, torch_dtype=dtype)
    return m, tok


@torch.no_grad()
def logits_pair(m, inp):
    lm = m.language_model
    ref = lm(inputs_embeds=inp['inputs_embeds'], attention_mask=torch.ones_like(inp['input_ids']),
             position_ids=torch.arange(inp['input_ids'].shape[1], device='cuda')[None]).logits.float()
    new = lm(inputs_embeds=inp['inputs_embeds'], attention_mask=inp['attention_mask_4d'],
             position_ids=inp['position_ids']).logits.float()
    return ref, new


def main(path):
    d = json.load(open('data/EOFuse-Bench/json/object_counting_all_unmatched.json'))[:4]
    prompts = []
    for x in d:
        rgb, sar = img_pair('data/EOFuse-Bench', x['file_name'])
        prompts.append((rgb, '<image>' + x['question'] + '\n' + '\n'.join(x['candidate']) + '\nonly output the correct option.'))

    m, tok = load(path, 'eager', torch.float32)
    for rgb, text in prompts[:2]:
        inp = m.gesa_prepare_inputs(optical=rgb, text=text, attn_mode='causal', pos_mode='seq')
        ref, new = logits_pair(m, inp)
        print(f'[fp32 eager] L={ref.shape[1]} max|d logit|={(ref - new).abs().max():.2e}  '
              f'last-token max|d|={(ref[0, -1] - new[0, -1]).abs().max():.2e}')
    fp32 = {}
    for i, (rgb, text) in enumerate(prompts[:2]):
        inp = m.gesa_prepare_inputs(optical=rgb, text=text, attn_mode='causal', pos_mode='seq')
        fp32[i] = logits_pair(m, inp)[0].cpu()
    del m
    torch.cuda.empty_cache()

    m, tok = load(path, 'sdpa', torch.bfloat16)
    for i, (rgb, text) in enumerate(prompts[:2]):
        inp = m.gesa_prepare_inputs(optical=rgb, text=text, attn_mode='causal', pos_mode='seq')
        ref, new = logits_pair(m, inp)
        r32 = fp32[i].cuda()
        txt = inp['input_ids'][0] != m.img_context_token_id
        print(f'[bf16 sdpa] 2D-path vs fp32: max {(ref - r32).abs().max():.2f} (text tokens {(ref - r32)[0, txt].abs().max():.2f}); '
              f'4D-path vs fp32: max {(new - r32).abs().max():.2f} (text tokens {(new - r32)[0, txt].abs().max():.2f}); '
              f'argmax agree on text tokens 4D vs 2D: {(ref[0, txt].argmax(-1) == new[0, txt].argmax(-1)).float().mean():.4f}')
    # 2) decoding equivalence
    for rgb, text in prompts:
        inp = m.gesa_prepare_inputs(optical=rgb, text=text, attn_mode='causal', pos_mode='seq')
        ours = gesa_greedy_generate(m.language_model, inp['inputs_embeds'], inp['attention_mask_4d'],
                                    inp['position_ids'], max_new_tokens=16, eos_token_ids=[tok.eos_token_id])
        hf = m.language_model.generate(inputs_embeds=inp['inputs_embeds'],
                                       attention_mask=torch.ones_like(inp['input_ids']),
                                       max_new_tokens=16, do_sample=False, eos_token_id=tok.eos_token_id,
                                       pad_token_id=tok.eos_token_id)
        a, b = tok.decode(ours['sequences']), tok.decode(hf[0])
        print(f'[decode] ours={a!r}  hf={b!r}  same={a.strip() == b.strip()}')
    print('EQUIVALENCE CHECK DONE')


if __name__ == '__main__':
    main(sys.argv[1])
