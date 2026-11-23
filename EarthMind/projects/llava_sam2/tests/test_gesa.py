"""Unit tests for GeSA mask / position ids (run: python -m pytest -q projects/llava_sam2/tests)."""
import numpy as np
import torch
from PIL import Image

from projects.llava_sam2.hf.models.gesa_utils import (
    MOD_OPT, MOD_SAR, build_vis_meta, build_position_ids, build_allowed, prepare_gesa_inputs,
    average_colocated_tokens, sar_to_pseudo_rgb, tile_coregistered, tiles_layout_to_token_layout,
    layout_num_tokens)

IMG = 9  # fake <IMG_CONTEXT> id


def _seq(prefix=3, n_opt=4, n_sar=2, k=2, suffix=3, pad=2):
    # [t t t] [<img>-like prefix included] [opt k*N] [sar N] [t t t] [pad pad]
    N = n_sar
    ids = [1] * prefix + [IMG] * (k * N) + [IMG] * N + [2] * suffix + [0] * pad
    layout = [[MOD_OPT, N, k], [MOD_SAR, N, 1]]
    am = [1] * (len(ids) - pad) + [0] * pad
    return torch.tensor([ids]), [layout], torch.tensor([am]).bool()


def test_meta_and_shared_positions():
    ids, layouts, am = _seq(prefix=3, n_sar=2, k=2, suffix=3, pad=2)
    vis_mod, vis_sp = build_vis_meta(ids, IMG, layouts)
    assert vis_mod[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0]
    assert vis_sp[0].tolist() == [-1, -1, -1, 0, 1, 0, 1, 0, 1, -1, -1, -1, -1, -1]
    pos = build_position_ids(vis_sp, 'shared')[0].tolist()
    # Eq. (4): every frame of both sensors shares positions s+i; text resumes at s+N
    assert pos[:9] == [0, 1, 2, 3, 4, 3, 4, 3, 4]
    assert pos[9:12] == [5, 6, 7]
    assert build_position_ids(vis_sp, 'seq')[0].tolist() == list(range(14))


def test_masks():
    ids, layouts, am = _seq(prefix=3, n_sar=2, k=2, suffix=3, pad=2)
    vis_mod, _ = build_vis_meta(ids, IMG, layouts)
    L = ids.shape[1]
    causal = build_allowed(vis_mod, am, 'causal')[0]
    full = build_allowed(vis_mod, am, 'full')[0]
    intra = build_allowed(vis_mod, am, 'intra')[0]
    vis = list(range(3, 9)); opt = list(range(3, 7)); sar = [7, 8]; txt = [9, 10, 11]
    # causal: optical cannot see SAR
    assert not causal[3, 7]
    # full (Eq. 3): all visual tokens see each other
    assert all(full[i, j] for i in vis for j in vis)
    # intra: bidirectional only within one sensor; SAR still sees optical causally
    assert all(intra[i, j] for i in opt for j in opt)
    assert not intra[3, 7] and intra[7, 3]
    # text remains strictly causal in every mode
    for m in (causal, full, intra):
        for i in txt:
            assert m[i, : i + 1].all() and not m[i, i + 1:].any()
    # padded keys are never attended by real tokens; no row is empty
    for m in (causal, full, intra):
        assert not m[:12, 12:].any()
        assert m.any(-1).all()


def test_prepare_inputs_dtype_and_values():
    ids, layouts, am = _seq()
    pos, mask, vis_mod, vis_sp = prepare_gesa_inputs(ids, IMG, layouts, am, 'full', 'shared', torch.bfloat16)
    assert mask.shape == (1, 1, ids.shape[1], ids.shape[1]) and mask.dtype == torch.bfloat16
    assert mask.max() == 0 and mask.min() == torch.finfo(torch.bfloat16).min


def test_vp_runs_are_not_image_block():
    # image block followed by Osprey-like <vp> region tokens: only the first run is the image
    ids = torch.tensor([[1, IMG, IMG, IMG, 2, 5, IMG, IMG, 6, 3]])
    vis_mod, vis_sp = build_vis_meta(ids, IMG, None)
    assert vis_mod[0].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert build_position_ids(vis_sp, 'shared')[0].tolist() == list(range(10))


def test_average_colocated():
    N, C = 3, 2
    opt = torch.arange(2 * N * C, dtype=torch.float).view(2, N, C)      # K=2 frames
    sar = torch.full((1, N, C), 100.0)
    emb = torch.cat([opt, sar], 0)                                        # (3 tiles, N, C)
    out = average_colocated_tokens(emb, [[MOD_OPT, N, 2], [MOD_SAR, N, 1]], N)
    assert torch.allclose(out, (opt[0] + opt[1] + sar[0]) / 3)


def test_sar_zero_padding_and_tiling():
    sar = Image.fromarray(np.random.randint(0, 255, (900, 900), dtype=np.uint8), mode='L')
    rgb3 = Image.fromarray(np.stack([np.array(sar)] * 3, -1))
    for img in (sar, rgb3):
        p = np.array(sar_to_pseudo_rgb(img, 'zero'))
        assert p.shape == (900, 900, 3) and p[..., 1].max() == 0 and p[..., 2].max() == 0
        assert np.array_equal(p[..., 0], np.array(sar))
    r = np.array(sar_to_pseudo_rgb(sar, 'replicate'))
    assert np.array_equal(r[..., 0], r[..., 2])
    opt = Image.fromarray(np.random.randint(0, 255, (900, 900, 3), dtype=np.uint8))
    tiles, layout, n_tiles, grid = tile_coregistered([(MOD_OPT, [opt]), (MOD_SAR, [sar_to_pseudo_rgb(sar)])])
    assert grid == (2, 2) and n_tiles == 5 and len(tiles) == 10
    tl = tiles_layout_to_token_layout(layout, 256)
    assert tl == [[MOD_OPT, 1280, 1], [MOD_SAR, 1280, 1]] and layout_num_tokens(tl) == 2560
