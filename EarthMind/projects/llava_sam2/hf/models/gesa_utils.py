# --------------------------------------------------------
# GeoSync Attention (GeSA) utilities for EarthMind.
#
# Implements Sec. IV-C of "EarthMind: Leveraging Cross-Sensor Data for Advanced
# Earth Observation Interpretation with a Unified Multimodal LLM" (GRSM):
#   * Eq. (3)  bidirectional attention among all visual tokens, causal text,
#   * Eq. (4)  shared RoPE position ids for co-located optical / SAR tokens,
#   * Eq. (1)  Modality Attention Gap (MAG) diagnostics,
# plus the ablation variants of Table III (causal / intra-modal / full attention,
# sequential / shared position ids) and the sensor pre-processing of Sec. IV-A
# (zero-padded SAR pseudo-RGB, multispectral band triplets as pseudo-frames).
#
# This file is pure PyTorch / numpy / PIL with no relative imports so that it can
# be used both by the xtuner training code and by the HF remote-code model.
# --------------------------------------------------------
import copy
import math
import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MOD_TEXT, MOD_OPT, MOD_SAR = 0, 1, 2
ATTN_MODES = ('causal', 'intra', 'full')
POS_MODES = ('seq', 'shared')

# A visual layout describes the single contiguous block of <IMG_CONTEXT> tokens of one
# sample, in token order:  [(modality, tokens_per_frame, n_frames), ...]
#   RGB + SAR, T tiles      -> [(1, 256*T, 1), (2, 256*T, 1)]
#   MS (K frames) + SAR     -> [(1, 256*T, K), (2, 256*T, 1)]
#   single RGB image        -> [(1, 256*T, 1)]
# Token j of every frame of every modality covers the same ground location because
# co-registered inputs are tiled with the same grid.
Layout = Sequence[Sequence[int]]


# ----------------------------------------------------------------------------------
# Layout / token metadata
# ----------------------------------------------------------------------------------
def layout_num_tokens(layout: Optional[Layout]) -> int:
    if not layout:
        return 0
    return int(sum(int(t) * int(f) for _, t, f in layout))


def layout_num_tiles(layout: Optional[Layout], tokens_per_tile: int) -> int:
    return layout_num_tokens(layout) // tokens_per_tile


def layout_modalities(layout: Optional[Layout]) -> List[int]:
    return sorted({int(m) for m, _, _ in layout}) if layout else []


def _contiguous_runs(mask_1d: torch.Tensor) -> List[Tuple[int, int]]:
    """[(start, end)] of the True runs of a 1D bool tensor."""
    idx = torch.nonzero(mask_1d, as_tuple=False).flatten()
    if idx.numel() == 0:
        return []
    idx = idx.tolist()
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((s, prev + 1))
            s = i
        prev = i
    runs.append((s, prev + 1))
    return runs


def build_vis_meta(input_ids: torch.Tensor,
                   img_context_token_id: int,
                   layouts: Optional[Sequence[Optional[Layout]]] = None):
    """Per-token modality id and within-frame spatial index.

    Only the FIRST contiguous run of <IMG_CONTEXT> tokens of a sample is the image block;
    later runs (e.g. Osprey `<vp>` region tokens) are ordinary causal tokens.

    Returns:
        vis_mod (B, L) long: 0 text/other, 1 optical, 2 SAR
        vis_sp  (B, L) long: spatial index of visual tokens inside their frame, -1 otherwise
    """
    B, L = input_ids.shape
    device = input_ids.device
    vis_mod = torch.zeros(B, L, dtype=torch.long, device=device)
    vis_sp = torch.full((B, L), -1, dtype=torch.long, device=device)
    is_img = input_ids == img_context_token_id
    for b in range(B):
        runs = _contiguous_runs(is_img[b])
        if not runs:
            continue
        s, e = runs[0]
        n = e - s
        layout = layouts[b] if layouts is not None and b < len(layouts) else None
        if layout and layout_num_tokens(layout) == n:
            mods, sps = [], []
            for m, t, f in layout:
                for _ in range(int(f)):
                    mods.append(torch.full((int(t),), int(m), dtype=torch.long))
                    sps.append(torch.arange(int(t), dtype=torch.long))
            vis_mod[b, s:e] = torch.cat(mods).to(device)
            vis_sp[b, s:e] = torch.cat(sps).to(device)
        else:
            if layout:
                print(f'[GeSA] warning: layout {layout} has {layout_num_tokens(layout)} tokens '
                      f'but the image block has {n}; treating it as one optical frame.')
            vis_mod[b, s:e] = MOD_OPT
            vis_sp[b, s:e] = torch.arange(n, device=device)
    return vis_mod, vis_sp


# ----------------------------------------------------------------------------------
# Eq. (4): shared position ids
# ----------------------------------------------------------------------------------
def build_position_ids(vis_sp: torch.Tensor, pos_mode: str = 'shared') -> torch.Tensor:
    """'seq': 0..L-1.  'shared': pos(v_{k,i}^opt) = pos(v_i^sar) = s + i for the image block
    starting at s; text after the block resumes from s + W (W = tokens per frame)."""
    assert pos_mode in POS_MODES, pos_mode
    B, L = vis_sp.shape
    pos = torch.arange(L, device=vis_sp.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    if pos_mode == 'seq':
        return pos
    for b in range(B):
        runs = _contiguous_runs(vis_sp[b] >= 0)
        if not runs:
            continue
        s, e = runs[0]
        sp = vis_sp[b, s:e]
        W = int(sp.max().item()) + 1
        pos[b, s:e] = s + sp
        pos[b, e:] -= (e - s) - W
    return pos


# ----------------------------------------------------------------------------------
# Eq. (3): GeSA attention mask (and the Table-III variants)
# ----------------------------------------------------------------------------------
def build_allowed(vis_mod: torch.Tensor,
                  attention_mask: Optional[torch.Tensor],
                  attn_mode: str = 'full') -> torch.Tensor:
    """Boolean (B, L, L) matrix, True = query i may attend key j.

    allowed(i, j) = key_valid(j) and ( j <= i  or  bidir(i, j) )
        causal: bidir = False                                   (standard decoder, Table III a)
        intra : bidir = vis(i) & vis(j) & mod(i) == mod(j)      (Table III b/c)
        full  : bidir = vis(i) & vis(j)                         (GeSA, Eq. 3)
    Text queries stay strictly causal, so autoregressive decoding is unchanged.  Visual queries
    additionally keep causal access to the (template) tokens preceding the image block.
    """
    assert attn_mode in ATTN_MODES, attn_mode
    B, L = vis_mod.shape
    device = vis_mod.device
    allowed = torch.ones(L, L, dtype=torch.bool, device=device).tril_().unsqueeze(0).repeat(B, 1, 1)
    if attn_mode != 'causal':
        vis = vis_mod > 0
        bidir = vis[:, :, None] & vis[:, None, :]
        if attn_mode == 'intra':
            bidir &= vis_mod[:, :, None] == vis_mod[:, None, :]
        allowed |= bidir
    if attention_mask is not None:
        allowed &= attention_mask.bool()[:, None, :]
        # padded queries: keep the diagonal so no row is fully masked (avoids NaNs in SDPA)
        eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
        allowed |= eye & ~attention_mask.bool()[:, :, None]
    return allowed


def build_attention_mask_4d(vis_mod: torch.Tensor,
                            attention_mask: Optional[torch.Tensor],
                            attn_mode: str,
                            dtype: torch.dtype) -> torch.Tensor:
    """Additive (B, 1, L, L) mask (0 = keep, finfo.min = blocked) as expected by HF Qwen2."""
    allowed = build_allowed(vis_mod, attention_mask, attn_mode)
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(1)


def prepare_gesa_inputs(input_ids: torch.Tensor,
                        img_context_token_id: int,
                        layouts=None,
                        attention_mask: Optional[torch.Tensor] = None,
                        attn_mode: str = 'full',
                        pos_mode: str = 'shared',
                        dtype: torch.dtype = torch.bfloat16):
    """Returns (position_ids, 4D additive mask, vis_mod, vis_sp)."""
    vis_mod, vis_sp = build_vis_meta(input_ids, img_context_token_id, layouts)
    position_ids = build_position_ids(vis_sp, pos_mode)
    mask4d = build_attention_mask_4d(vis_mod, attention_mask, attn_mode, dtype)
    return position_ids, mask4d, vis_mod, vis_sp


# ----------------------------------------------------------------------------------
# Visual token fusion helpers (Table III left: element-wise average, external fusion)
# ----------------------------------------------------------------------------------
class ExternalFusion(nn.Module):
    """'External Fusion' baseline of Table III (inspired by LLaVA-Mini's pre-fusion): extra
    Transformer blocks with the LLM (Qwen2) architecture between the visual encoder and the
    LLM that fuse the concatenated [optical; SAR] tokens with bidirectional attention.

    If `init_layers` is given (training), blocks are copied from those LLM layers and their
    residual output projections (o_proj, down_proj) are zero-initialised, so the module starts
    as the identity (== plain concatenation) and learns the fusion.  5 Qwen2.5-3B blocks add
    ~0.39B parameters (4.2B -> ~4.6B, Table IX)."""

    def __init__(self, llm_config, n_layers, init_layers=None):
        super().__init__()
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        cfg = copy.deepcopy(llm_config)
        cfg._attn_implementation = 'sdpa'
        self.layers = nn.ModuleList([Qwen2DecoderLayer(cfg, i) for i in range(n_layers)])
        if init_layers is not None:
            for i, layer in enumerate(self.layers):
                layer.load_state_dict(init_layers[i].state_dict())
                nn.init.zeros_(layer.self_attn.o_proj.weight)
                nn.init.zeros_(layer.mlp.down_proj.weight)
                layer.to(init_layers[i].self_attn.q_proj.weight.dtype)

    def forward(self, x):  # x: (1, n, C)
        n = x.shape[1]
        pos = torch.arange(n, device=x.device).unsqueeze(0)
        mask = torch.zeros(1, 1, n, n, dtype=x.dtype, device=x.device)  # fully bidirectional
        for layer in self.layers:
            x = layer(x, attention_mask=mask, position_ids=pos)[0]
        return x


def average_colocated_tokens(vit_embeds: torch.Tensor, layout: Layout, tokens_per_tile: int) -> torch.Tensor:
    """vit_embeds: (n_tiles, tokens_per_tile, C) of one sample ordered as in `layout`.
    Returns (tokens_per_frame, C): mean over every frame of every modality of co-located tokens."""
    C = vit_embeds.shape[-1]
    flat = vit_embeds.reshape(-1, C)
    frames, cur = [], 0
    for _, t, f in layout:
        for _ in range(int(f)):
            frames.append(flat[cur:cur + int(t)])
            cur += int(t)
    return torch.stack(frames, 0).mean(0)


# ----------------------------------------------------------------------------------
# Sensor pre-processing (Sec. IV-A, Appendix F / Table VI)
# ----------------------------------------------------------------------------------
def _to_uint8_single(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return percentile_stretch_to_uint8(arr)


def percentile_stretch_to_uint8(a: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    a = a.astype(np.float32)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a, dtype=np.uint8)
    p_lo, p_hi = np.percentile(v, lo), np.percentile(v, hi)
    if p_hi - p_lo < 1e-6:
        return np.zeros_like(a, dtype=np.uint8)
    a = np.clip((a - p_lo) / (p_hi - p_lo), 0, 1)
    return (a * 255.0).astype(np.uint8)


def sar_to_pseudo_rgb(img, pad_mode: str = 'zero') -> Image.Image:
    """Single-/dual-channel SAR -> 3-channel pseudo-RGB frame.

    pad_mode='zero'      : missing channels are zeros (paper default, Table VI)
    pad_mode='replicate' : missing channels copy the existing ones (channel replication)
    3-channel inputs whose channels are identical are treated as single-channel.
    """
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        if img.mode in ('I;16', 'I', 'F'):
            arr = np.array(img)
        else:
            arr = np.array(img.convert('RGB') if img.mode not in ('L', 'LA', 'RGB', 'RGBA') else img)
    else:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:      # drop alpha
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 2:      # LA or dual-pol (VV, VH)
        chans = [arr[..., 0], arr[..., 1]]
        if np.array_equal(chans[0], chans[1]):
            chans = chans[:1]
    elif arr.ndim == 3:
        if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
            chans = [arr[..., 0]]
        else:
            return Image.fromarray(_to_uint8_single(arr) if arr.dtype != np.uint8 else arr)
    else:
        chans = [arr]
    chans = [_to_uint8_single(c) for c in chans]
    while len(chans) < 3:
        chans.append(np.zeros_like(chans[0]) if pad_mode == 'zero' else chans[len(chans) - 1])
    return Image.fromarray(np.stack(chans[:3], -1).astype(np.uint8), mode='RGB')


S2_RGB_BANDS = ('B04', 'B03', 'B02')
# remaining Sentinel-2 L2A bands in wavelength order, grouped as consecutive triplets
S2_REST_BANDS = ('B01', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
_BAND_RE = re.compile(r'_(B\d{2}|B8A)\.tif{1,2}$', re.IGNORECASE)


def _read_band(fp: str) -> np.ndarray:
    arr = np.array(Image.open(fp))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def ms_bands_to_pseudo_frames(band_files: Sequence[str],
                              group_size: int = 3,
                              pad_mode: str = 'zero') -> List[Image.Image]:
    """Sentinel-2 band files -> K pseudo-RGB frames (paper: three-band grouping).

    frame 0 = true colour (B04, B03, B02); the remaining bands are taken in wavelength order and
    every `group_size` consecutive bands form one frame (B01,B05,B06 | B07,B08,B8A | B09,B11,B12).
    All bands are resampled to the 10 m grid and percentile-stretched to uint8.
    """
    band_to_file = {}
    for fp in band_files:
        m = _BAND_RE.search(os.path.basename(fp))
        if m:
            band_to_file[m.group(1).upper()] = fp
    for b in S2_RGB_BANDS:
        if b not in band_to_file:
            raise FileNotFoundError(f'missing band {b} in {band_files[:3]}...')
    rgb = [_read_band(band_to_file[b]) for b in S2_RGB_BANDS]
    H = max(a.shape[0] for a in rgb)
    W = max(a.shape[1] for a in rgb)

    def _rs(a):
        if a.shape == (H, W):
            return a
        return np.array(Image.fromarray(a.astype(np.float32)).resize((W, H), resample=Image.BILINEAR))

    frames = [Image.fromarray(np.stack([percentile_stretch_to_uint8(_rs(a)) for a in rgb], -1), mode='RGB')]
    rest = [_rs(_read_band(band_to_file[b])) for b in S2_REST_BANDS if b in band_to_file]
    if group_size == 1:
        groups = [[a] for a in rest]
    else:
        groups = [rest[i:i + group_size] for i in range(0, len(rest), group_size)]
    for g in groups:
        chans = [percentile_stretch_to_uint8(a) for a in g]
        while len(chans) < 3:
            chans.append(np.zeros_like(chans[0]) if (pad_mode == 'zero' or group_size == 1) else chans[-1])
        frames.append(Image.fromarray(np.stack(chans[:3], -1), mode='RGB'))
    return frames


def ms_folder_to_pseudo_frames(folder: str, prefix: Optional[str] = None, **kw) -> List[Image.Image]:
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(('.tif', '.tiff')) and (prefix is None or f.startswith(prefix + '_'))]
    return ms_bands_to_pseudo_frames(files, **kw)


# ----------------------------------------------------------------------------------
# Dynamic tiling shared by all frames of a co-registered sample
# ----------------------------------------------------------------------------------
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def tile_grid(width: int, height: int, min_num=1, max_num=12, image_size=448):
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                            for i in range(1, n + 1) for j in range(1, n + 1)
                            if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    return find_closest_aspect_ratio(width / height, target_ratios, width, height, image_size)


def tile_with_grid(image: Image.Image, grid, image_size=448, use_thumbnail=True) -> List[Image.Image]:
    """InternVL dynamic_preprocess with an externally fixed grid (cols, rows)."""
    cols, rows = grid
    resized = image.resize((image_size * cols, image_size * rows))
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tiles.append(resized.crop((c * image_size, r * image_size, (c + 1) * image_size, (r + 1) * image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def tile_coregistered(frames_by_modality, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """frames_by_modality: list of (modality, [PIL frames]) in token order.
    All frames are tiled with the grid of the first frame (co-registration => same grid).
    Returns (list of PIL tiles in token order, layout, n_tiles_per_frame)."""
    ref = frames_by_modality[0][1][0]
    grid = tile_grid(ref.size[0], ref.size[1], min_num, max_num, image_size)
    tiles, layout = [], []
    n_tiles = None
    for mod, frames in frames_by_modality:
        for fr in frames:
            if fr.size != ref.size:
                fr = fr.resize(ref.size, resample=Image.BILINEAR)
            t = tile_with_grid(fr, grid, image_size, use_thumbnail)
            n_tiles = len(t)
            tiles.extend(t)
        layout.append([int(mod), int(n_tiles), int(len(frames))])   # tokens filled in by caller
    return tiles, layout, n_tiles, grid


def tiles_layout_to_token_layout(layout, tokens_per_tile=256):
    return [[int(m), int(t) * tokens_per_tile, int(f)] for m, t, f in layout]


# ----------------------------------------------------------------------------------
# Greedy decoding with GeSA prefill (custom 4D mask + shared position ids)
# ----------------------------------------------------------------------------------
@torch.no_grad()
def gesa_greedy_generate(language_model,
                         inputs_embeds: torch.Tensor,
                         attention_mask_4d: torch.Tensor,
                         position_ids: torch.Tensor,
                         max_new_tokens: int = 512,
                         eos_token_ids: Sequence[int] = (),
                         stop_words: Sequence[str] = (),
                         tokenizer=None,
                         output_attentions: bool = False):
    """Batch-size-1 greedy decoding.

    The prompt is prefilled once with the GeSA mask/positions; every new token is a text
    token, so it attends causally to the whole cache and takes position max(prompt)+1, +2, ...
    Returns dict(sequences=LongTensor[n_gen], hidden_states=Tensor[n_gen-1, C] (last layer
    hidden state of every generated token except the last one, i.e. the state produced when
    the token is fed back -- the same convention as Sa2VA/EarthMind for [SEG]),
    first_logits=float logits of the first decoding step, shape (vocab,)).
    """
    assert inputs_embeds.shape[0] == 1, 'batch size 1 only'
    out = language_model(inputs_embeds=inputs_embeds,
                         attention_mask=attention_mask_4d,
                         position_ids=position_ids,
                         use_cache=True,
                         output_hidden_states=True,
                         output_attentions=output_attentions,
                         return_dict=True)
    past = out.past_key_values
    next_pos = position_ids[:, -1:].clone() + 1
    logits = out.logits[:, -1]
    first_logits = logits[0].float().clone()
    del out
    embed = language_model.get_input_embeddings()
    gen, hiddens = [], []
    eos = set(int(e) for e in eos_token_ids if e is not None)
    for step in range(max_new_tokens):
        tok = logits.argmax(-1)
        gen.append(int(tok.item()))
        if gen[-1] in eos:
            break
        if stop_words and tokenizer is not None:
            text = tokenizer.decode(gen[-16:], skip_special_tokens=False)
            if any(text.endswith(sw) for sw in stop_words):
                break
        if step == max_new_tokens - 1:
            break
        out = language_model(inputs_embeds=embed(tok[:, None]),
                             past_key_values=past,
                             position_ids=next_pos,
                             use_cache=True,
                             output_hidden_states=True,
                             return_dict=True)
        hiddens.append(out.hidden_states[-1][:, -1])
        past = out.past_key_values
        logits = out.logits[:, -1]
        next_pos = next_pos + 1
    seq = torch.tensor(gen, dtype=torch.long, device=inputs_embeds.device)
    hs = torch.cat(hiddens, 0) if hiddens else inputs_embeds.new_zeros((0, inputs_embeds.shape[-1]))
    return dict(sequences=seq, hidden_states=hs, first_logits=first_logits)


# ----------------------------------------------------------------------------------
# Eq. (1): Modality Attention Gap
# ----------------------------------------------------------------------------------
class ModalityAttentionRecorder:
    """Forward hooks on the LLM self-attention modules (eager attention, output_attentions=True)
    that reduce each layer's attention map to the text->optical / text->SAR attention mass and
    drop the full map immediately (memory-safe for long multi-sensor sequences).

    A^m_l = sum_{i in T} sum_{j in V_m} a^l_{ij} / sum_{i in T} sum_{k in V} a^l_{ik},
    with a^l averaged over heads.  MAG = 1/L sum_l |A^opt_l - A^sar_l|.
    """

    def __init__(self, attn_modules, text_query_mask: torch.Tensor, vis_mod: torch.Tensor):
        self.text_q = text_query_mask[0].bool()   # (L,)
        self.opt_k = (vis_mod[0] == MOD_OPT)
        self.sar_k = (vis_mod[0] == MOD_SAR)
        self.per_layer = []
        self.handles = [m.register_forward_hook(self._hook) for m in attn_modules]

    def _hook(self, module, inputs, output):
        attn = output[1]
        if attn is None:
            return output
        a = attn[0].float().mean(0)              # (Lq, Lk) head-averaged
        Lq = a.shape[0]
        tq = self.text_q[-Lq:] if Lq != self.text_q.shape[0] else self.text_q
        a_t = a[tq]                              # text queries
        s_opt = a_t[:, self.opt_k[:a.shape[1]]].sum().item()
        s_sar = a_t[:, self.sar_k[:a.shape[1]]].sum().item()
        tot = s_opt + s_sar
        self.per_layer.append((s_opt / tot if tot > 0 else float('nan'),
                               s_sar / tot if tot > 0 else float('nan')))
        return (output[0], None) + tuple(output[2:])

    def remove(self):
        for h in self.handles:
            h.remove()

    def result(self):
        arr = np.array(self.per_layer, dtype=np.float64)          # (L, 2)
        gap = np.abs(arr[:, 0] - arr[:, 1])
        return dict(A_opt=arr[:, 0].tolist(), A_sar=arr[:, 1].tolist(),
                    gap_per_layer=gap.tolist(), MAG=float(np.nanmean(gap)) if len(gap) else float('nan'))
