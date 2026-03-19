#!/usr/bin/env python3
"""
TerraScope demo: run inference on selected examples from TerraScope-Bench
and DisasterM3, then visualize segmentation masks with reasoning output.

Usage:
    python demo.py --model-path /path/to/model --device cuda:0
    python demo.py --output-dir demo_output
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr from model internals (iteration logs, tqdm, etc.)."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


# ═══════════════════════════════════════════════════════════════
# Model utilities
# ═══════════════════════════════════════════════════════════════

CUSTOM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The Assistant first thinks about the reasoning "
    "process in their mind and then provides the user a concise final answer in a "
    "short word or phrase. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>\n\n"
)


def extract_answer(response: str):
    """Extract final answer letter (A-E) from model output.
    Matches the logic in infer_disaster.py and infer_terrascope_bench.py."""
    if not response:
        return None

    # 1) Look inside <answer>...</answer> tags
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        answer_text = m.group(1).strip()
        letter = re.search(r"[A-E]", answer_text.upper())
        if letter:
            return letter.group(0)

    # 2) Fallback: search entire output for A-E
    letter = re.search(r"\b[A-E]\b", response.upper())
    if letter:
        return letter.group(0)

    return "N/A"


def extract_think(response: str):
    """Extract <think> reasoning text, cleaned of [SEG]/REGION tokens."""
    if not response:
        return ""
    m = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL | re.IGNORECASE)
    text = m.group(1).strip() if m else ""
    text = re.sub(r"\s*\[SEG\]\s*", " [mask] ", text)
    text = re.sub(r"<REGION>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_prompt(question, candidates):
    """Build inference prompt text."""
    formatted = f"{question}\n"
    for key in sorted(candidates.keys()):
        formatted += f"{key}. {candidates[key]}\n"
    return CUSTOM_PROMPT + "<image>\n" + formatted


# ═══════════════════════════════════════════════════════════════
# Mask to numpy
# ═══════════════════════════════════════════════════════════════

def mask_to_numpy(m, frame_idx=None):
    """Convert model mask output to (H, W) bool numpy array.
    For multi-image tasks, frame_idx selects which frame's mask to use."""
    import torch
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    # Multi-frame mask: (num_frames, H, W) — pick specific frame
    if m.ndim == 3 and frame_idx is not None:
        m = m[frame_idx]
    elif m.ndim == 3 and m.shape[0] <= 4:
        # Likely (num_frames, H, W), take last frame (post-disaster)
        m = m[-1]
    m = m.squeeze()
    if m.dtype != bool:
        m = m > 0.5 if m.max() <= 1.0 else m > 127
    return m.astype(bool)


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

MASK_COLORS = [
    (0.12, 0.56, 1.0, 0.45),   # dodger blue
    (1.0, 0.27, 0.0, 0.45),    # orange-red
    (0.0, 0.8, 0.4, 0.45),     # green
    (0.93, 0.86, 0.0, 0.45),   # gold
    (0.58, 0.0, 0.83, 0.45),   # purple
]


def vis_single_image(img, masks, question, answer_text, reasoning, gt_answer,
                     predicted, is_correct, title, out_path):
    """Visualize a single-image example (TerraScope-Bench)."""
    n_masks = len(masks)
    ncols = 1 + max(n_masks, 1)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    # Panel 1: original image
    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panels 2+: mask overlays
    for i, mask_np in enumerate(masks):
        axes[i + 1].imshow(img)
        color = MASK_COLORS[i % len(MASK_COLORS)]
        overlay = np.zeros((*mask_np.shape, 4))
        overlay[mask_np] = color
        axes[i + 1].imshow(overlay)
        coverage = mask_np.sum() / mask_np.size * 100
        axes[i + 1].set_title(f"Mask {i+1} ({coverage:.1f}%)", fontsize=12)
        axes[i + 1].axis("off")

    # If no masks, show a "no mask" panel
    if n_masks == 0:
        axes[1].imshow(img)
        axes[1].set_title("(No segmentation mask)", fontsize=12)
        axes[1].axis("off")

    # Suptitle with question and answer
    correct_mark = "Correct" if is_correct else "Wrong"
    color = "green" if is_correct else "red"
    wrapped_q = "\n".join(textwrap.wrap(question, width=90))
    fig.suptitle(f"{title}\nQ: {wrapped_q}\nPredicted: {predicted}  |  GT: {gt_answer}  [{correct_mark}]",
                 fontsize=11, y=0.02, va="bottom", color=color if not is_correct else "black")

    # Add reasoning as text below
    if reasoning:
        short = reasoning[:200] + ("..." if len(reasoning) > 200 else "")
        fig.text(0.5, 0.96, f"Reasoning: {short}", ha="center", fontsize=9,
                 style="italic", color="gray", wrap=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def vis_dual_image(pre_img, post_img, masks, question, answer_text, reasoning,
                   gt_answer, predicted, is_correct, title, out_path):
    """Visualize a dual-image example (DisasterM3 pre/post)."""
    n_masks = len(masks)
    ncols = 2 + max(n_masks, 1)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))

    # Panel 1: pre-disaster
    axes[0].imshow(pre_img)
    axes[0].set_title("Pre-disaster", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: post-disaster
    axes[1].imshow(post_img)
    axes[1].set_title("Post-disaster", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Panels 3+: mask overlays on post image
    for i, mask_np in enumerate(masks):
        axes[i + 2].imshow(post_img)
        color = MASK_COLORS[i % len(MASK_COLORS)]
        overlay = np.zeros((*mask_np.shape, 4))
        overlay[mask_np] = color
        axes[i + 2].imshow(overlay)
        coverage = mask_np.sum() / mask_np.size * 100
        axes[i + 2].set_title(f"Mask {i+1} ({coverage:.1f}%)", fontsize=12)
        axes[i + 2].axis("off")

    if n_masks == 0:
        axes[2].imshow(post_img)
        axes[2].set_title("(No mask)", fontsize=12)
        axes[2].axis("off")

    correct_mark = "Correct" if is_correct else "Wrong"
    wrapped_q = "\n".join(textwrap.wrap(question, width=100))
    fig.suptitle(f"{title}\nQ: {wrapped_q}\nPredicted: {predicted}  |  GT: {gt_answer}  [{correct_mark}]",
                 fontsize=11, y=0.02, va="bottom")

    if reasoning:
        short = reasoning[:250] + ("..." if len(reasoning) > 250 else "")
        fig.text(0.5, 0.96, f"Reasoning: {short}", ha="center", fontsize=9,
                 style="italic", color="gray", wrap=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════
# TerraScope-Bench examples
# ═══════════════════════════════════════════════════════════════

TERRASCOPE_EXAMPLES = [
    # (task_type, image_name, question, candidates, gt_answer)
    ("coverage_percentage", "3886_3141_patch00.png",
     "What proportion of the image is occupied by developed areas?",
     {"A": "58%", "B": "80%", "C": "68%", "D": "65%"}, "A"),
]


DISASTER_EXAMPLES = [
    # (pre_path, post_path, question, options_str, gt_option)
    # Volcano: 15 buildings, 9 intact
    ("guatemala_volcano_00000000_pre_disaster.png",
     "guatemala_volcano_00000000_post_disaster.png",
     "What is the total number of undamaged buildings following the disaster?",
     "A. 9, B. 1, C. 7, D. 13, E. 17.", "A"),
]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TerraScope demo with segmentation visualization")
    parser.add_argument("--model-path", type=str,
                        default="terrascope_new")
    parser.add_argument("--image-dir", type=str,
                        default="demo_imgs")
    parser.add_argument("--disaster-dir", type=str,
                        default="demo_imgs")
    parser.add_argument("--output-dir", type=str, default="demo_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-terrascope", action="store_true", help="Skip TerraScope-Bench examples")
    parser.add_argument("--skip-disaster", action="store_true", help="Skip DisasterM3 examples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device, trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("Model loaded.\n")

    # ─── TerraScope-Bench examples ───
    if not args.skip_terrascope:
        print("=" * 60)
        print("  TerraScope-Bench Examples")
        print("=" * 60)

        for idx, (task, img_name, question, candidates, gt) in enumerate(TERRASCOPE_EXAMPLES):
            print(f"\n{'─'*60}")
            print(f"[{idx+1}/{len(TERRASCOPE_EXAMPLES)}] {task} | {img_name}")
            print(f"  Question:  {question}")
            cands_str = "  ".join(f"{k}. {v}" for k, v in sorted(candidates.items()))
            print(f"  Options:   {cands_str}")

            img_path = os.path.join(args.image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"  [SKIP] Image not found: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            prompt = format_prompt(question, candidates)

            with suppress_output():
                result = model.predict_forward_with_grounding(
                    image=img, text=prompt, tokenizer=tokenizer, max_tokens_per_seg=8,
                )
            response = result.get("prediction", "")
            predicted = extract_answer(response)
            reasoning = extract_think(response)
            is_correct = (predicted == gt)

            masks = []
            raw_masks = result.get("prediction_masks", [])
            if raw_masks:
                for m in raw_masks:
                    masks.append(mask_to_numpy(m))

            print(f"  Reasoning: {reasoning}")
            print(f"  Predicted: {predicted} | GT: {gt} | {'Correct' if is_correct else 'Wrong'}")
            print(f"  Masks:     {len(masks)}")

            out_path = os.path.join(args.output_dir, f"terrascope_{idx+1:02d}_{task}.png")
            vis_single_image(
                np.array(img), masks, question, predicted, reasoning, gt,
                predicted, is_correct, f"TerraScope-Bench / {task}", out_path
            )

    # ─── DisasterM3 examples ───
    if not args.skip_disaster:
        print("\n" + "=" * 60)
        print("  DisasterM3 Examples")
        print("=" * 60)

        for idx, (pre_path, post_path, question, options_str, gt) in enumerate(DISASTER_EXAMPLES):
            print(f"\n{'─'*60}")
            print(f"[{idx+1}/{len(DISASTER_EXAMPLES)}] {os.path.basename(pre_path)}")
            print(f"  Question:  {question}")
            print(f"  Options:   {options_str}")

            pre_full = os.path.join(args.disaster_dir, pre_path)
            post_full = os.path.join(args.disaster_dir, post_path)

            if not os.path.exists(pre_full) or not os.path.exists(post_full):
                print(f"  [SKIP] Images not found")
                continue

            pre_img = Image.open(pre_full).convert("RGB")
            post_img = Image.open(post_full).convert("RGB")

            prompt = CUSTOM_PROMPT + "<image>\n" + question + "\n" + options_str + "\n"

            with suppress_output():
                result = model.predict_forward_with_grounding_multi(
                    image_list=[pre_img, post_img], text=prompt,
                    tokenizer=tokenizer, max_tokens_per_seg=8,
                )
            response = result.get("prediction", "")
            predicted = extract_answer(response)
            reasoning = extract_think(response)
            is_correct = (predicted == gt)

            masks = []
            raw_masks = result.get("prediction_masks", [])
            if raw_masks:
                for m in raw_masks:
                    # frame_idx=1 selects the post-disaster frame
                    masks.append(mask_to_numpy(m, frame_idx=1))

            print(f"  Reasoning: {reasoning}")
            print(f"  Predicted: {predicted} | GT: {gt} | {'Correct' if is_correct else 'Wrong'}")
            print(f"  Masks:     {len(masks)}")

            out_path = os.path.join(args.output_dir, f"disaster_{idx+1:02d}_{os.path.basename(pre_path).replace('_pre_disaster.png','')}.png")
            vis_dual_image(
                np.array(pre_img), np.array(post_img), masks, question,
                predicted, reasoning, gt, predicted, is_correct,
                f"DisasterM3 / Building Damage Counting", out_path
            )

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
