#!/usr/bin/env python3
"""
Inference script for Building Damage Counting tasks
Processes benchmark.json and outputs predictions
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import re
from tqdm import tqdm

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path: str, device: str = 'cuda:0'):
    """Load the Sa2VA model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    return model, tokenizer

def create_prompt_with_options(prompt_text: str, options_str: str, custom_prompt: str) -> str:
    """Create the full prompt with custom instructions and options"""
    # Combine custom prompt with the question and options
    full_prompt = custom_prompt + "<image>\n" + prompt_text + "\n" + options_str + "\n"
    return full_prompt

def extract_answer_from_output(output: str) -> str:
    """Extract the answer letter (A, B, C, D, E) from model output"""
    # Look for answer within <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, output, re.DOTALL)
    
    if match:
        answer_text = match.group(1).strip()
        # Extract just the letter if it's in format "A" or "A. X" etc
        letter_match = re.search(r'[A-E]', answer_text.upper())
        if letter_match:
            return letter_match.group(0)
    
    # Fallback: look for any letter A-E in the output
    letter_match = re.search(r'\b[A-E]\b', output.upper())
    if letter_match:
        return letter_match.group(0)
    
    return "N/A"  # Return N/A if no answer found

def run_inference_on_sample(
    model, 
    tokenizer, 
    sample: Dict[str, Any], 
    custom_prompt: str,
    base_image_dir: str = ""
) -> Dict[str, Any]:
    """Run inference on a single sample"""
    
    # Fix Windows-style paths (convert \\ to /)
    pre_image_path_fixed = sample['pre_image_path'].replace('\\', '/')
    post_image_path_fixed = sample['post_image_path'].replace('\\', '/')
    
    # Construct full image paths
    pre_img_path = os.path.join(base_image_dir, pre_image_path_fixed)
    post_img_path = os.path.join(base_image_dir, post_image_path_fixed)
    
    # Check if images exist
    if not os.path.exists(pre_img_path):
        print(f"Warning: Pre-disaster image not found: {pre_img_path}")
        return None
    if not os.path.exists(post_img_path):
        print(f"Warning: Post-disaster image not found: {post_img_path}")
        return None
    
    # Load images with error handling for corrupted files
    try:
        pre_img = Image.open(pre_img_path).convert('RGB')
        # Verify image can be loaded properly by accessing its size
        _ = pre_img.size
    except Exception as e:
        print(f"Error: Failed to load pre-disaster image {pre_img_path}: {str(e)}")
        print(f"  Image may be corrupted. Skipping this sample.")
        return None
    
    try:
        post_img = Image.open(post_img_path).convert('RGB')
        # Verify image can be loaded properly by accessing its size
        _ = post_img.size
    except Exception as e:
        print(f"Error: Failed to load post-disaster image {post_img_path}: {str(e)}")
        print(f"  Image may be corrupted. Skipping this sample.")
        return None
    
    # Create prompt with options
    full_prompt = create_prompt_with_options(
        sample['prompts'], 
        sample['options_str'], 
        custom_prompt
    )
    
    # Run inference
    try:
        result = model.predict_forward_with_grounding_multi(
            image_list=[pre_img, post_img],
            text=full_prompt,
            tokenizer=tokenizer,
            max_tokens_per_seg=8,
        )
        
        output = result['prediction']
        print("1111",output)
        # Extract answer
        pred_option = extract_answer_from_output(output)
        print("2222",pred_option,sample['ground_truth_option'])
        
        return {
            'pre_image_path': pre_image_path_fixed,  # Use fixed path
            'post_image_path': post_image_path_fixed,  # Use fixed path
            'ground_truth_option': sample['ground_truth_option'],
            'pred_option': pred_option,
            'model_output': output,  # Store full output for debugging
            'task': sample['task'],
            'prompts': sample['prompts'],
            'options_str': sample['options_str']
        }
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return {
            'pre_image_path': pre_image_path_fixed,  # Use fixed path
            'post_image_path': post_image_path_fixed,  # Use fixed path
            'ground_truth_option': sample['ground_truth_option'],
            'pred_option': 'ERROR',
            'model_output': str(e),
            'task': sample['task'],
            'prompts': sample['prompts'],
            'options_str': sample['options_str']
        }


def main():
    """Main inference pipeline"""
    
    # Configuration
    MODEL_PATH = "terrascope_new"
    BENCHMARK_PATH = "data/benchmark_release.json"
    OUTPUT_PATH = "eval_results/disaster_inference_results.json"
    BASE_IMAGE_DIR = "data/disaster"  # Adjust if images are in a different base directory
    
    # Custom prompt for chain-of-thought reasoning
    CUSTOM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in their mind and then provides the user a concise final answer in a "
        "short word or phrase. The reasoning process and answer are enclosed within "
        "<think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>\n\n"
    )
    
    # Load benchmark data
    print(f"Loading benchmark data from {BENCHMARK_PATH}...")
    with open(BENCHMARK_PATH, 'r') as f:
        benchmark_data = json.load(f)
    
    # Filter for Building Damage Counting tasks
    building_damage_samples = [
        sample for sample in benchmark_data 
        if sample.get('task') == 'Building Damage Counting' and sample.get('post_image_type') == 'Optical'
    ]
    # building_damage_samples = [
    #     sample for sample in benchmark_data 
    #     if sample.get('task') == 'Building Damage Counting' and sample.get('post_image_type') == 'SAR'
    # ]

    # Building Damage Counting
    
    print(f"Found {len(building_damage_samples)} Building Damage Counting samples")
    
    if len(building_damage_samples) == 0:
        print("No Building Damage Counting samples found!")
        return
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Run inference on all samples
    results = []
    correct_predictions = 0
    skipped_samples = 0
    
    print("\nStarting inference...")
    for i, sample in enumerate(tqdm(building_damage_samples, desc="Processing samples")):
        print(f"\n[{i+1}/{len(building_damage_samples)}] Processing sample...")
        print(f"  Pre-image: {sample['pre_image_path']}")
        print(f"  Post-image: {sample['post_image_path']}")
        
        result = run_inference_on_sample(
            model, 
            tokenizer, 
            sample, 
            CUSTOM_PROMPT,
            BASE_IMAGE_DIR
        )
        
        if result:
            results.append(result)
            
            # Check if prediction is correct
            if result['pred_option'] == result['ground_truth_option']:
                correct_predictions += 1
                print(f"  ✓ Correct! Predicted: {result['pred_option']}, Ground Truth: {result['ground_truth_option']}")
            else:
                print(f"  ✗ Wrong! Predicted: {result['pred_option']}, Ground Truth: {result['ground_truth_option']}")
        else:
            skipped_samples += 1
            print(f"  ⚠ Skipped due to missing or corrupted images")
    
    # Calculate accuracy
    if len(results) > 0:
        accuracy = correct_predictions / len(results) * 100
        print(f"\n{'='*80}")
        print(f"Inference Complete!")
        print(f"Total samples in dataset: {len(building_damage_samples)}")
        print(f"Samples processed successfully: {len(results)}")
        print(f"Samples skipped (missing/corrupted): {skipped_samples}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*80}")
    else:
        print("\n⚠️ No samples were processed successfully!")
        accuracy = 0
    
    # Save results
    output_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_path': MODEL_PATH,
        'total_samples_in_dataset': len(building_damage_samples),
        'samples_processed': len(results),
        'samples_skipped': skipped_samples,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy if len(results) > 0 else 0,
        'predictions': results
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {OUTPUT_PATH}")
    
    # Also save a simplified version with just the essential fields
    simplified_results = []
    for r in results:
        simplified_results.append({
            'pre_image_path': r['pre_image_path'],
            'post_image_path': r['post_image_path'],
            'ground_truth_option': r['ground_truth_option'],
            'pred_option': r['pred_option']
        })
    
    simplified_output_path = os.path.join(
        os.path.dirname(OUTPUT_PATH), 
        "inference_results_simplified.json"
    )
    with open(simplified_output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Simplified results saved to {simplified_output_path}")

if __name__ == "__main__":
    main()