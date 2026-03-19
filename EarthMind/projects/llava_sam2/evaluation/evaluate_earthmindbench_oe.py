import json
import openai
from tqdm import tqdm

# Set your OpenAI API key
openai.api_key = ""  # Replace with your actual key

# Input and output paths
input_json_path = "pair_multioe/caption_all_unmatched_result.json"
output_json_path = "scored_output.json"

# English prompt builder
def build_prompt(pred, gt):
    return f"""
You are a remote sensing expert evaluating how well a predicted image caption matches the human-written ground truth caption. 

Please rate the similarity between the **predicted caption** and the **ground truth** based on the following criteria:

1 - Completely unrelated (content is very different)  
2 - Slightly related, but most descriptions do not match  
3 - Somewhat similar, with a few common details but also clear differences  
4 - Mostly matching, only a few minor differences  
5 - Highly consistent, both descriptions describe the same content in detail

Now evaluate the following:

Predicted caption:
{pred}

Ground truth caption:
{gt}

Your score (only output a number from 1 to 5):
"""

# GPT evaluation function
def gpt_score(pred, gt):
    prompt = build_prompt(pred, gt)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        score_text = response['choices'][0]['message']['content'].strip()
        score = int(score_text[0])  # Just take the first digit
        return max(1, min(score, 5))
    except Exception as e:
        print(f"Error scoring item: {e}")
        return -1

# Load input data
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Score each item
scored_data = []
for item in tqdm(data):
    pred = item["pred"]
    gt = item["ground_truth"]
    score = gpt_score(pred, gt)
    item["gpt_score"] = score
    scored_data.append(item)

# Save results
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(scored_data, f, indent=2, ensure_ascii=False)

print(f"✅ Scoring complete. Results saved to: {output_json_path}")
