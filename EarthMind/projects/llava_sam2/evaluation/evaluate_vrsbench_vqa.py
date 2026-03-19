import openai
import json
import openai
import os
from tqdm import tqdm
import numpy as np

# Set your OpenAI API Key
openai.api_key = ""
# client = OpenAI()

def check_match_with_gpt(question, ground_truth, predicted):
    # Construct the prompt for GPT-4
    prompt = f"Question: {question}\nGround Truth Answer: {ground_truth}\nPredicted Answer: {predicted}\nDoes the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.\n"

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-1106",
        model="gpt-4o",
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": prompt,
                    },
                ]
            }
        ],
        max_tokens=100,
    )

    # answer = response.choices[0].text.strip()
    answer =  response.choices[0].message.content
    
    return answer

# qa_list = [json.loads(line) for line in open('/scqian/groundingLMM/rs_vqa_results/trainrsvqa_formal_vrsbench.json','r').readlines()]
with open ('vrs_vqaresult/vrsbench_vqa_new.json','r') as f:
    qa_list=json.load(f)

# Iterate over the list and check matches
results = []
f = open('vrs_vqaresult/vrsbench_eval_gpt.json', 'w') 
for ii, qa in enumerate(tqdm(qa_list)):
    question = qa['question']
    ground_truth = qa['ground_truth'].lower()
    predicted = qa['answer'].lower()
    if ground_truth in predicted:
        match_result = '1'
    elif ground_truth in ['yes', 'no'] + list(map(str, range(100))):
        match_result = '1' if ground_truth == predicted else '0'
    elif 'correct' not in qa or qa['correct'] not in ['1', '0']:
        match_result = check_match_with_gpt(question, ground_truth, predicted)
    else:
        match_result = qa['correct']
        
    result = {
        'question_id': qa['question_id'],
        'image_id': qa['image_id'],
        "type": qa['type'],
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": match_result,
    }
    results.append(result)

    f.write(json.dumps(result)+'\n')
    f.flush()

f.close()
for result in results:
    if ii>5:
        break
    print(result)


f = open('vrs_vqaresult/vrsbench_eval_gpt.json', 'r') 
results = [json.loads(line) for line in f.readlines()]
f.close()
correct = sum([int(result['correct']) for result in results if result['correct'] in ['1', '0']])
print(f"Correct: {correct}/{len(results)}:", correct/len(results))

