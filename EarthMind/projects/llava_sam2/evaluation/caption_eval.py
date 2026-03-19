import os
import json
import numpy as np

data_path = 'vrsbench_cap_pre.json'
# data_path = '../../outputs/GPT4_eval/gpt4_cap.json'

gt_answers= []
pred_answers = []
with open(data_path, 'r') as file:
    data=json.load(file)
    for item in data:
        # item = json.loads(line.strip())
        img_id = item['image_id']
        # if '09194_0000' in data_path:
        #     print(item)

        gt_ans = item['ground_truth'].strip().replace('\n', ' ')
        pred_ans = item['answer'].strip().replace('\n', ' ')

        if img_id is None or img_id=='\n' or pred_ans is None or pred_ans=='\n':
            print('empty', img_id, pred_ans)
            continue

        gt_answers.append([img_id, gt_ans])
        pred_answers.append([img_id, pred_ans])

print('number of captions', len(gt_answers))

np.savetxt('pred_cap.txt',  pred_answers, fmt='%s', delimiter='\t')
np.savetxt('gt_cap.txt',  gt_answers, fmt='%s', delimiter='\t')