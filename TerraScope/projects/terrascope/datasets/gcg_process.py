"""
Grounded Visual CoT data processing functions
Data already contains complete conversations, use directly
"""

import numpy as np
from xtuner.utils import DEFAULT_IMAGE_TOKEN


# ============================================================
# ============================================================

def glamm_grounded_cot_map_fn(example):
    """
    Main processing function for Grounded Visual CoT data
    
    Input: Grounded CoT JSON format (with complete conversations)
    {
        "image": "sa_10010541.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\n..."},
            {"from": "gpt", "value": "<think>...</think>\nFinal answer: ..."}
        ],
        "labels": [...],
        "masks": [...],
        "seg_info": [...],
        ...
    }
    
    Output: conversation format required for training
    """
    messages = example['conversations']
    input_text = ''
    conversation = []
    
    while messages and messages[0]['from'] == 'gpt':
        messages = messages[1:]
    
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input_text += msg['value']
            
        elif msg['from'] == 'gpt':
            conversation.append({
                'input': input_text, 
                'output': msg['value']
            })
            input_text = ''
        else:
            raise NotImplementedError(f"Unknown message type: {msg['from']}")
    
    example.update({
        'conversation': conversation,
        'image': example.get('image', ''),
        'file_name': example.get('image', ''),
    })
    
    return example


# ============================================================
# ============================================================

GCG_QUESTIONS = [
    DEFAULT_IMAGE_TOKEN + 'Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    DEFAULT_IMAGE_TOKEN + 'Can you provide a brief description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    DEFAULT_IMAGE_TOKEN + 'Could you give a brief explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Could you give me an brief explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Could you provide me with a briefly analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]


def grandf_parse_annotations(example):
    """Parse GranD/GranF raw format"""
    image_path = example['file_name']
    annotations = {
        'labels': [], 
        'caption': [], 
        'masks': [],
        'tokens_positive': [], 
        'file_name': image_path,
        'image': image_path
    }
    annotations['caption'] = example['caption'].strip('"').strip()

    for word, grounding in example["groundings"].items():
        if grounding is None:
            continue
        annotations['labels'].append(word)
        annotations['tokens_positive'].append(grounding["token_positives"])
        annotations['masks'].append(grounding["rle_masks"])

    return annotations


def grandf_conversation(caption, tokens_positive):
    """Generate original GCG format dialogue (with <p></p>[SEG] markers)"""
    import random
    question = random.choice(GCG_QUESTIONS).strip()

    def tag_caption(caption, tokens):
        for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
            caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
        return caption

    detailed_answer = tag_caption(caption, tokens_positive)
    conversations = [
        {'from': 'human', 'value': question}, 
        {'from': 'gpt', 'value': detailed_answer}
    ]
    return conversations


def grandf_preprocess(example):
    """Preprocess GranD/GranF data"""
    data_labels = example['labels']
    masks = example['masks']
    caption = example['caption']
    tokens_positive = example['tokens_positive']

    def sort_by_start_index(items, order):
        return [items[i] for i in order]

    phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
    masks = sort_by_start_index(masks, phrase_order)
    data_labels = sort_by_start_index(data_labels, phrase_order)
    tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

    conversations = grandf_conversation(caption, tokens_positive)
    example['conversations'] = conversations
    example['labels'] = data_labels
    example['masks'] = masks
    example['tokens_positive'] = tokens_positive
    return example


def glamm_granf_map_fn(example):
    """
    GranD/GranF raw data processing (backward compatible)
    
    This function generates conversations since raw data doesn't have them
    """
    example = grandf_parse_annotations(example)
    example = grandf_preprocess(example)

    messages = example['conversations']
    input_text = ''
    conversation = []
    
    while messages and messages[0]['from'] == 'gpt':
        messages = messages[1:]
    
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input_text += msg['value']
            
        elif msg['from'] == 'gpt':
            conversation.append({'input': input_text, 'output': msg['value']})
            input_text = ''
        else:
            raise NotImplementedError
    
    example.update({'conversation': conversation})
    return example


# ============================================================
# ============================================================

def refcocog_parse_annotations(example):
    annotations = {
        'labels': [], 
        'caption': [], 
        'masks': [], 
        'tokens_positive': [],
        'file_name': example['img_file_name'], 
        'image': example['img_file_name']
    }

    orig_caption = example['caption'].strip('"').strip()
    annotations['caption'] = orig_caption.lower()

    for detail in example['refs']:
        phrase = detail['sentence']
        if phrase.lower() in annotations['caption']:
            annotations['labels'].append(phrase)
            index = annotations['caption'].find(phrase)
            end_index = index + len(phrase) if index != -1 else -1
            annotations['tokens_positive'].append([index, end_index])
            annotations['masks'].append(detail["segmentation"])

    tokens_positive = annotations['tokens_positive']
    sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
    annotations['tokens_positive'] = [tokens_positive[i] for i in sorted_indices]
    annotations['masks'] = [annotations['masks'][i] for i in sorted_indices]
    annotations['labels'] = [annotations['labels'][i] for i in sorted_indices]

    for i in range(len(tokens_positive)):
        for j in range(i + 1, len(tokens_positive)):
            if tokens_positive[i][1] >= tokens_positive[j][0]:
                tokens_positive[i][1] = tokens_positive[j][0] - 1
                annotations['labels'][i] = orig_caption[tokens_positive[i][0]:tokens_positive[i][1] + 1]
                break

    return annotations


def refcocog_conversation(caption, tokens_positive):
    import random
    question = random.choice(GCG_QUESTIONS).strip()

    def tag_caption(caption, tokens):
        for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
            caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
        return caption

    detailed_answer = tag_caption(caption, tokens_positive)
    conversations = [
        {'from': 'human', 'value': question}, 
        {'from': 'gpt', 'value': detailed_answer}
    ]
    return conversations


def refcocog_preprocess(example):
    data_labels = example['labels']
    masks = example['masks']
    caption = example['caption']
    tokens_positive = example['tokens_positive']

    def sort_by_start_index(items, order):
        return [items[i] for i in order]

    phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
    masks = sort_by_start_index(masks, phrase_order)
    data_labels = sort_by_start_index(data_labels, phrase_order)
    tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

    conversations = refcocog_conversation(caption, tokens_positive)
    example['conversations'] = conversations
    example['labels'] = data_labels
    example['masks'] = masks
    example['tokens_positive'] = tokens_positive

    return example


def glamm_refcocog_map_fn(example):
    example = refcocog_parse_annotations(example)
    example = refcocog_preprocess(example)

    messages = example['conversations']
    input_text = ''
    conversation = []
    
    while messages and messages[0]['from'] == 'gpt':
        messages = messages[1:]
    
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input_text += msg['value']
            
        elif msg['from'] == 'gpt':
            conversation.append({'input': input_text, 'output': msg['value']})
            input_text = ''
        else:
            raise NotImplementedError
    
    example.update({'conversation': conversation})
    return example


# ============================================================
# ============================================================

def flickr_parse_annotations(example):
    annotations = {
        'bboxes': [], 
        'labels': [], 
        'bboxes_ignore': [], 
        'caption': example['caption'], 
        'masks': [],
        'tokens_positive': [], 
        'image': example['file_name']
    }
    
    ann_info = example["ann_info"]
    for ann in ann_info:
        if ann.get('ignore', False):
            continue
        
        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, example['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, example['height']) - max(y1, 0))
        
        if inter_w * inter_h == 0 or ann['area'] <= 0 or w < 1 or h < 1:
            continue
        
        bbox = [x1, y1, x1 + w, y1 + h]
        annotations['bboxes'].append(bbox)
        
        tokens_positive = ann['tokens_positive']
        gt_label = [example['caption'][span[0]:span[1]] for span in tokens_positive]
        annotations['labels'].append(gt_label[0])
        annotations['tokens_positive'].append(tokens_positive[0])
        
        rle = ann['sam_mask']
        annotations['masks'].append(rle)

    annotations['bboxes'] = np.array(annotations['bboxes'], dtype=np.float32) if annotations['bboxes'] else np.zeros((0, 4), dtype=np.float32)
    annotations['bboxes_ignore'] = np.array(annotations['bboxes_ignore'], dtype=np.float32) if annotations['bboxes_ignore'] else np.zeros((0, 4), dtype=np.float32)
    
    return annotations


def flickr_preprocess(example):
    data_labels = example['labels']
    masks = example['masks']
    caption = example['caption']
    tokens_positive = example['tokens_positive']

    def sort_by_start_index(items, order):
        return [items[i] for i in order]

    phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
    masks = sort_by_start_index(masks, phrase_order)
    data_labels = sort_by_start_index(data_labels, phrase_order)
    tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

    conversations = grandf_conversation(caption, tokens_positive)
    example['conversations'] = conversations
    example['labels'] = data_labels
    example['masks'] = masks
    example['tokens_positive'] = tokens_positive
    
    return example


def glamm_flickr_map_fn(example):
    example = flickr_parse_annotations(example)
    example = flickr_preprocess(example)

    messages = example['conversations']
    input_text = ''
    conversation = []
    
    while messages and messages[0]['from'] == 'gpt':
        messages = messages[1:]
    
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input_text += msg['value']
            
        elif msg['from'] == 'gpt':
            conversation.append({'input': input_text, 'output': msg['value']})
            input_text = ''
        else:
            raise NotImplementedError
    
    example.update({'conversation': conversation})
    return example


glamm_openpsg_map_fn = glamm_granf_map_fn