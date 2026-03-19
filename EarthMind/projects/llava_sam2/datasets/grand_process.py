import numpy as np
import random
from xtuner.utils import DEFAULT_IMAGE_TOKEN

GCG_QUESTIONS = [
    DEFAULT_IMAGE_TOKEN + 'Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    DEFAULT_IMAGE_TOKEN + 'Can you provide a brief description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    DEFAULT_IMAGE_TOKEN + 'Could you give a brief explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Could you give me an brief explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Could you provide me with a briefly analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

def grand_parse_annotations(example):
    annotations = {
        'caption': [], 'masks': [],
        'tokens_positive': [], 'labels': []}
    annotations['caption'] = example['dense_caption']['caption'].strip('"').strip()
    object_infos = example['dense_caption']['details']

    all_seg_objects_dict = {}
    for seg_object_dict in example["objects"]:
        all_seg_objects_dict[seg_object_dict['id']] = seg_object_dict
    for seg_object_dict in example["floating_objects"]:
        all_seg_objects_dict[seg_object_dict['id']] = seg_object_dict

    for object_info in object_infos:
        ids = object_info["ids"]
        if object_info["tokens_positive"] is None:
            continue
        annotations['labels'].append(object_info["phrase"])
        annotations['tokens_positive'].append(object_info["tokens_positive"])
        _masks = []
        for _id in ids:
            _masks.append(all_seg_objects_dict[_id]['segmentation'])
        annotations['masks'].append(_masks)
    return annotations

def grand_conversation(caption, tokens_positive):
    question = random.choice(GCG_QUESTIONS).strip()

    # Prepare caption with tags
    def tag_caption(caption, tokens):
        for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
            caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
        return caption

    detailed_answer = tag_caption(caption, tokens_positive)

    conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': detailed_answer}]
    return conversations

def grand_preprocess(example):
    data_labels = example['labels']
    masks = example['masks']
    caption = example['caption']
    tokens_positive = example['tokens_positive']

    # Function to sort elements based on the start index of each phrase
    def sort_by_start_index(items, order):
        return [items[i] for i in order]

    # Sort phrases based on their appearance in the sentence
    phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
    masks = sort_by_start_index(masks, phrase_order)
    data_labels = sort_by_start_index(data_labels, phrase_order)
    tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

    conversations = grand_conversation(caption, tokens_positive)
    example['conversations'] = conversations
    example['labels'] = data_labels
    example['masks'] = masks
    example['tokens_positive'] = tokens_positive
    return example

def glamm_grand_map_fn(example):
    # example {'file_name': str, "height": int, "width": int, "image_id": str, caption: "str",
    # "groundings": {ground_words: {'token_positives', 'rle_masks', }}}
    example = grand_parse_annotations(example)
    # example 'labels': [], 'caption': str, 'masks': [], 'tokens_positive': [], 'file_name': image_file

    example = grand_preprocess(example)

    # do llava preprocess
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example




