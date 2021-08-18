import torch
import os
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
import numpy as np
from nltk.tokenize import sent_tokenize
import copy

random.seed(42)
DATASET_NAME = "FineFood_full"
SMALL_NAME = "finefood"

DATASET_PATH = f"../dataset/{DATASET_NAME}/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_1word_augmented_1x_{SMALL_NAME}"
OUTPUT_PATH =  f"../dataset/{DATASET_NAME}/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_{SMALL_NAME}"

NUM_MASK = 4

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

with open(os.path.join(DATASET_PATH, "train.json")) as f:
    data = json.load(f)

grouping = []
cur_id = data[0]['id']
cur_start = 0
for idx in range(len(data)):
    if data[idx]['id'] != cur_id:
        grouping.append(data[cur_start:idx])
        cur_id = data[idx]['id']
        cur_start = idx
grouping.append(data[cur_start:])

output = []
for group in grouping:
    mask_flags = [g['triplet_sample_mask'] for g in group]
    if sum(mask_flags) > NUM_MASK:
        mask_indices = np.nonzero(mask_flags)[0].tolist()
        remove_indices = random.choices(mask_indices, k=len(mask_indices)-NUM_MASK)
        for ri in remove_indices:
            mask_flags[ri] = False

    agg_anchor_text = ''
    agg_positive_text = ''
    agg_negative_text = ''
    for g, mask_flag in zip(group, mask_flags):
        agg_anchor_text += g['anchor_text']
        agg_anchor_text += ' '
        if mask_flag:
            agg_positive_text += g['positive_text']
            agg_positive_text += ' '
            agg_negative_text += g['negative_text']
            agg_negative_text += ' '
        else:
            agg_positive_text += g['anchor_text']
            agg_positive_text += ' '
            agg_negative_text += g['anchor_text']
            agg_negative_text += ' '
    sample = copy.deepcopy(group[0])
    sample['anchor_text'] = agg_anchor_text
    sample['positive_text'] = agg_positive_text
    sample['negative_text'] = agg_negative_text
    sample['triplet_sample_mask'] = True if sum(mask_flags) else False
    output.append(sample)

with open(os.path.join(DATASET_PATH, "valid.json"), 'r') as f:
    valid_data = json.load(f)

with open(os.path.join(DATASET_PATH, "test.json"), 'r') as f:
    test_data = json.load(f)

with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(output, f)

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(valid_data, f)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test_data, f)



