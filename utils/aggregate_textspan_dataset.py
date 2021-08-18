import torch
import os
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
from nltk.tokenize import sent_tokenize
import copy

random.seed(42)
DATASET_NAME = "IMDb"
SMALL_NAME = "aclImdb"

DATASET_PATH = f"../dataset/{DATASET_NAME}/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_1word_augmented_1x_{SMALL_NAME}"
OUTPUT_PATH =  f"../dataset/{DATASET_NAME}/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_1word_augmented_1x_{SMALL_NAME}"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

with open(os.path.join(DATASET_PATH, "train.json")) as f:
    data = json.load(f)

output = []
cur_id = data[0]['id']
mask_flag = False
agg_anchor_text = ''
agg_positive_text = ''
agg_negative_text = ''

for idx, d in enumerate(data):
    if d['id'] != cur_id:
        sample = copy.deepcopy(data[idx-1])
        sample['anchor_text'] = agg_anchor_text
        sample['positive_text'] = agg_positive_text
        sample['negative_text'] = agg_negative_text
        sample['triplet_sample_mask'] = mask_flag
        output.append(sample)

        cur_id = d['id']
        mask_flag = False
        agg_anchor_text = ''
        agg_positive_text = ''
        agg_negative_text = ''

    agg_anchor_text += d['anchor_text']
    agg_anchor_text += ' '
    
    if d['triplet_sample_mask']:
        mask_flag = True
        agg_positive_text += d['positive_text']
        agg_positive_text += ' '
        agg_negative_text += d['negative_text']
        agg_negative_text += ' '
    else:
        agg_positive_text += d['anchor_text']
        agg_positive_text += ' '
        agg_negative_text += d['anchor_text']
        agg_negative_text += ' '


sample = copy.deepcopy(data[idx])
sample['anchor_text'] = agg_anchor_text
sample['positive_text'] = agg_positive_text
sample['negative_text'] = agg_negative_text
sample['triplet_sample_mask'] = mask_flag
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



