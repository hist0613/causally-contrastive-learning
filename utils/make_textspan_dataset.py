import torch
import os
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
from nltk.tokenize import sent_tokenize
import copy
import numpy as np

random.seed(42)
DATASET_NAME = "IMDb"
SMALL_NAME = "aclImdb"

DATASET_PATH = f"../dataset/{DATASET_NAME}/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_{SMALL_NAME}"
OUTPUT_PATH = f"../dataset/{DATASET_NAME}/original_augmented_1x_sentTokenize_{SMALL_NAME}"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

with open(os.path.join(DATASET_PATH, "train.json")) as f:
    data = json.load(f)


sent_lens = []
output = []
for d in data:
    sents = sent_tokenize(d['anchor_text'])
    sent_lens.append(len(sents))
    for sent in sents:
        sample = copy.deepcopy(d)
        sample['anchor_text'] = sent
        sample['positive_text'] = None
        sample['negative_text'] = None
        sample['triplet_sample_mask'] = False
        output.append(sample)

avg_sent_lens = np.mean(sent_lens)
print(f"the avg. length of sentences is {avg_sent_lens}")

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



