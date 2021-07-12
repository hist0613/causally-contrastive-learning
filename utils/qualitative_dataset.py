import torch
import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)

DATASET_PATH = "../dataset/SST-2/lm_qualitative"
REPS_PATH = "../reps"
OUTPUT_PATH = "../dataset/SST-2/lm_qualitative_softed"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def distill_label(data, logits):
    for d, l in zip(data, logits):
        d.pop('positive_text', None)
        d.pop('negative_text', None)
        d['soft_label'] = l.tolist()
        #d['label'] = l.tolist()
    return data

train_logits = torch.load(os.path.join(REPS_PATH, "train_logits.pt"))

train_logits = torch.sigmoid(train_logits)


with open(os.path.join(DATASET_PATH, "train.json")) as f:
    train_data = json.load(f)


distilled_train_data = distill_label(train_data, train_logits)



with open(os.path.join(DATASET_PATH, "valid.json"), 'r') as f:
    valid_data = json.load(f)

with open(os.path.join(DATASET_PATH, "test.json"), 'r') as f:
    test_data = json.load(f)

with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(distilled_train_data, f, indent=4)

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(valid_data, f)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test_data, f)

