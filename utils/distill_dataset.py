import torch
import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)

DATASET_PATH = "../dataset/reform_aclImdb"
REPS_PATH = "../reps"
OUTPUT_PATH = "../dataset/distill_aclImdb"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def distill_label(data, logits):
    for d, l in zip(data, logits):
        d['label'] = l.tolist()
    return data

train_logits = torch.load(os.path.join(REPS_PATH, "train_logits.pt"))
val_logits = torch.load(os.path.join(REPS_PATH, "val_logits.pt"))

train_logits = torch.sigmoid(train_logits)
val_logits = torch.sigmoid(val_logits)


with open(os.path.join(DATASET_PATH, "train.json")) as f:
    train_data = json.load(f)

with open(os.path.join(DATASET_PATH, "valid.json")) as f:
    val_data = json.load(f)

distilled_train_data = distill_label(train_data, train_logits)
distilled_val_data = distill_label(val_data, val_logits)

with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(distilled_train_data, f)

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(distilled_val_data, f)
