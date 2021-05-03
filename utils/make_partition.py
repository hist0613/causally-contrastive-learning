import torch
import os
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
import numpy as np

random.seed(42)
DATASET_NAME = "FineFood_full"
SMALL_NAME = "finefood"


DATASET_PATH = f"../dataset/{DATASET_NAME}/original_augmented_1x_{SMALL_NAME}"
PARTITION_PATH = "../partition_indices"
PARTITION_SIZE_DICT = {"025": 0.25, "050": 0.5, "075": 0.75}

if not os.path.exists(PARTITION_PATH):
    os.makedirs(PARTITION_PATH)

with open(os.path.join(DATASET_PATH, "train.json"), 'r') as f:
    train_data = json.load(f)

train_indices = list(range(len(train_data)))
random.shuffle(train_indices)

for p_str, p_size in PARTITION_SIZE_DICT.items():
    partition_index = train_indices[:int(len(train_data) * p_size)]
    with open(os.path.join(PARTITION_PATH, f"{SMALL_NAME}_{p_str}_index.pkl"), 'wb') as f:
        pickle.dump(partition_index, f)


"""
with open(os.path.join(CF_EXAMPLES_PATH, "triplets_dev.pickle"), 'rb') as fb:
    paired_val = pickle.load(fb)

val_data = reform(*return_triplet_text(paired_val))

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(val_data, f)

with open(os.path.join(CF_EXAMPLES_PATH, "triplets_test.pickle"), 'rb') as fb:
    paired_test = pickle.load(fb)

test_data = reform(*return_triplet_text(paired_test))

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test_data, f)
"""
