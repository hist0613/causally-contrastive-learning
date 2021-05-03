import torch
import os
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)
DATASET_NAME = "FineFood_full"
SMALL_NAME = "finefood"
PARTITION_SIZE_DICT = {"025": 0.2, "050": 0.5, "075": 0.75}
PARTITION_SIZE = "075" # Dot is skipped

DATASET_PATH = f"../dataset/{DATASET_NAME}/original_augmented_1x_{SMALL_NAME}"
OUTPUT_PATH = f"../dataset/{DATASET_NAME}/original_augmented_1x_partition_{PARTITION_SIZE}_{SMALL_NAME}"
PARTITION_PATH = "../partition_indices"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

with open(os.path.join(PARTITION_PATH, f"{SMALL_NAME}_{PARTITION_SIZE}_index.pkl"), 'rb') as f:
    partition_index = pickle.load(f)

with open(os.path.join(DATASET_PATH, "train.json"), 'r') as f:
    train_data = json.load(f)

with open(os.path.join(DATASET_PATH, "valid.json"), 'r') as f:
    valid_data = json.load(f)

with open(os.path.join(DATASET_PATH, "test.json"), 'r') as f:
    test_data = json.load(f)

partition_train = []
for p_idx in partition_index:
    partition_train.append(train_data[p_idx])


with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(partition_train, f)

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(valid_data, f)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test_data, f)


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
