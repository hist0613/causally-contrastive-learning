import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import pickle
import random
from tqdm import tqdm
random.seed(42)

DATASET_NAME = "SST-2"
DATASET_PATH = f"../dataset/{DATASET_NAME}"

SPLIT_PATH = f"{DATASET_NAME}_train_split.pickle"
SPLIT_SIZE = 0.9
split_indices = None

for folder in os.listdir(DATASET_PATH):
    data = None
    train = []
    val = []
    if "triplet" not in folder:
        continue

    try:
        with open(os.path.join(DATASET_PATH, folder, "train.json")) as f:
            data = json.load(f)
    except:
        print(f"ERRORED: {folder}")
        continue
    
    if not split_indices:
        split_indices = list(range(len(data)))
        random.shuffle(split_indices)
        with open(os.path.join(DATASET_PATH, SPLIT_PATH), 'wb') as f:
            pickle.dump(split_indices, f)
        print("indices is created...")


    train_indices = split_indices[:int(len(data) * SPLIT_SIZE)]
    val_indices = split_indices[int(len(data) * SPLIT_SIZE):]
    try:
        for ti in train_indices:
            train.append(data[ti])
        for vi in val_indices:
            val.append(data[vi])
    except:
        print(f"ERRORED: {folder}")
        continue

    with open(os.path.join(DATASET_PATH, folder, "train.json"), 'w') as f:
        json.dump(train, f)

    with open(os.path.join(DATASET_PATH, folder, "valid.json"), 'w') as f:
        json.dump(val, f)

    print(f"PROCESSED: {folder}")
