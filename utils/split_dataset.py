import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import pickle
import random
from tqdm import tqdm
random.seed(42)

DATASET_NAME = "IMDb"
DATASET_PATH = f"../dataset/{DATASET_NAME}"

SPLIT_PATH = f"{DATASET_NAME}_train_split.pickle"
SPLIT_SIZE = 0.9
split_indices = None

for folder in tqdm(os.listdir(DATASET_PATH)):
    if "triplet" not in folder:
        continue


    with open(os.path.join(DATASET_PATH, folder, "train.json")) as f:
        data = json.load(f)
    
    if not split_indices:
        split_indices = list(range(len(data)))
        random.shuffle(split_indices)
        with open(os.path.join(DATASET_PATH, SPLIT_PATH), 'wb') as f:
            pickle.dump(split_indices, f)
    
    train_indices = split_indices[:int(len(data) * SPLIT_SIZE)]
    val_indices = split_indices[int(len(data) * SPLIT_SIZE):]

    train = []
    for ti in train_indices:
        train.append(data[ti])
    val = []
    for vi in val_indices:
        val.append(data[vi])


    with open(os.path.join(DATASET_PATH, folder, "train.json"), 'w') as f:
        json.dump(train, f)

    with open(os.path.join(DATASET_PATH, folder, "valid.json"), 'w') as f:
        json.dump(val, f)


