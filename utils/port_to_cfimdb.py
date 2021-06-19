import json
import pickle
import shutil
import os

MATCH_ID_PATH = "../dataset/cf_imdb_matching_ids.pickle"
CF_DATASET_PATH = "../dataset/CFIMDb/cf_augmented_aclImdb"

DATASET_NAME = "triplet_posneg_1word_augmented_1x_aclImdb/"
DATASET_PATH = os.path.join("../dataset_invalid/IMDb/", DATASET_NAME)
OUTPUT_PATH = os.path.join("../dataset/CFIMDb/", DATASET_NAME)

with open(MATCH_ID_PATH, 'rb') as f:
    match_id = pickle.load(f)

with open(os.path.join(DATASET_PATH, "train.json")) as f:
    train = json.load(f)

with open(os.path.join(DATASET_PATH, "valid.json")) as f:
    valid = json.load(f)

data = train + valid

output = []
for mi in match_id:
    for d in data:
        if mi == d['id']:
            output.append(d)
            break

os.makedirs(OUTPUT_PATH)
with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(output, f)

shutil.copy(os.path.join(CF_DATASET_PATH, "valid.json"), os.path.join(OUTPUT_PATH, "valid.json"))
shutil.copy(os.path.join(CF_DATASET_PATH, "test.json"), os.path.join(OUTPUT_PATH, "test.json"))
