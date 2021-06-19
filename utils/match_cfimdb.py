import json
import re
import string
import os
import pickle

CFIMDB_PATH = "../dataset_invalid/CFIMDb/cf_augmented_aclImdb"
IMDB_PATH = "../dataset_invalid/IMDb/original_augmented_1x_aclImdb"

with open(os.path.join(CFIMDB_PATH, "train.json"), encoding="utf-8") as f:
    cf_data = json.load(f)

with open(os.path.join(IMDB_PATH, "train.json"), encoding="utf-8") as f:
    ori_data = json.load(f)

cf_texts = [" ".join(re.findall("[a-zA-Z]+", cd['anchor_text'])) for cd in cf_data]
ori_tuple = [(od['id'], " ".join(re.findall("[a-zA-Z]+", od['anchor_text']))) for od in ori_data]

match_ids = []
for ot in ori_tuple:
    if ot[1] in cf_texts:
        match_ids.append(ot[0])
    if "the movie is so slow" in ot[1]:
        print(ot[1])
        match_ids.append(ot[0])
    if "and naked bodies are being enjoyed" in ot[1]:
        print(ot[1])
        match_ids.append(ot[0])

with open("cf_imdb_matching_ids.pickle", 'wb') as f:
        pickle.dump(match_ids, f)
