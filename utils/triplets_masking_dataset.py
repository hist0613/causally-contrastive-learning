import torch
import os
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)
DATASET_NAME = "SST-2"
SMALL_NAME = "sst2"

CF_EXAMPLES_PATH = f"../dataset/{DATASET_NAME}/cf_augmented_examples"
DATASET_PATH = f"../dataset/{DATASET_NAME}/original_augmented_1x_{SMALL_NAME}"
OUTPUT_PATH = f"../dataset/{DATASET_NAME}/triplet_automated_averaged_gradient_LM_dropout_02_flip_1word_augmented_1x_{SMALL_NAME}"
REPS_PATH = "../reps"
FILE_NAME = "triplets_automated_averaged_gradient_LM_dropout_02_flip_sampling1_augmenting1_train.pickle"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def return_triplet_text(data):

    anchor_texts = []
    samelabel_texts = []
    difflabel_texts = []
    triplet_sample_masks = []
    labels = []
    flipped_cnt = 0


    for d in data:
        anchor_text = d[1]
        if not d[4]:    
            samelabel_text = d[3]
            difflabel_text = d[2]
            flipped_cnt += 1
        else:
            samelabel_text = d[1]
            difflabel_text = d[1]

        triplet_sample_mask = d[4]

        label = 0 if d[0] == 'Negative' else 1
           
        anchor_texts.append(anchor_text)
        samelabel_texts.append(samelabel_text)
        difflabel_texts.append(difflabel_text)
        triplet_sample_masks.append(triplet_sample_mask)
        labels.append(label)
    print(flipped_cnt / len(data))
    return anchor_texts, samelabel_texts, difflabel_texts, triplet_sample_masks, labels

def reform(anchor_texts, positive_texts, negative_texts, triplet_sample_masks, labels):
    output = []
    for i, (anc_text, pos_text, neg_text, tri_mask, label) in enumerate(zip(anchor_texts, positive_texts, negative_texts, triplet_sample_masks, labels)):
        sample = dict()
        sample['id'] = i
        sample['anchor_text'] = anc_text
        sample['positive_text'] = pos_text
        sample['negative_text'] = neg_text
        #sample['label'] = label 
       
        if tri_mask:
            sample['triplet_sample_mask'] = False
        else:
            sample['triplet_sample_mask'] = True

        if label == 0:
            sample['label'] = [1., 0.]
        else:
            sample['label'] = [0., 1.]
        
        output.append(sample)

    return output


with open(os.path.join(CF_EXAMPLES_PATH, FILE_NAME), 'rb') as fb:
    paired_train = pickle.load(fb)

train_data = reform(*return_triplet_text(paired_train))


with open(os.path.join(DATASET_PATH, "valid.json"), 'r') as f:
    valid_data = json.load(f)

with open(os.path.join(DATASET_PATH, "test.json"), 'r') as f:
    test_data = json.load(f)


with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(train_data, f)

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
