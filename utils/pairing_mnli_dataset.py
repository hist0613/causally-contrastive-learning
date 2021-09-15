import torch
import os
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)
DATASET_NAME = "CFNLI"
SMALL_NAME = "cfnli"

CF_EXAMPLES_PATH = f"../dataset/{DATASET_NAME}/cf_augmented_examples"
DATASET_PATH = f"../dataset/{DATASET_NAME}/original_augmented_1x_{SMALL_NAME}"
OUTPUT_PATH = f"../dataset/{DATASET_NAME}/triplet_posneg_1word_augmented_1x_{SMALL_NAME}"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def return_triplet_text(data):

    ent_pool = []
    cont_pool = []
    neut_pool = []

    for d in data:
        if d[0] == 'contradiction': 
            cont_pool.append(d[1])
        elif d[0] == 'entailment': 
            ent_pool.append(d[1])
        else:
            neut_pool.append(d[1])

    print(len(cont_pool))
    print(len(ent_pool))
    print(len(neut_pool))

    anchor_texts = []
    samelabel_texts = []
    difflabel_texts = []
    labels = []

    for d in data:
        anchor_text = d[1]
        if d[0] == 'contradiction': 
            label = 0 
        elif d[0] == 'entailment': 
            label = 1
        else:
            label = 2

        
        if label == 0:
            samelabel_text = random.choice(cont_pool)
            difflabel_text = random.choice(ent_pool + neut_pool)

        elif label == 1:
            samelabel_text = random.choice(ent_pool)
            difflabel_text = random.choice(cont_pool + neut_pool)
        else:
            samelabel_text = random.choice(neut_pool)
            difflabel_text = random.choice(cont_pool + ent_pool)
            
        anchor_texts.append(anchor_text)
        samelabel_texts.append(samelabel_text)
        difflabel_texts.append(difflabel_text)
        labels.append(label)

    return anchor_texts, samelabel_texts, difflabel_texts, labels

def reform(anchor_texts, positive_texts, negative_texts, labels):
    output = []
    for i, (anc_text, pos_text, neg_text, label) in enumerate(zip(anchor_texts, positive_texts, negative_texts, labels)):
        sample = dict()
        sample['id'] = i
        sample['anchor_text'] = anc_text
        sample['positive_text'] = pos_text
        sample['negative_text'] = neg_text
        #sample['label'] = label

        sample['triplet_sample_mask'] = True

        
        if label == 0:
            sample['label'] = [1., 0., 0.]
        elif label == 1:
            sample['label'] = [0., 1., 0.]
        else:
            sample['label'] = [0., 0., 1.]

        
        output.append(sample)

    return output


with open(os.path.join(CF_EXAMPLES_PATH, "triplets_automated_averaged_gradient_sampling1_augmenting1_train.pickle"), 'rb') as fb:
#with open(os.path.join(CF_EXAMPLES_PATH, "triplets_sampling2_augmenting3_train.pickle"), 'rb') as fb:
    paired_train = pickle.load(fb)

train_data = reform(*return_triplet_text(paired_train))

with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(train_data, f)


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
