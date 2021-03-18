import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)


DATASET_PATH = "../dataset/MASKER_DATASET/datasets/foods/"
TRAIN_SPLIT = "foods_train.txt"
TEST_SPLIT = "foods_test.txt"
OUTPUT_PATH = "../dataset/FineFood/original_augmented_1x_finefood"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def read_food_split(split_dir):
    with open(split_dir, encoding='utf-8') as f:
        lines = f.readlines()

    texts = []
    labels = []
    for line in lines:
        toks = line.split(':')

        if int(toks[1]) == 1:  # pre-defined class 0
            label = 0
        elif int(toks[1]) == 5:  # pre-defined class 1
            label = 1
        else:
            continue

        text = toks[0]
        texts.append(text)
        labels.append(label)

    return texts, labels

def reform(texts, labels):
    output = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        sample = dict()
        sample['id'] = i
        #sample['text'] = text
        sample['anchor_text'] = text
        sample['positive_text'] = ''
        sample['negative_text'] = ''
        #sample['label'] = label 
        
        if label == 0:
            sample['label'] = [1., 0.]
        else:
            sample['label'] = [0., 1.]
        
        output.append(sample)

    return output

train_texts, train_labels = read_food_split(os.path.join(DATASET_PATH, TRAIN_SPLIT))
test_texts, test_labels = read_food_split(os.path.join(DATASET_PATH, TEST_SPLIT))
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.3, random_state=44)

#Fake Validation Set
#val_texts, val_labels = [train_texts[0]], [train_labels[0]]

train = reform(train_texts, train_labels)
val = reform(val_texts, val_labels)
test = reform(test_texts, test_labels)

with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(train, f)

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(val, f)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test, f)

