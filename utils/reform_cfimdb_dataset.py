import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)


DATASET_PATH = "../dataset/CFIMDb/aclImdb/new"
TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"
OUTPUT_PATH = "../dataset/CFIMDb/revised_augmented_1x_aclImdb"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def read_tsv(tsv_file):
    f = open(tsv_file)
    data = f.readlines()[1:]

    texts = []
    labels = []
    for d in data:
        label = d.split("\t")[0]
        text = d.split("\t")[1]

        texts.append(text)
        if label == 'Negative':
            labels.append(0)
        elif label == 'Positive':
            labels.append(1)
        else:
            raise ValueError("There is no label??")

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

train_texts, train_labels = read_tsv(os.path.join(DATASET_PATH, TRAIN_SPLIT + '.tsv'))
val_texts, val_labels = read_tsv(os.path.join(DATASET_PATH, DEV_SPLIT + '.tsv'))
test_texts, test_labels = read_tsv(os.path.join(DATASET_PATH, TEST_SPLIT + '.tsv'))

train = reform(train_texts, train_labels)
val = reform(val_texts, val_labels)
test = reform(test_texts, test_labels)

with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(train, f)

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(val, f)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test, f)

