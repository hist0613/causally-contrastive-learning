import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)

DATASET_PATH = "../dataset/CFNLI/cfnli/revised_combined/"
TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"
OUTPUT_PATH = "../dataset/CFNLI/revised_combined_augmented_1x_cfnli"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def read_tsv(tsv_file):
    f = open(tsv_file)
    data = f.readlines()[1:]

    texts = []
    labels = []
    for d in data:
        sentence1 = d.split("\t")[0]
        sentence2 = d.split("\t")[1]
        label = d.split("\t")[2]
        text = sentence1 + " [SEP] " + sentence2
        texts.append(text)
        labels.append(label.strip())

    return texts, labels

def reform(texts, labels):
    neg_cnt = 0
    neu_cnt = 0
    pos_cnt = 0
    output = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        sample = dict()
        sample['id'] = i
        #sample['text'] = text
        sample['anchor_text'] = text
        sample['positive_text'] = ''
        sample['negative_text'] = ''
        #sample['label'] = label 
        
        if label == 'contradiction':
            neg_cnt += 1
            sample['label'] = [1., 0., 0.]
        elif label == 'entailment':
            pos_cnt += 1
            sample['label'] = [0., 1., 0.]
        else:
            neu_cnt += 1
            sample['label'] = [0., 0., 1.]
        
        output.append(sample)

    print(neg_cnt, pos_cnt, neu_cnt)

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


