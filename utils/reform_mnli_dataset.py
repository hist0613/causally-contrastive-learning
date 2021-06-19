import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)

#GENRE_NAME = "all"
GENRE_NAME = "telephone"
#GENRE_NAME = "letters"
#GENRE_NAME = "facetoface"

DATASET_PATH = "../dataset/MULTINLI_DATASET/"
TRAIN_SPLIT = "multinli_1.0_train.jsonl"
TEST_SPLIT = "multinli_1.0_dev_matched.jsonl"
OUTPUT_PATH = f"../dataset/MultiNLI_{GENRE_NAME}/original_augmented_1x_mnli"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def read_mnli_split(split_dir):
    with open(split_dir, encoding='utf-8') as f:
        lines = f.readlines()

    lines = [json.loads(l) for l in lines]

    texts = []
    labels = []
    for line in lines:
        if line['genre'] != GENRE_NAME:
            continue
        text = line['sentence1'] + " [SEP] " + line['sentence2']
        label = line['gold_label']
        texts.append(text)
        labels.append(label)

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

train_texts, train_labels = read_mnli_split(os.path.join(DATASET_PATH, TRAIN_SPLIT))
if len(train_texts):
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.1, random_state=42)

    train = reform(train_texts, train_labels)
    val = reform(val_texts, val_labels)

    with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
        json.dump(train, f)

    with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
        json.dump(val, f)


test_texts, test_labels = read_mnli_split(os.path.join(DATASET_PATH, TEST_SPLIT))
test = reform(test_texts, test_labels)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test, f)

