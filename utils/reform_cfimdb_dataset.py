import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)


DATASET_PATH = ["../dataset/CFIMDb/aclImdb/orig",
                "../dataset/CFIMDb/aclImdb/new", 
                "../dataset/CFIMDb/aclImdb/combined"]
TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"
OUTPUT_PATH = ["../dataset/CFIMDb/original_augmented_1x_aclImdb",
               "../dataset/CFIMDb/revised_augmented_1x_aclImdb",
               "../dataset/CFIMDb/combined_augmented_1x_aclImdb"]


for each_output_path in OUTPUT_PATH:
    if not os.path.exists(each_output_path):
        os.makedirs(each_output_path)

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

for each_dataset_path, each_output_path in zip(DATASET_PATH, OUTPUT_PATH):
    train_texts, train_labels = read_tsv(os.path.join(each_dataset_path, TRAIN_SPLIT + '.tsv'))
    val_texts, val_labels = read_tsv(os.path.join(each_dataset_path, DEV_SPLIT + '.tsv'))
    test_texts, test_labels = read_tsv(os.path.join(each_dataset_path, TEST_SPLIT + '.tsv'))

    train = reform(train_texts, train_labels)
    val = reform(val_texts, val_labels)
    test = reform(test_texts, test_labels)

    with open(os.path.join(each_output_path, "train.json"), 'w') as f:
        json.dump(train, f)

    with open(os.path.join(each_output_path, "valid.json"), 'w') as f:
        json.dump(val, f)

    with open(os.path.join(each_output_path, "test.json"), 'w') as f:
        json.dump(test, f)

