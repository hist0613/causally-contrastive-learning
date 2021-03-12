import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import random
random.seed(42)


DATASET_PATH = "../dataset/IMDb/aclImdb"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
#OUTPUT_PATH = "dataset/reform_aclImdb"
OUTPUT_PATH = "../dataset/IMDb/original_augmented_1x_aclImdb_full"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

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

train_texts, train_labels = read_imdb_split(os.path.join(DATASET_PATH, TRAIN_SPLIT))
test_texts, test_labels = read_imdb_split(os.path.join(DATASET_PATH, TEST_SPLIT))
#train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2, random_state=44)

#Fake Validation Set
val_texts, val_labels = [train_texts[0]], [train_labels[0]]

train = reform(train_texts, train_labels)
val = reform(val_texts, val_labels)
test = reform(test_texts, test_labels)

with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(train, f)

with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(val, f)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test, f)

