import os
import json
from transformers import BertTokenizer


DATASET_PATH = "../dataset/IMDb/original_augmented_1x_aclImdb/" 
OUTPUT_PATH = "../dataset/IMDb/original_augmented_1x_shortest_025_aclImdb/"
#DATASET_PATH = "../dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb" 
#OUTPUT_PATH = "../dataset/IMDb/triplet_posneg_1word_augmented_1x_shortest_025_aclImdb/"
#DATASET_PATH = "../dataset/IMDb/triplet_automated_averaged_gradient_1word_augmented_1x_aclImdb/" 
#OUTPUT_PATH = "../dataset/IMDb/triplet_automated_averaged_gradient_1word_augmented_shortest_025_aclImdb/"
#DATASET_PATH = "../dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" 
#OUTPUT_PATH = "../dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_shortest_025_aclImdb/"





tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
 
with open(os.path.join(DATASET_PATH, "train.json")) as f:
    data = json.load(f)


output = []
for d in data:
    anc_text =d['anchor_text']
    if len(d['anchor_text']) <= 696:
        output.append(d)

print(len(data))
print(len(output))

with open(os.path.join(DATASET_PATH, "valid.json"), 'r') as f:
    valid_data = json.load(f)

with open(os.path.join(DATASET_PATH, "test.json"), 'r') as f:
    test_data = json.load(f)


with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(output, f)
        
with open(os.path.join(OUTPUT_PATH, "valid.json"), 'w') as f:
    json.dump(valid_data, f)

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(test_data, f)

