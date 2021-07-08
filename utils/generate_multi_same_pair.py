import json
import random
from transformers import BertTokenizer
import os

DATASET_PATH = "../dataset/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_sst2"
OUTPUT_PATH = "../dataset/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_1word_augmented_1x_sst2"
#os.makedirs(OUTPUT_PATH)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

with open(os.path.join(DATASET_PATH, "train.json")) as f:
    train_data = json.load(f)

for d in train_data:
    d['aux_positive_text'] = d['positive_text']
    d['aux_negative_text'] = d['negative_text']
    d['aux_triplet_sample_mask'] = False

    if d['triplet_sample_mask']:
        anc_tokens = tokenizer.encode(d['anchor_text'])[1:-1]
        pos_tokens = tokenizer.encode(d['positive_text'])[1:-1]
        neg_tokens = tokenizer.encode(d['negative_text'])[1:-1]
        mask_id = tokenizer.mask_token_id
        
        
        if len(anc_tokens) < 3:
            continue
        while True:
            aux_pos_tokens = anc_tokens[:]
            masking_idx = random.randint(0, len(aux_pos_tokens) - 1)
            aux_pos_tokens[masking_idx] = mask_id
            if aux_pos_tokens != pos_tokens and aux_pos_tokens != neg_tokens:
                d['aux_positive_text'] = tokenizer.decode(aux_pos_tokens)
                d['aux_triplet_sample_mask'] = True
                break


with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(train_data, f)


