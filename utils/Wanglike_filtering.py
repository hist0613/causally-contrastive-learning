#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))
def l2distance(A, B):
       return norm(A-B)


# In[6]:


DATASET_PATH = "../dataset/IMDb/triplet_automated_averaged_gradient_1word_augmented_1x_partition_075_aclImdb/"
OUTPUT_PATH  = "../dataset/IMDb/triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_partition_075_aclImdb/"

BATCH_SIZE = 4

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()

with open(os.path.join(DATASET_PATH, "train.json")) as f:
    data = json.load(f)

causal_sentences = [d['negative_text'] for d in data]
causal_encodings = tokenizer(causal_sentences, truncation=True, padding=True)

causal_dataset = IMDbDataset(causal_encodings)
causal_loader = DataLoader(causal_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[30]:


causal_reps = []
for batch in tqdm(causal_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask, output_hidden_states=True)[2]
        reps = torch.cat((logits[-1], logits[-2], logits[-3], logits[-4]), dim=-1)[:, 0, :]
        #causal_reps.append(logits.detach().cpu())
        causal_reps.append(reps.detach().cpu())

causal_reps = torch.cat(causal_reps)
causal_reps = causal_reps.detach().cpu().numpy()


# In[31]:


positive_reps = []
negative_reps = []


# In[32]:


for rep, d in zip(causal_reps, data):
    if d['label'] == [1.0, 0.0]:
        negative_reps.append((d['id'], rep))
    else:
        positive_reps.append((d['id'], rep))

for rep, d in tqdm(zip(causal_reps, data)):
    d['triplet_sample_mask'] = False

    if d['label'] == [1.0, 0.0]:
        for pr in positive_reps:
            if d['id'] == pr[0]:
                continue
            if cos_sim(rep, pr[1]) > 0.95:
                d['triplet_sample_mask'] = True
                break

    else:
        for nr in negative_reps:
            if d['id'] == nr[0]:
                continue
            if cos_sim(rep, nr[1]) > 0.95:
                d['triplet_sample_mask'] = True
                break
    


# In[9]:


import shutil
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
with open(os.path.join(OUTPUT_PATH, "train.json"), 'w') as f:
    json.dump(data, f)
shutil.copy(os.path.join(DATASET_PATH, "valid.json"), os.path.join(OUTPUT_PATH, "valid.json"))
shutil.copy(os.path.join(DATASET_PATH, "test.json"), os.path.join(OUTPUT_PATH, "test.json"))
print(sum([d['triplet_sample_mask'] for d in data])/len(data))

