import json
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
DATASET_PATH = "dataset/aclImdb"
REFORMED_DATASET_PATH = "dataset/reform_aclImdb"
OUTPUT_PATH = "checkpoints/output"
REPS_PATH = "reps"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def correct_count(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item()

# Load dataset
with open(os.path.join(REFORMED_DATASET_PATH, "train.json")) as f:
    train = json.load(f) 
with open(os.path.join(REFORMED_DATASET_PATH, "valid.json")) as f:
    val = json.load(f)

train_texts = [d['text'] for d in train]
train_labels = [d['label'] for d in train]
val_texts = [d['text'] for d in val]
val_labels = [d['label'] for d in val]
#Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Encode dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

#make dataset class
train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_PATH, 'epoch_2'))
model.to(device)


train_logits = []
for batch in tqdm(train_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        train_logits.append(logits)

val_logits = []
for batch in tqdm(val_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        val_logits.append(logits)

torch.save(torch.cat(train_logits, 0), os.path.join(REPS_PATH, "train_logits.pt"))
torch.save(torch.cat(val_logits, 0), os.path.join(REPS_PATH, "val_logits.pt"))
