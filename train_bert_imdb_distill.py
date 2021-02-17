import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
DATASET_PATH = "dataset/aclImdb"
#REFORMED_DATASET_PATH = "dataset/reform_aclImdb"
#OUTPUT_PATH = "output"
REFORMED_DATASET_PATH = "dataset/distill_aclImdb"
OUTPUT_PATH = "distill_output"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
num_labels = 2

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
    _, label_indices = torch.max(labels, dim=1)
    correct = torch.sum(indices == label_indices)
    return correct.item()

# Load dataset
with open(os.path.join(REFORMED_DATASET_PATH, "train.json")) as f:
    train = json.load(f) 
with open(os.path.join(REFORMED_DATASET_PATH, "valid.json")) as f:
    val = json.load(f)
with open(os.path.join(REFORMED_DATASET_PATH, "test.json")) as f:
    test = json.load(f)

train_texts = [d['text'] for d in train]
train_labels = [d['label'] for d in train]
val_texts = [d['text'] for d in val]
val_labels = [d['label'] for d in val]
test_texts = [d['text'] for d in test]
test_labels = [d['label'] for d in test]

#Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Encode dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

#make dataset class
train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)

#Train & Evaluation
best_epoch = -1
best_acc = 0
steps = 0
for epoch in range(3):
    model.train()
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        _, labels = torch.max(labels, dim=1)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]

        """
        # Compute Binary Cross Entropy
        outputs = model(input_ids, attention_mask=attention_mask)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(outputs[0].view(-1, num_labels), labels.view(-1, num_labels))
        """
        loss.backward()
        optim.step()
        steps += 1
    model.eval()
    cor_cnt = 0
    total_size = 0
    for batch in tqdm(val_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

            logits = outputs[0]
            cor_cnt += correct_count(logits, labels)
            total_size += len(labels)

    accuracy = cor_cnt * 1.0 / total_size
    if accuracy > best_acc:
        best_epoch = epoch
        best_acc = accuracy
    print(f"Accuracy: {accuracy}")
    model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}"))

print(f"\nBest Model is epoch {best_epoch}. load and evaluate test...")
model = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_PATH, f'epoch_{epoch}'))
model.to(device)

# Test
cor_cnt = 0
total_size = 0
for batch in tqdm(test_loader):
    with torch.no_grad():
    	input_ids = batch['input_ids'].to(device)
    	attention_mask = batch['attention_mask'].to(device)
    	labels = batch['labels'].to(device)
    	outputs = model(input_ids, attention_mask=attention_mask)

    	logits = outputs[0]
    	cor_cnt += correct_count(logits, labels)
    	total_size += len(labels)

accuracy = cor_cnt * 1.0 / total_size
print(f"Accuracy: {accuracy}")

