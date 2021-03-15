import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
DATASET_PATH = "dataset/aclImdb"
REFORMED_DATASET_PATH = "dataset/reformed_aclImdb"
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

""" This part is for loading original IMDB dataset """
def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

# Set dataset
train_texts, train_labels = read_imdb_split(os.path.join(DATASET_PATH, TRAIN_SPLIT))
test_texts, test_labels = read_imdb_split(os.path.join(DATASET_PATH, TEST_SPLIT))
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    


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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

optim = AdamW(model.parameters(), lr=5e-5)



#Train & Evaluation
for epoch in range(3):

    model.train()
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

    model.eval()
    total_loss = 0
    cor_cnt = 0
    total_size = 0
    for batch in tqdm(val_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs[0].item()

            logits = outputs[1]
            cor_cnt += correct_count(logits, labels)
            total_size += len(labels)

    accuracy = cor_cnt * 1.0 / total_size
    print(f"Total Loss: {total_loss}")
    print(f"Accuracy: {accuracy}")

total_loss = 0
cor_cnt = 0
total_size = 0
for batch in tqdm(val_loader):
    with torch.no_grad():
	input_ids = batch['input_ids'].to(device)
	attention_mask = batch['attention_mask'].to(device)
	labels = batch['labels'].to(device)
	outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	total_loss += outputs[0].item()

	logits = outputs[1]
	cor_cnt += correct_count(logits, labels)
	total_size += len(labels)

accuracy = cor_cnt * 1.0 / total_size
print(f"Total Loss: {total_loss}")
print(f"Accuracy: {accuracy}")


