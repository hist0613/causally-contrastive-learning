import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from train_bert_imdb_pairwise_shellscript import BertForCounterfactualRobustness, IMDbDataset, CFIMDbDataset

parser = argparse.ArgumentParser(description='Counterfactual Robustness Inferencing')
parser.add_argument('--model-path', 
                    type=str, 
                    required=True,
                    help='path for model')
parser.add_argument('--dataset-path', 
                    type=str,
                    required=True,
                    help='path for dataset')
parser.add_argument('--reps-path', 
                    type=str, 
                    required=True,
                    help='path for output')
parser.add_argument('--batch-size', 
                    type=int,
                    required=True,
                    default=16,
                    help="batch size for train/valid/test")
parser.add_argument('--epoch',
                    type=int,
                    required=True,
                    default=15,
                    help="epoch number for training")
parser.add_argument('--use-margin-loss',
                    action='store_true',
                    help="use margin loss for training")
args = parser.parse_args()

MODEL_PATH = args.model_path
REFORMED_DATASET_PATH = args.dataset_path
REPS_PATH = args.reps_path
BATCH_SIZE = args.batch_size
EPOCH_NUM = args.epoch
if args.use_margin_loss:
    print("Use margin loss mode: triplet loss will be calculated")

if not os.path.exists(REPS_PATH):
    os.makedirs(REPS_PATH)

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
num_labels = 2

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
#with open(os.path.join(REFORMED_DATASET_PATH, "test.json")) as f:
#    test = json.load(f)

anc_train_texts = [d['anchor_text'] for d in train]
pos_train_texts = [d['positive_text'] for d in train]
neg_train_texts = [d['negative_text'] for d in train]
train_labels = [d['label'] for d in train]
anc_val_texts = [d['anchor_text'] for d in val]
pos_val_texts = [d['positive_text'] for d in val]
neg_val_texts = [d['negative_text'] for d in val]
val_labels = [d['label'] for d in val]
#anc_test_texts = [d['anchor_text'] for d in test]
#pos_test_texts = [d['positive_text'] for d in test]
#neg_test_texts = [d['negative_text'] for d in test]
#test_labels = [d['label'] for d in test]

#Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Encode dataset
anc_train_encodings = tokenizer(anc_train_texts, truncation=True, padding=True)
anc_val_encodings = tokenizer(anc_val_texts, truncation=True, padding=True)
#anc_test_encodings = tokenizer(anc_test_texts, truncation=True, padding=True)

pos_train_encodings = tokenizer(pos_train_texts, truncation=True, padding=True)
pos_val_encodings = tokenizer(pos_val_texts, truncation=True, padding=True)
#pos_test_encodings = tokenizer(pos_test_texts, truncation=True, padding=True)

neg_train_encodings = tokenizer(neg_train_texts, truncation=True, padding=True)
neg_val_encodings = tokenizer(neg_val_texts, truncation=True, padding=True)
#neg_test_encodings = tokenizer(neg_test_texts, truncation=True, padding=True)



#make dataset class
train_dataset = CFIMDbDataset(anc_train_encodings, pos_train_encodings, neg_train_encodings, train_labels)
val_dataset = CFIMDbDataset(anc_val_encodings, pos_val_encodings, neg_val_encodings, val_labels)
#test_dataset = CFIMDbDataset(anc_test_encodings, pos_test_encodings, neg_test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForCounterfactualRobustness.from_pretrained(MODEL_PATH)
model = torch.nn.DataParallel(model)
model.to(device)

train_logits = []
train_attentions = []
train_input_ids = []
for batch in tqdm(train_loader):
    with torch.no_grad():
        anc_input_ids = batch['anchor_input_ids'].to(device)
        anc_attention_mask = batch['anchor_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(anc_input_ids, anc_attention_mask, output_attentions=True)

        logits = outputs[0]
        attentions = outputs[1]
        train_logits.append(logits)
        train_attentions.append(attentions)
        train_input_ids.append(anc_input_ids)

val_logits = []
val_attentions = []
val_input_ids = []
for batch in val_loader:
    with torch.no_grad():
        anc_input_ids = batch['anchor_input_ids'].to(device)
        anc_attention_mask = batch['anchor_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(anc_input_ids, anc_attention_mask, output_attentions=True)

        logits = outputs[0]
        attentions = outputs[1]
        val_logits.append(logits)
        val_attentions.append(attentions)
        val_input_ids.append(anc_input_ids)

torch.save(torch.cat(train_logits, 0), os.path.join(REPS_PATH, "train_logits.pt"))
torch.save(torch.cat(val_logits, 0), os.path.join(REPS_PATH, "val_logits.pt"))
torch.save(torch.cat(train_attentions, 0), os.path.join(REPS_PATH, "train_attentions.pt"))
torch.save(torch.cat(val_attentions, 0), os.path.join(REPS_PATH, "val_attentions.pt"))
torch.save(torch.cat(train_input_ids, 0), os.path.join(REPS_PATH, "train_input_ids.pt"))
torch.save(torch.cat(val_input_ids, 0), os.path.join(REPS_PATH, "val_input_ids.pt"))

