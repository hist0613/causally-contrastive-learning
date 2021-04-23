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
from classes.modeling import *
from classes.datasets import *
import pickle

parser = argparse.ArgumentParser(description='Counterfactual Robustness Training')
parser.add_argument('--dataset-path', 
                    type=str,
                    required=True,
                    help='path for dataset')
parser.add_argument('--output-path', 
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
parser.add_argument('--use-encoding-cache',
                    action='store_true',
                    help="use encoding cache for training")

args = parser.parse_args()


REFORMED_DATASET_PATH = args.dataset_path
OUTPUT_PATH = args.output_path
BATCH_SIZE = args.batch_size
EPOCH_NUM = args.epoch
if args.use_margin_loss:
    print("Use margin loss mode: triplet loss will be calculated")
#DATASET_PATH = "dataset/aclImdb"
#REFORMED_DATASET_PATH = "dataset/reform_aclImdb"
#OUTPUT_PATH = "output"
#REFORMED_DATASET_PATH = "dataset/cf_augmented_aclImdb"
#REFORMED_DATASET_PATH = "dataset/triplet_augmented_aclImdb"
#REFORMED_DATASET_PATH = "dataset/triplet_1word_augmented_2x_aclImdb"

#REFORMED_DATASET_PATH = "dataset/cf_not_augmented_aclImdb_full"
#OUTPUT_PATH = "checkpoints/triplet_1word_augmented_2x_output_scheduling_warmup"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
num_labels = 2

def correct_count(logits, labels):
    _, indices = torch.max(logits, dim=1)
    _, label_indices = torch.max(labels, dim=1)
    correct = torch.sum(indices == label_indices)
    return correct.item()

if args.use_encoding_cache:
    print("Load Stored Cache...")
    if not os.path.exists(".encoding_cache"):
        raise ValueError("There is no encoding cache. please make cache before use tihs option.")

    with open(".encoding_cache/train_dataset.pkl", 'rb') as fp:
        train_dataset = pickle.load(fp)
    with open(".encoding_cache/val_dataset.pkl", 'rb') as fp:
        val_dataset = pickle.load(fp)
    with open(".encoding_cache/test_dataset.pkl", 'rb') as fp:
        test_dataset = pickle.load(fp)
else:
    print("Encode dataset...")
    # Load dataset
    with open(os.path.join(REFORMED_DATASET_PATH, "train.json")) as f:
        train = json.load(f) 
    with open(os.path.join(REFORMED_DATASET_PATH, "valid.json")) as f:
        val = json.load(f)
    with open(os.path.join(REFORMED_DATASET_PATH, "test.json")) as f:
        test = json.load(f)

    anc_train_texts = [d['anchor_text'] for d in train]
    pos_train_texts = [d['positive_text'] for d in train]
    neg_train_texts = [d['negative_text'] for d in train]
    train_triplet_sample_masks = [d['triplet_sample_mask'] for d in train]
    train_labels = [d['label'] for d in train]
    anc_val_texts = [d['anchor_text'] for d in val]
    pos_val_texts = [d['positive_text'] for d in val]
    neg_val_texts = [d['negative_text'] for d in val]
    val_labels = [d['label'] for d in val]
    anc_test_texts = [d['anchor_text'] for d in test]
    pos_test_texts = [d['positive_text'] for d in test]
    neg_test_texts = [d['negative_text'] for d in test]
    test_labels = [d['label'] for d in test]

    #Define tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #Encode dataset
    anc_train_encodings = tokenizer(anc_train_texts, truncation=True, padding=True)
    anc_val_encodings = tokenizer(anc_val_texts, truncation=True, padding=True)
    anc_test_encodings = tokenizer(anc_test_texts, truncation=True, padding=True)

    pos_train_encodings = tokenizer(pos_train_texts, truncation=True, padding=True)
    pos_val_encodings = tokenizer(pos_val_texts, truncation=True, padding=True)
    pos_test_encodings = tokenizer(pos_test_texts, truncation=True, padding=True)

    neg_train_encodings = tokenizer(neg_train_texts, truncation=True, padding=True)
    neg_val_encodings = tokenizer(neg_val_texts, truncation=True, padding=True)
    neg_test_encodings = tokenizer(neg_test_texts, truncation=True, padding=True)



    #make dataset class
    train_dataset = CFClassifcationDataset(anc_train_encodings, pos_train_encodings, neg_train_encodings, train_triplet_sample_masks, train_labels)
    val_dataset = CFIMDbDataset(anc_val_encodings, pos_val_encodings, neg_val_encodings, val_labels)
    test_dataset = CFIMDbDataset(anc_test_encodings, pos_test_encodings, neg_test_encodings, test_labels)

    """
    print("Save encoding cache...")
    if not os.path.exists(".encoding_cache"):
        os.mkdir(".encoding_cache")

    with open(".encoding_cache/train_dataset.pkl", 'wb') as fp:
        pickle.dump(train_dataset, fp)
    with open(".encoding_cache/val_dataset.pkl", 'wb') as fp:
        pickle.dump(val_dataset, fp)
    with open(".encoding_cache/test_dataset.pkl", 'wb') as fp:
        pickle.dump(test_dataset, fp)
    """
    
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased')
#model = BertForCounterfactualRobustnessWithMasker.from_pretrained('bert-base-uncased')
model = torch.nn.DataParallel(model)
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader) * EPOCH_NUM)
#Train & Evaluation
best_epoch = -1
best_acc = 0
steps = 0
all_loss = []
for epoch in range(EPOCH_NUM):
    epoch_loss = []
    model.train()
    train_progress_bar = tqdm(train_loader)

    for batch in train_progress_bar:
        optim.zero_grad()

        anc_input_ids = batch['anchor_input_ids'].to(device)
        anc_attention_mask = batch['anchor_attention_mask'].to(device)

        pos_input_ids = batch['positive_input_ids'].to(device)
        pos_attention_mask = batch['positive_attention_mask'].to(device)

        neg_input_ids = batch['negative_input_ids'].to(device)
        neg_attention_mask = batch['negative_attention_mask'].to(device)

        triplet_sample_masks = batch['triplet_sample_masks'].to(device)

        labels = batch['labels'].to(device)

        """
        ###### MASKER LOSS #####
        uniform_labels = torch.ones(labels.size()).float().to(device)
        uniform_labels = uniform_labels / num_labels
        ssl_labels = anc_input_ids - pos_input_ids
        ssl_labels = ssl_labels + (ssl_labels != 0) * (tokenizer.mask_token_id + 1) - 1
        ssl_labels = ssl_labels.long().to(device)
        ##### MASKER LOSS ENDED #####
        """
        # CrossEntropy Loss
        _, labels = torch.max(labels, dim=1)
        #"""TMP: triplet loss is calculated only when pos/neg gived."""
        if args.use_margin_loss:
            #outputs = model(anc_input_ids, anc_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, labels=labels)
            outputs = model(anc_input_ids, anc_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, triplet_sample_masks=triplet_sample_masks, labels=labels)
        else:
            outputs = model(anc_input_ids, anc_attention_mask, labels=labels)
        loss = outputs[0]

        """
        # Compute Binary Cross Entropy
        outputs = model(input_ids, attention_mask=attention_mask)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(outputs[0].view(-1, num_labels), labels.view(-1, num_labels))
        """

        """TMP: loss summation is need for dataparallel."""
        #loss.backward()
        
        loss.sum().backward()
        epoch_loss.append(loss.sum())
        train_progress_bar.set_description("Current Loss: %f" % loss.sum())
        optim.step()
        scheduler.step()
        steps += 1

    all_loss.append(epoch_loss)
    model.eval()
    cor_cnt = 0
    total_size = 0
    accuracy = 0
    for batch in val_loader:
        with torch.no_grad():
            anc_input_ids = batch['anchor_input_ids'].to(device)
            anc_attention_mask = batch['anchor_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(anc_input_ids, anc_attention_mask)

            logits = outputs[0]
            cor_cnt += correct_count(logits, labels)
            total_size += len(labels)

    if total_size:
        accuracy = cor_cnt * 1.0 / total_size

    if accuracy > best_acc:
        best_epoch = epoch
        best_acc = accuracy
    print(f"Accuracy: {accuracy}")
    model.module.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}"))

with open(os.path.join(OUTPUT_PATH, "training_loss.pkl"), 'wb') as f:
    pickle.dump(all_loss, f)

#print(f"\nBest Model is epoch {best_epoch}. load and evaluate test...")
#model = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_PATH, f'epoch_{epoch}'))
#model.to(device)

# Test
cor_cnt = 0
total_size = 0
for batch in test_loader:
    with torch.no_grad():
        anc_input_ids = batch['anchor_input_ids'].to(device)
        anc_attention_mask = batch['anchor_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(anc_input_ids, anc_attention_mask)

        logits = outputs[0]
        cor_cnt += correct_count(logits, labels)
        total_size += len(labels)

accuracy = cor_cnt * 1.0 / total_size
print(f"Test Accuracy: {accuracy}")

