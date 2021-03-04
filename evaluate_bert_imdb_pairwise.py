import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
DATASET_PATH = "dataset/aclImdb"
#REFORMED_DATASET_PATH = "dataset/reform_aclImdb"
#OUTPUT_PATH = "output"
#REFORMED_DATASET_PATH = "dataset/cf_augmented_aclImdb"
REFORMED_DATASET_PATH = "dataset/triplet_augmented_aclImdb"
#REFORMED_DATASET_PATH = "dataset/triplet_1word_augmented_aclImdb"

#REFORMED_DATASET_PATH = "dataset/cf_not_augmented_aclImdb_full"
OUTPUT_PATH = "checkpoints/triplet_1word_augmented_output_scheduling_warmup"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
BATCH_SIZE = 16
EPOCH_NUM = 15
num_labels = 2

#Use triplet margin loss for CF robustness
class BertForCounterfactualRobustness(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        anchor_input_ids=None,
        anchor_attention_mask=None,
        positive_input_ids=None,
        positive_attention_mask=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False

        anchor_outputs = self.bert(
            anchor_input_ids,
            attention_mask=anchor_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        #Sequence Classification
        pooled_output = anchor_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #Sequence Classification Loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #Triplet Margin Loss
        if positive_input_ids is not None and negative_input_ids is not None and labels is not None:
            positive_outputs = self.bert(
                positive_input_ids,
                attention_mask=positive_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            negative_outputs = self.bert(
                negative_input_ids,
                attention_mask=negative_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            triplet_loss = None
            triplet_loss_fct = torch.nn.TripletMarginLoss()
            triplet_loss = triplet_loss_fct(anchor_outputs[1], positive_outputs[1], negative_outputs[1])
            loss = loss + triplet_loss

        if not return_dict:
            output = (logits,) + anchor_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        """
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        """

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

class CFIMDbDataset(torch.utils.data.Dataset):
    def __init__(self, anchor_encodings, positive_encodings, negative_encodings, labels):
        self.anchor_encodings = anchor_encodings
        self.positive_encodings = positive_encodings
        self.negative_encodings = negative_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = dict()
        item.update({'anchor_'+key: torch.tensor(val[idx]) for key, val in self.anchor_encodings.items()})
        item.update({'positive_'+key: torch.tensor(val[idx]) for key, val in self.positive_encodings.items()})
        item.update({'negative_'+key: torch.tensor(val[idx]) for key, val in self.negative_encodings.items()})
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

anc_train_texts = [d['anchor_text'] for d in train]
pos_train_texts = [d['positive_text'] for d in train]
neg_train_texts = [d['negative_text'] for d in train]
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
train_dataset = CFIMDbDataset(anc_train_encodings, pos_train_encodings, neg_train_encodings, train_labels)
val_dataset = CFIMDbDataset(anc_val_encodings, pos_val_encodings, neg_val_encodings, val_labels)
test_dataset = CFIMDbDataset(anc_test_encodings, pos_test_encodings, neg_test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForCounterfactualRobustness.from_pretrained(os.path.join(OUTPUT_PATH, 'epoch_14'))
model = torch.nn.DataParallel(model)
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader) * EPOCH_NUM)
#Train & Evaluation
best_epoch = -1
best_acc = 0
steps = 0

# Test
cor_cnt = 0
total_size = 0
for batch in tqdm(test_loader):
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

