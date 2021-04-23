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

class CFClassifcationDataset(torch.utils.data.Dataset):
    def __init__(self, anchor_encodings, positive_encodings, negative_encodings, triplet_sample_masks, labels):
        self.anchor_encodings = anchor_encodings
        self.positive_encodings = positive_encodings
        self.negative_encodings = negative_encodings
        self.triplet_sample_masks = triplet_sample_masks
        self.labels = labels

    def __getitem__(self, idx):
        item = dict()
        item.update({'anchor_'+key: torch.tensor(val[idx]) for key, val in self.anchor_encodings.items()})
        item.update({'positive_'+key: torch.tensor(val[idx]) for key, val in self.positive_encodings.items()})
        item.update({'negative_'+key: torch.tensor(val[idx]) for key, val in self.negative_encodings.items()})
        item['triplet_sample_masks'] = torch.tensor(self.triplet_sample_masks[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


