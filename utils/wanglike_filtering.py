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
        return len(self.encodings)

DATASET_PATH = "../dataset/SST-2/triplet_automated_averaged_gradient_1word_augmented_1x_sst2_try2"
OUTPUT_PATH  ="../dataset/SST-2/triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_sst2_try2"
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

causal_reps = []
for batch in tqdm(causal_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)[1]
        causal_reps.append(logits.detach().cpu())

causal_reps = torch.cat(causal_reps)


