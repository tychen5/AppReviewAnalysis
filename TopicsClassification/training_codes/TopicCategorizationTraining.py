import pandas as pd
import numpy as np
import pickle
import random
import transformers
import torch
import functools
import operator
from collections import Counter
from torch import nn
from torchvision import models
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForPermutationLanguageModeling
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error
from sklearn.metrics import accuracy_score
import os

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load data
df_ios = pd.read_pickle("path/to/df_ios.pkl")
df_android = pd.read_pickle("path/to/df_android.pkl")
df_external = pd.read_pickle("path/to/df_external.pkl").reset_index(drop=True)

# Preprocess data


# Define the custom BERT model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Define Dataset class


# Create train and test datasets
train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=16)

# Define optimizer and learning rate scheduler
# ... (omitted for brevity)

# Define loss function

# Set device
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Train and evaluate the model

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sub = self.df.iloc[idx]['subject']
        msg = self.df.iloc[idx]['message']
        text = sub + ' ' + msg if type(sub) == str else msg

        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        pt_batch['input_ids'] = pt_batch['input_ids'].squeeze()
        pt_batch['attention_mask'] = pt_batch['attention_mask'].squeeze()

        ori = np.array([0] * len(encode_di))
        ans = self.df.iloc[idx]['topics_clean']
        idx_li = [encode_di[a] for a in ans]
        ori[idx_li] = 1
        one_hot = torch.tensor(ori)

        return pt_batch, one_hot

    def __len__(self):
        return len(self.df)

# Replace the following paths and model names with your own
tokenizer_paths = [
    "xlm-roberta-base",
    "cardiffnlp/twitter-xlm-roberta-base",
    "microsoft/infoxlm-base",
    "microsoft/xlm-align-base"
]

model_paths = [
    "path/to/your/model1",
    "cardiffnlp/twitter-xlm-roberta-base",
    "microsoft/infoxlm-base",
    "microsoft/xlm-align-base"
]

# Initialize tokenizers and datasets
tokenizers = [AutoTokenizer.from_pretrained(path) for path in tokenizer_paths]
datasets = [CustomDataset(labeled_df, tokenizer) for tokenizer in tokenizers]

# Initialize dataloaders
dataloaders = [
    DataLoader(dataset, batch_size=32, num_workers=16, shuffle=False)
    for dataset in datasets
]

# Load models
model1 = mybert()
loaded_state_dict = torch.load("path/to/your/weights.pt")
model1.load_state_dict(loaded_state_dict)

models = [
    model1,
    AutoModel.from_pretrained(model_paths[1], config=AutoConfig.from_pretrained(model_paths[1])),
    AutoModel.from_pretrained(model_paths[2], config=AutoConfig.from_pretrained(model_paths[2])),
    AutoModel.from_pretrained(model_paths[3], config=AutoConfig.from_pretrained(model_paths[3]))
]

# Set device list and weights
device_li = ['cuda:1', 'cuda:0', 'cuda:1', 'cuda:0']
weight_li = [2, 1, 1, 1]

def extract_features_emb(dataloader_li, model_li, device_li, weight_li):
    all_emb = []
    for j, (dataloader, model, device) in enumerate(zip(dataloader_li, model_li, device_li)):
        model_emb = []
        ground_true = []
        model.to(device)
        model.eval()
        for step, batch in enumerate(dataloader):
            input_batch = {k: v.to(device) for k, v in batch[0].items()}
            with torch.no_grad():
                hidden_layers = model(**input_batch, output_hidden_states=True)
            try:
                last_4_layer = hidden_layers[-1] + hidden_layers[-2] + hidden_layers[-3] + hidden_layers[-4] + hidden_layers[-5]
            except TypeError:
                hidden_layers = hidden_layers['hidden_states']
                last_4_layer = hidden_layers[-1] + hidden_layers[-2] + hidden_layers[-3] + hidden_layers[-4] + hidden_layers[-5]
            last_4_layer = torch.mean(last_4_layer, 1)
            final_output = last_4_layer.cpu().detach().numpy()
            model_emb.append(final_output)
            ans_batch = batch[1].cpu().detach().numpy()
            ground_true.append(ans_batch)
        model_emb = np.concatenate(model_emb)
        all_emb.append(model_emb)
    for i, (emb, weight) in enumerate(zip(all_emb, weight_li)):
        if i == 0:
            final_emb = emb * weight
        else:
            final_emb = final_emb + emb * weight
    final_emb = final_emb / sum(weight_li)
    ground_true = np.concatenate(ground_true)
    return final_emb, ground_true

embeddings, ans = extract_features_emb(dataloaders, models, device_li, weight_li)
embeddings.shape, ans.shape
