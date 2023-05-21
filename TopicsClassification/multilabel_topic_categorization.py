import pandas as pd
import numpy as np
import pickle
import random
import transformers
import torch
import functools, operator
from collections import Counter

# Load data
df_ios = pd.read_pickle("path/to/your/df_ios.pkl")
df_android = pd.read_pickle("path/to/your/df_android.pkl")
df_external = pd.read_pickle("path/to/your/df_external.pkl").reset_index(drop=True)

# Preprocess data
unkid = df_external[df_external['topics_clean'].apply(lambda x: 'UNK' in x)].index.tolist()
unk_takeid = random.sample(unkid, 9)  # Select 9 UNK samples (minimum category count)
unk_df = df_external.loc[unk_takeid].copy()
labeled_df = df_ios.append(df_android).append(df_external)
labeled_df = labeled_df[labeled_df['topics_clean'].apply(lambda x: 'UNK' not in x)]
labeled_df = labeled_df.append(unk_df).reset_index(drop=True)
labeled_df['num'] = labeled_df.topics_clean.apply(len)

# Count topics
all_topics = labeled_df.topics_clean.tolist()
all_topics = list(list(functools.reduce(operator.iconcat, all_topics, [])))
count_di = dict(Counter(all_topics))
count_di = {k: v for k, v in sorted(count_di.items(), key=lambda item: item[1], reverse=False)}

# Split data into train and test sets
test_df = test_df.sort_index()
test_idx = test_df.index.tolist()
train_df = labeled_df[~labeled_df.index.isin(test_idx)]

# Save train and test data
train_df.to_pickle("path/to/your/categorized_traindf.pkl")
test_df.to_pickle("path/to/your/categorized_testdf.pkl")

# Import necessary libraries
import os
import gc
from torch import nn
from torchvision import models
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling
from tqdm import tqdm
from transformers import Trainer
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error
from sklearn.metrics import accuracy_score

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Count topics in train data
all_topics = labeled_df.topics_clean.tolist()
all_topics = list(list(functools.reduce(operator.iconcat, all_topics, [])))
count_di = dict(Counter(all_topics))
count_di = {k: v for k, v in sorted(count_di.items(), key=lambda item: item[1], reverse=True)}

# Encode topics
i = 0
encode_di = {}
for k, v in count_di.items():
    if k == 'UNK':
        encode_di[k] = i
        i += 1

# Calculate positive weights for topics
pos_weight_topic = []
train_topics = train_df['topics_clean'].tolist()
train_topics = list(list(functools.reduce(operator.iconcat, train_topics, [])))
total = len(train_df)
train_topics = Counter(train_topics)
for key in encode_di.keys():
    pos_num = train_topics[key]
    pos_weight_topic.append((total - pos_num) / pos_num)  # Number of 0s / number of 1s for each category
pos_weight_topic = torch.tensor(pos_weight_topic)

# Load data and model
encode_di, pos_weight_topic, train_df, test_df = pickle.load(open('path/to/your/category_datas.pkl', 'rb'))

# Define custom model
class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained('xlm-roberta-base')
        self.multilabel_layers = nn.Sequential(nn.Linear(768, 256),
                                               nn.Mish(),
                                               nn.BatchNorm1d(256),
                                               nn.Dropout(0.1),
                                               nn.Linear(256, 64),
                                               nn.Mish(),
                                               nn.BatchNorm1d(64),
                                               nn.Dropout(0.1),
                                               nn.Linear(64, len(encode_di))
                                               )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        s1 = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             position_ids=position_ids, head_mask=head_mask,
                             inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values,
                             use_cache=use_cache, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict)
        downs_topics = self.multilabel_layers(s1['pooler_output'])
        if output_hidden_states == True:
            return s1['hidden_states']
        elif output_attentions == True:
            return s1['attentions']
        elif (output_hidden_states == True) and (output_attentions == True):
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics

# Rest of the code for training and evaluation
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

models = [model1] + [
    AutoModel.from_pretrained(path, config=AutoConfig.from_pretrained(path))
    for path in model_paths[1:]
]

# Set device and weights
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
                last_4_layer = sum(hidden_layers[-5:])
            except TypeError:
                hidden_layers = hidden_layers['hidden_states']
                last_4_layer = sum(hidden_layers[-5:])
            last_4_layer = torch.mean(last_4_layer, 1)
            final_output = last_4_layer.cpu().detach().numpy()
            model_emb.append(final_output)
            ans_batch = batch[1].cpu().detach().numpy()
            ground_true.append(ans_batch)
        model_emb = np.concatenate(model_emb)
        all_emb.append(model_emb)

    final_emb = sum(emb * weight for emb, weight in zip(all_emb, weight_li)) / sum(weight_li)
    ground_true = np.concatenate(ground_true)
    return final_emb, ground_true

embeddings, ans = extract_features_emb(dataloaders, models, device_li, weight_li)
embeddings.shape, ans.shape