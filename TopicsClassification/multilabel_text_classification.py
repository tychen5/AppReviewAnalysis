import os
import functools
import operator
import itertools
import random
import pickle
import json
import glob
import math
from collections import Counter

import pandas as pd
import numpy as np
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    mean_absolute_error,
)

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained("xlm-roberta-base")
        self.multilabel_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.Mish(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, len(encode_di)),
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
        s1 = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        downs_topics = self.multilabel_layers(s1["pooler_output"])

        if output_hidden_states:
            return s1["hidden_states"]
        elif output_attentions:
            return s1["attentions"]
        elif output_hidden_states and output_attentions:
            return s1["hidden_states"], s1["attentions"]
        else:
            return downs_topics


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sub = self.df.iloc[idx]["subject"]
        msg = self.df.iloc[idx]["message"]
        text = sub + " " + msg if (type(sub) == str) and (len(sub) > 1) else msg

        pt_batch = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        pt_batch["input_ids"] = pt_batch["input_ids"].squeeze()
        pt_batch["attention_mask"] = pt_batch["attention_mask"].squeeze()

        ori = np.array([0] * len(encode_di))
        ans = self.df.iloc[idx]["topics_clean"]
        assert len(ans) > 0
        if "UNK" in ans:
            one_hot = torch.tensor(ori)
        else:
            idx_li = [encode_di[a] for a in ans]
            ori[idx_li] = 1
            assert np.sum(ori) >= 1
            one_hot = torch.tensor(ori)
        return pt_batch, one_hot

    def __len__(self):
        return len(self.df)


# Replace the following paths with your own paths
data_path = "/path/to/your/data"
model_path = "/path/to/your/model"

encode_di, pos_weight_topic, train_df, test_df = pickle.load(
    open(os.path.join(data_path, "category_datas_20220218_2.pkl"), "rb")
)
print(f"Training#: {len(train_df)}, Testing#: {len(test_df)}")
labeled_df = train_df.append(test_df).sort_index()
labeled_df = labeled_df[
    labeled_df["topics_clean"].apply(lambda x: (len(x) < 2) and ("UNK" not in x))
]

tokenizer1 = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlmr_dataset = Dataset(labeled_df, tokenizer1)
dataloader1 = DataLoader(xlmr_dataset, batch_size=16, num_workers=16, shuffle=False)

# Add more tokenizers and dataloaders as needed


dataloader_li = [dataloader1]
device_li = ["cuda:0"]