
import pandas as pd
import numpy as np
import pickle
import random
import transformers
import torch
import functools
import operator
import itertools
import os
import json
import glob
import gc
import math

from torch import nn
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
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error
from collections import Counter
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained('xlm-roberta-base')
        self.multilabel_layers = nn.Sequential(
            nn.Linear(768, 256),
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
        downs_topics = self.multilabel_layers(s1['pooler_output'])

        if output_hidden_states:
            return s1['hidden_states']
        elif output_attentions:
            return s1['attentions']
        elif output_hidden_states and output_attentions:
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sub = self.df.iloc[idx]['subject']
        msg = self.df.iloc[idx]['message']
        text = sub + ' ' + msg if (type(sub) == str) and (len(sub) > 1) else msg

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
        assert len(ans) > 0

        if 'UNK' in ans:
            one_hot = torch.tensor(ori)
        else:
            idx_li = [encode_di[a] for a in ans]
            ori[idx_li] = 1
            assert np.sum(ori) >= 1
            one_hot = torch.tensor(ori)

        return pt_batch, one_hot

    def __len__(self):
        return len(self.df)


def extract_features_emb(dataloader_li, model_li, device_li, weight_li, take_layerli):
    all_emb = []
    for j, (dataloader, model, device, layeridx) in enumerate(zip(dataloader_li, model_li, device_li, take_layerli)):
        model_emb = []
        ground_true = []
        model.to(device)
        model.eval()
        print(j)
        for step, batch in enumerate(dataloader):
            input_batch = {k: v.to(device) for k, v in batch[0].items()}
            with torch.no_grad():
                hidden_layers = model(**input_batch, output_hidden_states=True)
            try:
                hidden_layers = hidden_layers['hidden_states']
                last_4_layer = hidden_layers[layeridx]
            except (TypeError, IndexError, KeyError):
                last_4_layer = hidden_layers[layeridx]

            last_4_layer = torch.mean(last_4_layer, 1)
            final_output = last_4_layer.cpu().detach().numpy()
            model_emb.append(final_output)
            ans_batch = batch[1].cpu().detach().numpy()
            ground_true.append(ans_batch)
            torch.cuda.empty_cache()
            gc.collect()

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
