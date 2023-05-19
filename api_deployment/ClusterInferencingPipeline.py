
import functools
import itertools
import json
import math
import operator
import os
import pickle
import random
import re
import requests
import string
import time
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import pycountry
import spacy
import stanza
import torch
import transformers
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline.multilingual import MultilingualPipeline
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          DataCollatorForPermutationLanguageModeling, EarlyStoppingCallback,
                          Trainer, TrainingArguments, pipeline)

# Dummy paths and values
DUMMY_PATH = "/path/to/your/dummy"
DUMMY_VALUE = "enter_your_dummy_value"

# Replace the following paths and values with your actual paths and values
param_dict = {
    "identification": DUMMY_VALUE,
    "sentiment_model_path": DUMMY_PATH,
    "sentiment_bound_path": DUMMY_PATH,
    "loner_bound_path": DUMMY_PATH,
    "spacy_libnames_path": DUMMY_PATH,
    "topic_donot_predict": DUMMY_VALUE,
    "model1_device": DUMMY_VALUE,
    "model1_path": DUMMY_PATH,
    "encoding_path": DUMMY_PATH,
    "threshold": DUMMY_VALUE,
    "model2_device": DUMMY_VALUE,
    "model3_device": DUMMY_VALUE,
    "model4_device": DUMMY_VALUE,
    "batch_size": DUMMY_VALUE,
    "stop_words_path": DUMMY_PATH,
    "trainbase_clusters_path": DUMMY_PATH,
}


import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import json
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

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
                                               nn.Linear(64, len(encode_reverse))
                                           )

    def forward(self,
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
                return_dict=None):
        s1 = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                             inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values,
                             use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        downs_topics = self.multilabel_layers(s1['pooler_output'])
        if output_hidden_states:
            return s1['hidden_states']
        elif output_attentions:
            return s1['attentions']
        elif output_hidden_states and output_attentions:
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics

# Replace the following with your own paths and variables
model1_path = "path/to/your/model"
model1_device = "cuda" if torch.cuda.is_available() else "cpu"
df_comments = pd.read_csv("path/to/your/data.csv")
batch_size = 16
threshold = 0.5
topic_not_take_li = ["topic1", "topic2"]
encode_reverse = np.array(["topic1", "topic2", "topic3"])

model1 = MyBert()
loaded_state_dict = torch.load(model1_path, map_location=model1_device)
model1.load_state_dict(loaded_state_dict)
tokenizer1 = AutoTokenizer.from_pretrained("xlm-roberta-base")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['comment_li']
        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tmp = pt_batch['input_ids'].clone()
        pt_batch['input_ids'] = tmp.squeeze()
        tmp = pt_batch['attention_mask'].clone()
        pt_batch['attention_mask'] = tmp.squeeze()
        return pt_batch

    def __len__(self):
        return len(self.df)

dataset1 = Dataset(df_comments, tokenizer1)
dataloader1 = DataLoader(
    dataset1, batch_size=batch_size, num_workers=int(os.cpu_count()), shuffle=False
)


import torch
import numpy as np
import pandas as pd
import gc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics.pairwise import pairwise_distances


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['sent_translation']
        text = preprocess(text)
        text_len = self.tokenizer(text, truncation=True, max_length=512)
        text_len = sum(text_len['attention_mask'])
        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tmp = pt_batch['input_ids'].clone()
        pt_batch['input_ids'] = tmp.squeeze()
        tmp = pt_batch['attention_mask'].clone()
        pt_batch['attention_mask'] = tmp.squeeze()
        return pt_batch, torch.tensor(text_len)

    def __len__(self):
        return len(self.df)


# Replace the following paths and model names with your own
labeled_df_path = "/path/to/your/labeled_df"
base_cluster_path = "/path/to/your/base_cluster"
model1_path = "/path/to/your/model1"

labeled_df = pd.read_csv(labeled_df_path)

tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
xlmr_dataset = Dataset(labeled_df, tokenizer1)
batch_size = 16
dataloader1 = DataLoader(
    xlmr_dataset, batch_size=batch_size, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
xlmt_dataset = Dataset(labeled_df, tokenizer2)
dataloader2 = DataLoader(
    xlmt_dataset, batch_size=batch_size, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer3 = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
infoxlm_dataset = Dataset(labeled_df, tokenizer3)
dataloader3 = DataLoader(
    infoxlm_dataset, batch_size=batch_size, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer4 = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
xlmalign_dataset = Dataset(labeled_df, tokenizer4)
dataloader4 = DataLoader(
    xlmalign_dataset, batch_size=batch_size, num_workers=int(os.cpu_count()), shuffle=False
)

dataloader_li = [dataloader1, dataloader2, dataloader3, dataloader4]

config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
model2 = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=config)

config = AutoConfig.from_pretrained("microsoft/infoxlm-base")
model3 = AutoModel.from_pretrained("microsoft/infoxlm-base", config=config)

config = AutoConfig.from_pretrained("microsoft/xlm-align-base")
model4 = AutoModel.from_pretrained("microsoft/xlm-align-base", config=config)

model_li = [model1, model2, model3, model4]
weight_li = [4, 0.5, 1, 1.5]


import json
import numpy as np
import requests
from transformers import pipeline


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        """Encode numpy objects as JSON serializable objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Save the sentiment analysis model
sentiment_model_path = "path/to/your/sentiment_model"
sentiment_task = pipeline(
    "sentiment-analysis",
    config=sentiment_model_path,
    model=sentiment_model_path,
    tokenizer=sentiment_model_path,
    max_length=512,
    truncation=True,
)
sentiment_task.save_pretrained("path/to/your/saved_model")

# Load the sentiment analysis model
sentiment_model_path = "path/to/your/saved_model"
sentiment_task = pipeline(
    "sentiment-analysis",
    config=sentiment_model_path,
    model=sentiment_model_path,
    tokenizer=sentiment_model_path,
    max_length=512,
    truncation=True,
)

# Send a request to the API
url = "http://your_api_url/api/v1/nlp/cluster/appreview"
payload = json.dumps(
    {
        "text": [
            "great application!",
            "i love you UI!!!!!",
            "Hasta los momentos la he usado y me parece muy esencial... El único detalle es que no logro colocarla en español para hacerla mucho más practica debo ir a mi ordenador y hacer desde ahí las configuraciones y rastreros... Si tiene la opción de español seria de gran ayuda para muchos... O no la consigoe podrían ayudar.....Hasta los momentos la he usado y me parece muy esencial... El único detalle es que no logro colocarla en español para hacerla mucho más practica debo ir a mi ordenador y hacer desde ahí las configuraciones y rastreros. Si tiene la opción de español seria de gran ayuda para muchos. O no la consigoe podrían ayudar.",
            "Hasta los momentos la he usado y me parece muy esencial... El único detalle es que no logro colocarla en español para hacerla mucho más practica debo ir a mi ordenador y hacer desde ahí las configuraciones y rastreros.",
            "broken ui",
            "its terrible, I cant use this shit",
            "hate this",
        ]
        * 19
    }
)
headers = {"Content-Type": "application/json"}
response = requests.request("POST", url, headers=headers, data=payload)
