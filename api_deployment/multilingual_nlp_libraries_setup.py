
#!/usr/bin/env python
# coding: utf-8

import spacy
import pickle
import nltk, stanza, gc
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline.multilingual import MultilingualPipeline
from nltk.tokenize import sent_tokenize
import json

# Load parameters from JSON file
with open('path/to/your/parameter.json', 'r') as f:
    param_dict = json.load(f)
spacy_libnames_path = param_dict['spacy_libnames_path']

# Download Spacy language models
prefix = ['ca', 'zh', 'da', 'nl', 'en', 'fr', 'de', 'el', 'it', 'ja', 'lt', 'mk', 'nb', 'pl', 'pt', 'ro', 'ru', 'es']
postfix = ['lg', 'md', 'sm', 'trf']
mid_text = ['_core_news_', '_dep_news_', '_core_web_']
spacy_libraries = []

for pre in prefix:
    for mid in mid_text:
        for post in postfix:
            name = pre + mid + post
            try:
                spacy.load(name)
                spacy_libraries.append(name)
            except OSError:
                try:
                    spacy.cli.download(name)
                    spacy_libraries.append(name)
                except:
                    continue

spacy_libraries = sorted(list(set(spacy_libraries)))

# Download Stanza language models
stanza_lang = "af ar be bg bxr ca cop cs cu da de el en es et eu fa fi fr fro ga gd gl got grc he hi hr hsb hu hy id it ja kk kmr ko la is lt lv lzh mr mt nl nn no olo orv pl pt ro ru sk sl sme sr sv swl ta te tr th ug uk ur vi wo zh-hans zh-hant"
stanza_lang = stanza_lang.split(" ")
stanza.download(lang="multilingual")

for langu in stanza_lang:
    try:
        stanza.download(langu)
    except ValueError:
        print(langu)
        continue

# Download NLTK data
nltk.download('popular')
nltk.download('all')

# Load transformer models
import numpy as np
import pandas as pd
import transformers
import torch

from torch import nn
import os, pickle

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")

model = AutoModel.from_pretrained('xlm-roberta-base')
config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
model = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=config)
config = AutoConfig.from_pretrained("microsoft/infoxlm-base")
model = AutoModel.from_pretrained("microsoft/infoxlm-base", config=config)
config = AutoConfig.from_pretrained("microsoft/xlm-align-base")
model = AutoModel.from_pretrained("microsoft/xlm-align-base", config=config)

sentiment_model_path = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
model = pipeline("sentiment-analysis", config=sentiment_model_path, model=sentiment_model_path, tokenizer=sentiment_model_path, max_length=512, truncation=True)
