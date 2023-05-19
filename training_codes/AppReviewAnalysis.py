
import functools
import operator
import ast
import pandas as pd
import numpy as np
from transformers import pipeline
import gc
import requests
import json
import random
import time
import pickle
import os
import re
import pycountry
import nltk
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline.multilingual import MultilingualPipeline
from nltk.tokenize import sent_tokenize
from nltk.stem import SnowballStemmer
import torch

# Load data
df_ori = pd.read_csv("path/to/your/review_export_report-Unifi-Network.csv")
df_new1 = pd.read_csv("path/to/your/review_export_report_Feb9_to_May10.csv")

# Preprocess data
df_external = pd.concat([df_ori, df_new1])
df_external = df_external.reset_index(drop=True)

# Remove duplicate reviews
repeat_review = pd.DataFrame(df_external['Reply URL'].value_counts())
repeat_review = repeat_review[repeat_review['Reply URL'] > 1]
remove_reviews = repeat_review.index.tolist()
for url in remove_reviews:
    tmpdf = df_external[df_external['Reply URL'] == url]
    duplicate_idx = tmpdf.index[1:]
    df_external = df_external.drop(duplicate_idx)
df_external = df_external.reset_index(drop=True)

# Clean external data
def clean_externaldata(strli):
    final_li = ast.literal_eval(strli)
    if len(final_li) < 1:
        return ['UNK']
    else:
        return final_li

df_external["topics_clean"] = df_external.Topics.apply(clean_externaldata)

# Rename columns and filter data
df_external.columns = ['App Name', 'App Store', 'App', 'Store', 'App ID', 'Review ID',
                       'country', 'version', 'rating', 'published_at', 'author', 'subject', 'message',
                       'Translated Subject', 'Translated Body', 'sentiment', 'device',
                       'Language', 'OS Version', 'Reply URL', 'topics', 'Custom Topics', 'Tags',
                       'topics_clean']
df_external = df_external[['App Name', 'App Store', 'App', 'Store', 'App ID', 'Review ID',
                           'author', 'rating', 'version', 'published_at', 'subject', 'message', 'country', 'topics', 'device', 'topics_clean', 'sentiment',
                           'Translated Subject', 'Translated Body', 'Language', 'Reply URL']]
take_apps = ['UniFi Network', 'UniFi Protect', 'UniFi Access', 'AmpliFi WiFi', 'AmpliFi Teleport', 'UISP Mobile (UNMS)', 'UNMS']
df_external = df_external[df_external['App Name'].isin(take_apps)]

# Clean comments
bad_list = ['What can we do better?', '_CENSORED_EMAIL_', 'none',
            'No', 'no', 'na', 'test', '', 'yes', 'Yes', 'nothing', '...', 'hh']
substring_list = ['_CENSORED_EMAIL_', 'Aija Kra', 'Aija.Kra', 'aijaKra', 'aijakra', 'aija.kra', 'AijaKra', 'aija kra', 'aija_kra', 'Aija_Kra']

def clean_comments(ori_comments):
    if ori_comments in bad_list:
        return 0
    try:
        if '@ui.com' in ori_comments:
            return 0
    except TypeError:
        return 0
    if any(map(ori_comments.__contains__, substring_list)):
        return 0
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", ori_comments, re.I)
    try:
        email = email.group(1)
        ori_comments = ori_comments.replace(email, '')
    except AttributeError:
        pass
    o_c = ori_comments.replace('\n', '')
    o_c = o_c.replace('Test', '')
    o_c = o_c.replace('test', '')
    o_c = o_c.replace('TEST', '')
    o_c = o_c.replace('hi', '')
    o_c = o_c.replace('Hi', '')
    o_c = o_c.replace('HI', '')
    o_c = o_c.replace(' ', '')
    if len(o_c) < 2:
        return 0
    else:
        return 1

df_external['take'] = df_external['message'].apply(clean_comments)
df_external = df_external[df_external['take'] == 1]

# Sentiment analysis
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", config=model_path, model=model_path, tokenizer=model_path, max_length=512, truncation=True)

def xlm_sentiment_local(comment):
    tmp = sentiment_task(comment, return_all_scores=True)
    tmp = tmp[0]
    tmp = pd.DataFrame(tmp)
    score_li = tmp['score'].tolist()
    score_xlm = max(score_li)
    label_xlm = tmp[tmp['score'] == score_xlm]['label'].values[0]
    if label_xlm == "Neutral":
        score_xlm = score_li[0] * -1 + score_li[-1]
        label_xlm = "NEU"
    elif label_xlm == "Negative":
        score_xlm = score_li[0] * -1
        label_xlm = "NEG"
    else:
        score_xlm = score_li[-1]
        label_xlm = "POS"
    return label_xlm, score_xlm

def chk_null(msg):
    take = True
    try:
        flag = pd.isnull(msg)
        take = not flag
    except Exception as e:
        try:
            flag = pd.isna(msg)
            take = not flag
        except Exception as e:
            flag = np.isnan(msg)
            take = not flag
    cmt = str(msg)
    if len(cmt) <= 3:
        cmt = ""
        take = False
    return flag, cmt, take

# Add your sentiment analysis function here
# Example: def semtiment_analytic_inhouse(subject, body):

# Load and preprocess data
df_external = pickle.load(open("path/to/your/chkpoint_network_sentiment_tranlation.pkl", 'rb'))

# Load Spacy libraries
spacy_libraries = pickle.load(open('path/to/your/spacy_all_libnames.pkl', 'rb'))
prefix = ['ca', 'zh', 'da', 'nl', 'en', 'fr', 'de', 'el', 'it', 'ja', 'lt', 'mk', 'nb', 'pl', 'pt', 'ro', 'ru', 'es']


import pycountry
import stanza
import text_classification as tc
from nltk.tokenize import sent_tokenize
import spacy
import gc
import torch
from stanza.pipeline.core import Pipeline, Document


def define_language(ori_text, ori_langid, multi_spacydoc):
    alpha_2li = [ori_langid]
    try:
        name_li = [pycountry.languages.get(alpha_2=ori_langid).name.lower()]
    except AttributeError:
        name_li = []
    stanza_li = [ori_langid]
    nlp = Pipeline(lang="multilingual", processors="langid", use_gpu=False, verbose=False)
    docs = [ori_text]
    docs = [Document([], text=text) for text in docs]
    nlp(docs)
    lang_id = docs[0].lang
    stanza_li.append(lang_id)
    if len(lang_id) == 3:
        try:
            name = pycountry.languages.get(alpha_3=lang_id).name.lower()
            two_word = lang_id
            name_li.append(name)
            alpha_2li.append(two_word)
        except AttributeError:
            two_word = lang_id[:2]
            try:
                name_li.append(pycountry.languages.get(alpha_2=two_word).name.lower())
                alpha_2li.append(two_word)
            except AttributeError:
                pass
    else:
        two_word = lang_id[:2]
        alpha_2li.append(two_word)
        try:
            name = pycountry.languages.get(alpha_2=two_word).name.lower()
            name_li.append(name)
        except AttributeError:
            pass
    two_word = multi_spacydoc._.language['language']
    alpha_2li.append(two_word)
    stanza_li.append(two_word)
    try:
        name_li.append(pycountry.languages.get(alpha_2=two_word).name.lower())
    except AttributeError:
        pass
    guess = tc.guess_language(ori_text)
    try:
        two_word = pycountry.languages.get(alpha_3=guess).alpha_2
    except AttributeError:
        two_word = guess
    alpha_2li.append(two_word)
    stanza_li.append(two_word)
    try:
        name_li.append(pycountry.languages.get(alpha_3=guess).name.lower())
    except AttributeError:
        try:
            name_li.append(pycountry.languages.get(alpha_2=two_word).name.lower())
        except AttributeError:
            pass
    return alpha_2li, name_li, stanza_li  # two-char country code, full language name, stanza special name


# Other functions (spacy_cut, stanza_cut, nltk_cut, cut_into_sentences) go here

# Example usage
# df_external['comment_li'] = df_external.apply(lambda x: cut_into_sentences(x.ori_comment, x.translation, x.language), axis=1)

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


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
                return_dict=None,
                ):
        s1 = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             position_ids=position_ids, head_mask=head_mask,
                             inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values,
                             use_cache=use_cache, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict)
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


# Load model and tokenizer
category_model = MyBert()
loaded_state_dict = torch.load(model_path, map_location=device)
category_model.load_state_dict(loaded_state_dict)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Prepare dataset and dataloader
df_comments = pd.DataFrame()  # Replace with your DataFrame containing comments
xlmr_dataset = Dataset(df_comments, tokenizer)
dataloader = DataLoader(
    xlmr_dataset, batch_size=64, num_workers=int(os.cpu_count()), shuffle=False
)

# Perform inference
category_model.to(device).eval()
sig_func = nn.Sigmoid().to(device)

# Replace the following code with your desired inference logic

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
                return_dict=None,
                ):
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


# Replace the path with your own path to the stop_words_english.txt file
with open('/path/to/your/stop_words_english.txt') as f:
    stop_words_list = f.read().splitlines()

# Preprocessing functions and variables
stop_words_list = list(set(stop_words_list))
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
short = ['.', ',', '"', "\'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "'at",
         "_", "`", "\'\'", "--", "``", ".,", "//", ":", "___", '_the', '-', "'em", ".com",
         '\'s', '\'m', '\'re', '\'ll', '\'d', 'n\'t', 'shan\'t', "...", "\'ve", 'u']
import string
punc = string.punctuation
punc = [w for w in punc]
stop_words_list.extend(punc)
stop_words_list.extend(short)
stop_words.update(stop_words_list)


def preprocess(texts):
    tokens = [i for i in word_tokenize(texts.lower()) if i not in stop_words]
    token_result = ''
    for i, token in enumerate(tokens):
        token_result += ps.stem(token) + ' '
    token_result = ''.join([i for i in token_result if not i.isdigit()])
    sent = token_result
    sent = sent.replace("  ", " ")
    sent = sent.replace("  ", " ")
    sent = sent.replace("  ", " ")
    if len(sent.split()) < 1:
        return ""
    try:
        if sent[0] == " ":
            sent = sent[1:]
        if sent[-1] == " ":
            sent = sent[:-1]
    except IndexError:
        return ""
    return sent


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


# Replace the following paths with your own paths
model_path = "/path/to/your/model"
output_path = "/path/to/your/output"

# Tokenizers and Dataloaders
tokenizer1 = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlmr_dataset = Dataset(labeled_df, tokenizer1)
dataloader1 = DataLoader(
    xlmr_dataset, batch_size=32, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
xlmt_dataset = Dataset(labeled_df, tokenizer2)
dataloader2 = DataLoader(
    xlmt_dataset, batch_size=16, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer3 = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
infoxlm_datset = Dataset(labeled_df, tokenizer3)
dataloader3 = DataLoader(
    infoxlm_datset, batch_size=16, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer4 = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
xlmalign_datset = Dataset(labeled_df, tokenizer4)
dataloader4 = DataLoader(
    infoxlm_datset, batch_size=16, num_workers=int(os.cpu_count()), shuffle=False
)

dataloader_li = [dataloader1, dataloader2, dataloader3, dataloader4]
device_li = ['cuda:0', 'cuda:1', 'cuda:1', 'cuda:1']

model1 = MyBert()
loaded_state_dict = torch.load(model_path, map_location=device_li[0])
model1.load_state_dict(loaded_state_dict)

config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
model2 = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=config)

config = AutoConfig.from_pretrained("microsoft/infoxlm-base")
model3 = AutoModel.from_pretrained("microsoft/infoxlm-base", config=config)

config = AutoConfig.from_pretrained("microsoft/xlm-align-base")
model4 = AutoModel.from_pretrained("microsoft/xlm-align-base", config=config)

model_li = [model1, model2, model3, model4]
weight_li = [4, 0.5, 1, 1.5]

# Function to extract features and embeddings


# Save the processed DataFrame
pickle.dump(obj=labeled_df, file=open(output_path, 'wb'))
