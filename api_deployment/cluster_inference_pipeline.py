
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
                          DataCollatorForPermutationLanguageModeling, Trainer,
                          TrainingArguments, pipeline)

# Dummy paths and parameters
PARAMETER_JSON_PATH = "path/to/parameter.json"
SPACY_LIBNAMES_PATH = "path/to/spacy_all_libnames.pkl"
SENTIMENT_MODEL_PATH = "path/to/sentiment_model_path"
SENTIMENT_BOUND_PATH = "path/to/sentiment_bound_path"
LONER_BOUND_PATH = "path/to/loner_bound_path"
STOP_WORDS_PATH = "path/to/stop_words_path"
BASE_CLUSTER_PATH = "path/to/trainbase_clusters_path"

# Load parameters from JSON file
with open(PARAMETER_JSON_PATH, 'r') as f:
    param_dict = json.load(f)

# Set parameters
identification = param_dict["identification"]
sentiment_model_path = os.path.join(SENTIMENT_MODEL_PATH, param_dict['sentiment_model_path'])
sentiment_bound_path = os.path.join(SENTIMENT_BOUND_PATH, str(identification) + '_sentiment_inference_threshold.pkl')
loner_bound_path = os.path.join(LONER_BOUND_PATH, str(identification) + '_clustering_inference_threshold.pkl')
spacy_libnames_path = SPACY_LIBNAMES_PATH
topic_not_take_li = param_dict["topic_donot_predict"]
model1_device = param_dict["model1_device"]
model1_path = os.path.join(BASE_CLUSTER_PATH, param_dict['model1_path'])
encoding_path = os.path.join(BASE_CLUSTER_PATH, param_dict['encoding_path'])
threshold = param_dict['threshold']
model2_device = param_dict["model2_device"]
model3_device = param_dict["model3_device"]
model4_device = param_dict["model4_device"]
device_li = [model1_device, model2_device, model3_device, model4_device]
batch_size = param_dict["batch_size"]
loner_dist_thr = pickle.load(open(loner_bound_path, 'rb'))
neu_lower, neu_upper = pickle.load(open(sentiment_bound_path, 'rb'))
stop_words_path = os.path.join(STOP_WORDS_PATH, param_dict["stop_words_path"])
base_cluster_path = os.path.join(BASE_CLUSTER_PATH, param_dict["trainbase_clusters_path"])
bad_comments = []
sentiment_task = pipeline("sentiment-analysis", config=sentiment_model_path, model=sentiment_model_path,
                          tokenizer=sentiment_model_path, max_length=512, truncation=True)

# Functions and variables for text processing

import pycountry
import stanza
import gc
import torch
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import pickle

def define_language(ori_text, ori_langid, multi_spacydoc):
    """
    Define the language of the input text.

    Args:
        ori_text (str): The original text.
        ori_langid (str): The original language ID.
        multi_spacydoc (spacy.tokens.Doc): The multilingual Spacy document.

    Returns:
        tuple: A tuple containing lists of alpha-2 language codes, language names, and stanza language codes.
    """
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
    return alpha_2li, name_li, stanza_li

# The rest of the code remains the same, with comments added according to Google style guidelines for Python.

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.utils.data
import pandas as pd
import numpy as np
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

model1 = MyBert()
loaded_state_dict = torch.load(model1_path, map_location=model1_device)
model1.load_state_dict(loaded_state_dict)
tokenizer1 = AutoTokenizer.from_pretrained("xlm-roberta-base")

class DatasetTopic(torch.utils.data.Dataset):
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
        pt_batch['input_ids'] = pt_batch['input_ids'].squeeze()
        pt_batch['attention_mask'] = pt_batch['attention_mask'].squeeze()
        return pt_batch

    def __len__(self):
        return len(self.df)

sig_func = nn.Sigmoid().to(model1_device)
model1.to(model1_device).eval()
bad_comments = []

def xlm_sentiment_local(comment):
    # Your implementation here
    pass

def semtiment_analytic_inhouse(comment):
    # Your implementation here
    pass

with open(stop_words_path) as f:
    stop_words_list = f.read().splitlines()
stop_words_list = list(set(stop_words_list))
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
short = ['.', ',', '"', "\'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "'at",
         "_", "`", "\'\'", "--", "``", ".,", "//", ":", "___", '_the', '-', "'em", ".com",
         '\'s', '\'m', '\'re', '\'ll', '\'d', 'n\'t', 'shan\'t', "...", "\'ve", 'u']
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

import torch
import torch.utils.data
import gc
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from transformers import AutoConfig, AutoModel

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

config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
model2 = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=config)
config = AutoConfig.from_pretrained("microsoft/infoxlm-base")
model3 = AutoModel.from_pretrained("microsoft/infoxlm-base", config=config)
config = AutoConfig.from_pretrained("microsoft/xlm-align-base")
model4 = AutoModel.from_pretrained("microsoft/xlm-align-base", config=config)

def extract_features_emb(dataloader_li, model_li, device_li, weight_li):
    all_emb = []
    for j, (dataloader, model, device) in enumerate(zip(dataloader_li, model_li, device_li)):
        torch.cuda.empty_cache()
        gc.collect()
        model_emb = []
        model.to(device)
        model.eval()
        for step, batch in enumerate(dataloader):
            input_batch = {k: v.to(device) for k, v in batch[0].items()}
            lens = batch[1].to(device)
            take_layer_li = []
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                hidden_layers = model(**input_batch, output_hidden_states=True)
            # Process hidden layers for different models
            # ...
            final_output = last_4_layer.cpu().detach().numpy()
            model_emb.append(final_output)
            del final_output
            del last_4_layer
            del hidden_layers
            torch.cuda.empty_cache()
            gc.collect()
        model_emb = np.concatenate(model_emb)
        all_emb.append(model_emb)
        del model
        del model_emb
        gc.collect()
        torch.cuda.empty_cache()
    # Combine embeddings with weights

    return final_emb

def if_bad(nparray):
    if pd.isna(nparray[0]):
        return 0
    else:
        return 1

# Load topic data
base_cluster_path = '/path/to/your/base_cluster.pkl'
comment_topic_di, topic_cluster_di, topic_statistic_df, topic_di = pickle.load(open(base_cluster_path, 'rb'))
topic_di = {}
topic_name_li = topic_statistic_df['topic_name'].tolist()
topic_id_li = topic_statistic_df['topic_id'].tolist()
for name, idx in zip(topic_name_li, topic_id_li):
    topic_di[name] = idx


import json
import os
import pickle
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prefix = '/path/to/your/project/'

with open(prefix + 'parameter.json', 'r') as f:
    param_dict = json.load(f)

identification = param_dict["identification"]
sentiment_model_path = prefix + param_dict['sentiment_model_path']
sentiment_bound_path = prefix + param_dict["sentiment_bound_path"] + str(identification) + '_sentiment_inference_threshold.pkl'
loner_bound_path = prefix + param_dict['loner_bound_path'] + str(identification) + '_clustering_inference_threshold.pkl'
spacy_libnames_path = prefix + param_dict['spacy_libnames_path']
topic_not_take_li = param_dict["topic_donot_predict"]
model1_device = param_dict["model1_device"]
model1_path = prefix + param_dict['model1_path']
encoding_path = prefix + param_dict['encoding_path']
threshold = param_dict['threshold']
model2_device = param_dict["model2_device"]
model3_device = param_dict["model3_device"]
model4_device = param_dict["model4_device"]
device_li = [model1_device, model2_device, model3_device, model4_device]
xlmr_model_path = prefix + param_dict['xlmr_path']
xlmr_tok_path = prefix + param_dict['xlmr_tok']
model2_path = prefix + param_dict['model2_path']
model2_tok = prefix + param_dict['model2_tok']
model3_path = prefix + param_dict['model3_path']
model3_tok = prefix + param_dict['model3_tok']
model4_path = prefix + param_dict['model4_path']
model4_tok = prefix + param_dict['model4_tok']
batch_size = param_dict["batch_size"]
loner_dist_thr = pickle.load(open(loner_bound_path, 'rb'))
neu_lower, neu_upper = pickle.load(open(sentiment_bound_path, 'rb'))
stop_words_path = prefix + param_dict["stop_words_path"]
base_cluster_path = prefix + param_dict["trainbase_clusters_path"]
bad_comments = []

sentiment_task = pipeline("sentiment-analysis",
                          config=sentiment_model_path,
                          model=sentiment_model_path,
                          tokenizer=sentiment_model_path,
                          max_length=512,
                          truncation=True)

import json
import pandas as pd
import numpy as np
import requests
import re
import pycountry
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import nltk
import stanza
from stanza.pipeline.core import Pipeline, MultilingualPipeline
from stanza.models.common.doc import Document
import pickle
import gc
import torch

def xlm_sentiment_local(comment):
    """
    Function to get sentiment score and label using XLM model.

    Args:
    comment (str): The input text for sentiment analysis.

    Returns:
    label_xlm (str): The sentiment label (POS, NEG, NEU).
    score_xlm (float): The sentiment score.
    """
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
    """
    Function to check if a message is null, empty, or NaN.

    Args:
    msg (str): The input message.

    Returns:
    flag (bool): True if the message is null, empty, or NaN, False otherwise.
    cmt (str): The cleaned message.
    take (bool): True if the message is not null, empty, or NaN, False otherwise.
    """
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

def semtiment_analytic_inhouse1(body):
    """
    Function to perform sentiment analysis using in-house API.

    Args:
    body (str): The input text for sentiment analysis.

    Returns:
    tuple: Contains sentiment analysis results from the API and XLM model.
    """
    url = "https://your_api_url_here/api/v1/nlp/sentiment"
    flag_bod, cmt_bod, take_bod = chk_null(body)

    if take_bod:
        comment = cmt_bod
    else:
        comment = ""

    comment = comment.replace("\n", " ")
    comment = comment.replace("\t", " ")
    comment = comment.replace("  ", " ")
    comment = comment.replace("  ", " ")
    comment = comment.replace("  ", " ")

    payload = json.dumps({"text": comment})
    headers = {'Content-Type': 'application/json'}

    label_xlm, score_xlm = xlm_sentiment_local(comment)
    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    if not response.ok:
        return np.nan, np.nan, np.nan, label_xlm, score_xlm, score_xlm, language, comment

    response_json = response.json()
    if response_json['result'] == 'success':
        language = response_json['lang']
        json_sentiment = json.loads(response_json['sentiment'])
        tmpstr = json_sentiment[0]['text']
        api_score = json_sentiment[0]['score']
        avg_score = (api_score + score_xlm) / 2
        return tmpstr, json_sentiment[0]['type'], api_score, label_xlm, score_xlm, avg_score, language, comment
    else:
        return np.nan, np.nan, np.nan, label_xlm, score_xlm, score_xlm, language, comment

requests.packages.urllib3.disable_warnings()

# Add other functions and code here

import nltk
from nltk.tokenize import sent_tokenize
import pickle
import numpy as np

def nltk_cut(trans_text, ori_text=None, name_li=None):
    """
    Tokenize the text using NLTK and return the best number of sentences and the best sentences.
    """
    nltk_len = []
    nltk_textli = []       
    nl_sent = sent_tokenize(trans_text) 
    nl_len = len(nl_sent)
    nltk_len.append(nl_len)
    nltk_textli.append(nl_sent)
    
    if ori_text:
        for name in name_li:
            nl_sent = None
            if name == 'english':
                try:
                    nl_sent = nltk.tokenize.sent_tokenize(ori_text, language=name)
                except LookupError: 
                    pass
            if nl_sent:
                nl_len = len(nl_sent)
                nltk_len.append(nl_len)
                nltk_textli.append(nl_sent)
    
    best_num = max(nltk_len)
    idx = nltk_len.index(best_num)
    best_sent = nltk_textli[idx]
    
    return best_num, best_sent 

def cut_into_sentences(ori_text, trans_text, language_id):
    """
    Cut the text into sentences using different tokenizers and return the best sentences.
    """
    only_en = False
    if ori_text == trans_text:
        only_en = True
        ori_text = clean_text(ori_text)
    else:
        ori_text = clean_text(ori_text)
        trans_text = clean_text(trans_text)
        if ori_text == trans_text:
            only_en = True     
        else:
            spacy_multi2_doc = spacynlp_multi2(ori_text)
    
    if (ori_text == "") or (trans_text == ""):  # bad comment / too short / not actionable
        return []
    
    if not only_en:
        twoword_li, fullname_li, stanza_li = define_language(ori_text, language_id, spacy_multi2_doc)
    
    best_numli = []
    best_sentli = []
    
    if only_en:
        spacy_num, spacy_sent = spacy_cut(ori_text)
    else:
        spacy_num, spacy_sent = spacy_cut(trans_text, ori_text, twoword_li)
    
    best_numli.append(spacy_num)
    best_sentli.append(spacy_sent)
    
    if only_en:
        stanza_num, stanza_sent = stanza_cut(ori_text)
    else:
        stanza_num, stanza_sent = stanza_cut(trans_text, ori_text, stanza_li)
    
    best_numli.append(stanza_num)
    best_sentli.append(stanza_sent)  
    
    if only_en:
        nltk_num, nltk_sent = nltk_cut(ori_text)
    else:
        nltk_num, nltk_sent = nltk_cut(trans_text, ori_text, stanza_li)
    
    best_numli.append(nltk_num)
    best_sentli.append(nltk_sent)     
    
    best_num = max(best_numli)
    idx = best_numli.index(best_num)
    best_sent = best_sentli[idx]    
    
    take_sents = []
    for sent in best_sent:
        sent = str(sent)
        if (len(sent.split()) < 3 and only_en) or (len(set(sent)) < 4) or (len(sent.split()) == 2):  # too short sentences
            continue
        take_sents.append(sent)        
    
    return take_sents

encoding_path = '/path/to/your/encoding.pkl'
encode_reverse = pickle.load(open(encoding_path, 'rb'))
encode_reverse = np.array(list(encode_reverse.values()))  # ensure the order is consistent

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import json
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained('path/to/your/pretrained/model')
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

model1 = MyBert()
loaded_state_dict = torch.load('path/to/your/model1', map_location='model1_device')
model1.load_state_dict(loaded_state_dict)
tokenizer1 = AutoTokenizer.from_pretrained('path/to/your/tokenizer')

class DatasetTopic(torch.utils.data.Dataset):
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

sig_func = nn.Sigmoid().to('model1_device')
model1.to('model1_device').eval()
bad_comments = []

def xlm_sentiment_local(comment):
    # Implement your sentiment analysis function here
    pass

url = "https://your-api-url.com/api/v1/nlp/sentiment"

def sentiment_analytic_inhouse(comment):
    payload = json.dumps({"text": comment})
    headers = {
        'Content-Type': 'application/json'
    }
    label_xlm, score_xlm = xlm_sentiment_local(comment)
    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    if not response.ok:
        bad_comments.append(comment)
        return np.nan, label_xlm, score_xlm
    response_json = response.json()
    if response_json['result'] == 'success':
        language = response_json['lang']
        json_sentiment = json.loads(response_json['sentiment'])
        tmpstr = json_sentiment[0]['text']
        api_score = json_sentiment[0]['score']
        avg_score = (api_score + score_xlm) / 2
        return tmpstr, json_sentiment[0]['type'], label_xlm, avg_score
    else:
        bad_comments.append(comment)
        return np.nan, label_xlm, score_xlm

with open('path/to/your/stop_words') as f:
    stop_words_list = f.read().splitlines()
stop_words_list = list(set(stop_words_list))
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
short = ['.', ',', '"', "\'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "'at",
         "_", "`", "\'\'", "--", "``", ".,", "//", ":", "___", '_the', '-', "'em", ".com",
         '\'s', '\'m', '\'re', '\'ll', '\'d', 'n\'t', 'shan\'t', "...", "\'ve", 'u']
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

import torch
import torch.utils.data
import gc
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import pickle

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

model2 = AutoModel.from_pretrained('model2_path')
model3 = AutoModel.from_pretrained('model3_path')
model4 = AutoModel.from_pretrained('model4_path')

def extract_features_emb(dataloader_li, model_li, device_li, weight_li):
    all_emb = []
    for j, (dataloader, model, device) in enumerate(zip(dataloader_li, model_li, device_li)):
        torch.cuda.empty_cache()
        gc.collect()
        model_emb = []
        model.to(device)
        model.eval()
        for step, batch in enumerate(dataloader):
            input_batch = {k: v.to(device) for k, v in batch[0].items()}
            lens = batch[1].to(device)
            take_layer_li = []
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                hidden_layers = model(**input_batch, output_hidden_states=True)
            # Process hidden layers for different models
            # ...
            final_output = last_4_layer.cpu().detach().numpy()
            model_emb.append(final_output)
            del final_output
            del last_4_layer
            del hidden_layers
            torch.cuda.empty_cache()
            gc.collect()
        model_emb = np.concatenate(model_emb)
        all_emb.append(model_emb)
        del model
        del model_emb
        gc.collect()
        torch.cuda.empty_cache()
    for i, (emb, weight) in enumerate(zip(all_emb, weight_li)):
        if i == 0:
            final_emb = emb * weight
        else:
            final_emb = final_emb + emb * weight
    final_emb = final_emb / sum(weight_li)
    return final_emb

def if_bad(nparray):
    if pd.isna(nparray[0]):
        return 0
    else:
        return 1

topic_cluster_di, topic_statistic_df = pickle.load(open('base_cluster_path', 'rb'))
topic_di = {}
topic_name_li = topic_statistic_df['topic_name'].tolist()
topic_id_li = topic_statistic_df['topic_id'].tolist()
for name, idx in zip(topic_name_li, topic_id_li):
    topic_di[name] = idx

def calc_avg_dist(centroid_vector, eigen_vector, sent_emb, loner_dist_thr):
    centroid_vector = centroid_vector.reshape(1, 768)
    eigen_vector = eigen_vector.reshape(1, 768)
    sent_emb = np.array(sent_emb).reshape(1, 768)
    cen_dist_c = pairwise_distances(sent_emb, centroid_vector, metric='cosine')[0][0]
    cen_dist_e = pairwise_distances(sent_emb, centroid_vector, metric='euclidean')[0][0]
    eig_dist_c = pairwise_distances(sent_emb, eigen_vector, metric='cosine')[0][0]
    eig_dist_e = pairwise_distances(sent_emb, eigen_vector, metric='euclidean')[0][0]
    avg_dist = (cen_dist_c + cen_dist_e + eig_dist_c + eig_dist_e) / 4
    cen_dist = (cen_dist_c + cen_dist_e) / 2
    if avg_dist > loner_dist_thr:
        return loner_dist_thr + 1e-6, cen_dist
    else:
        return avg_dist, cen_dist

def compare_function(topic_li, emb_vec):
    topic_name = topic_li[0]
    topic_key_df = topic_cluster_di[topic_name]
    topic_id = topic_di[topic_name]
    tid = str(topic_id)
    if len(tid) < 2:
        tid = '0' + tid
    topic_key_df[['avg_dist', 'cen_dist']] = topic_key_df.apply(lambda x: calc_avg_dist(x.centroid_vector, x.eigen_vector, emb_vec, loner_dist_thr), axis=1, result_type='expand')
    if topic_key_df['avg_dist'].mean() > loner_dist_thr:
        cid = tid + '-1'
        centroid_dist = np.nan
    else:
        take_df = topic_key_df[topic_key_df['avg_dist'] == topic_key_df['avg_dist'].min()]
        cid = take_df['cluster_id'].iloc[0]
        centroid_dist = take_df['cen_dist'].iloc[0]
    return topic_name, topic_id, cid, centroid_dist

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

def cluster_text_inference_inhouse(input_comment):
    comment_df = pd.DataFrame()
    comment_df['full_comment'] = input_comment

    # Apply sentiment analysis and other preprocessing steps
    comment_df[['translation', 'sentiment_type', 'sentiment_score', 'sentiment_type_xlm', 'sentiment_score_xlm',
                'sentiment_overallscore', 'language', 'ori_comment']] = comment_df.apply(
        lambda x: semtiment_analytic_inhouse1(x.full_comment), axis=1, result_type='expand')

    comment_df['sentiment_overalltype'] = comment_df['sentiment_overallscore'].apply(convert_type)
    comment_df['comment_li'] = comment_df.apply(
        lambda x: cut_into_sentences(x.ori_comment, x.translation, x.language), axis=1)
    comment_df['sent_len'] = comment_df['comment_li'].apply(len)
    comment_df['comment_id'] = comment_df.index.tolist()
    ori_comment_df = comment_df.copy()

    # Process the comments and perform topic modeling
    # ...

    # Save and load models and tokenizers
    model = AutoModel.from_pretrained('xlm-roberta-base')
    model.save_pretrained("/path/to/your/models/xlmr_based_model")
    # Repeat for other models and tokenizers

    # Load and save topic_cluster_di and topic_statistic_df
    _, topic_cluster_di, topic_statistic_df, _ = pickle.load(open("path/to/your/base_cluster.pkl", 'rb'))
    pickle.dump(obj=(topic_cluster_di, topic_statistic_df), file=open("path/to/your/base_cluster.pkl", 'wb'))

    # Return the JSON object
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    json_object = json.dumps(return_text, cls=NpEncoder)
    return json_object
