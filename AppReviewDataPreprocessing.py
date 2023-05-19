
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

# Load data from CSV files
df_ori = pd.read_csv("path/to/your/review_export_report-Unifi-Network.csv")
df_new1 = pd.read_csv("path/to/your/review_export_report_Feb9_to_May10.csv")

# Combine dataframes
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


def clean_externaldata(strli):
    """Clean external data by converting string to list and handling empty lists."""
    final_li = ast.literal_eval(strli)
    if len(final_li) < 1:
        return ['UNK']
    else:
        return final_li


# Clean topics and rename columns
df_external["topics_clean"] = df_external.Topics.apply(clean_externaldata)
df_external.columns = ['App Name', 'App Store', 'App', 'Store', 'App ID', 'Review ID',
                       'country', 'version', 'rating', 'published_at', 'author', 'subject', 'message',
                       'Translated Subject', 'Translated Body', 'sentiment', 'device',
                       'Language', 'OS Version', 'Reply URL', 'topics', 'Custom Topics', 'Tags',
                       'topics_clean']

# Filter columns and apps
df_external = df_external[['App Name', 'App Store', 'App', 'Store', 'App ID', 'Review ID',
                           'author', 'rating', 'version', 'published_at', 'subject', 'message', 'country', 'topics', 'device', 'topics_clean', 'sentiment',
                           'Translated Subject', 'Translated Body', 'Language', 'Reply URL']]
take_apps = ['UniFi Network', 'UniFi Protect', 'UniFi Access', 'AmpliFi WiFi', 'AmpliFi Teleport', 'UISP Mobile (UNMS)', 'UNMS']
df_external = df_external[df_external['App Name'].isin(take_apps)]

# Count unique authors for a specific app
appname = 'Network'
len(df_external[df_external['App Name'].str.contains(appname)]['author'].unique())

import re

bad_list = ['What can we do better?', '_CENSORED_EMAIL_', 'none',
            'No', 'no', 'na', 'test', '', 'yes', 'Yes', 'nothing', '...', 'hh']
substring_list = ['_CENSORED_EMAIL_', 'Aija Kra', 'Aija.Kra', 'aijaKra', 'aijakra', 'aija.kra', 'AijaKra', 'aija kra', 'aija_kra', 'Aija_Kra']


def clean_comments(ori_comments):
    """Clean comments by removing bad comments and unnecessary characters."""
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


# Clean comments and filter dataframe
df_external['take'] = df_external['message'].apply(clean_comments)
df_external = df_external[df_external['take'] == 1]

# Add your sentiment analysis and text processing functions here
# ...

# Save and load dataframe checkpoint
pickle.dump(obj=df_external, file=open("path/to/your/chkpoint_network_sentiment_tranlation.pkl", 'wb'))
df_external = pickle.load(open("path/to/your/chkpoint_network_sentiment_tranlation.pkl", 'rb'))

# Load additional libraries and models for text processing
# ...

# Add your text cleaning and language detection functions here

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


def spacy_cut(trans_text, ori_text=None, two_wordli=None):
    spacy_len = []
    spacy_textli = []
    for smodel in spacy_basicmodels:
        doc = smodel(trans_text)
        sents = list(doc.sents)
        length = len(sents)
        spacy_len.append(length)
        spacy_textli.append(sents)
    take_module = []
    if ori_text:  # 有其他語言
        for lang in two_wordli:
            if lang != 'en':
                for module in spacy_libraries:
                    if module.startswith(lang):
                        take_module.append(module)
                if len(take_module) >= 1:
                    pass
        for module in take_module:
            smodel = spacy.load(module)
            smodel.add_pipe('sentencizer')
            doc = smodel(ori_text)
            sents = list(doc.sents)
            length = len(sents)
            spacy_len.append(length)
            spacy_textli.append(sents)
    best_num = max(spacy_len)
    idx = spacy_len.index(best_num)
    best_sent = spacy_textli[idx]
    return best_num, best_sent


def stanza_cut(trans_text, ori_text=None, stanza_li=None):
    stanza_len = []
    stanza_textli = []
    nlp = Pipeline(lang='en', processors="tokenize", use_gpu=False, verbose=False)
    st_doc = nlp(trans_text)
    st_doc_s = st_doc.sentences
    st_len = len(st_doc_s)
    sents = []
    for s in st_doc_s:
        sents.append(s.text)
    stanza_len.append(st_len)
    stanza_textli.append(sents)
    if ori_text:
        for i, lang_id in enumerate(stanza_li):
            if lang_id == 'en':
                pass
            nlp = None
            gc.collect()
            torch.cuda.empty_cache()
            try:
                nlp = Pipeline(lang=lang_id, processors="tokenize", use_gpu=False, verbose=False)
            except KeyError:
                if i < len(stanza_li) - 1:
                    pass
                else:
                    try:
                        nlp = MultilingualPipeline(max_cache_size=1, ld_batch_size=1, use_gpu=False)
                    except ValueError:
                        bad_list.append(ori_text)
            except stanza.pipeline.core.LanguageNotDownloadedError:
                stanza.download(lang_id)
                nlp = Pipeline(lang=lang_id, processors="tokenize", use_gpu=False, verbose=False)
            except Exception as e:
                pass
            if nlp:
                try:
                    st_doc = nlp(ori_text)
                except ValueError:
                    bad_list.append(ori_text)
                st_doc_s = st_doc.sentences
                st_len = len(st_doc_s)
                sents = []
                for s in st_doc_s:
                    sents.append(s.text)
                stanza_len.append(st_len)
                stanza_textli.append(sents)
    best_num = max(stanza_len)
    idx = stanza_len.index(best_num)
    best_sent = stanza_textli[idx]
    return best_num, best_sent


def nltk_cut(trans_text, ori_text=None, name_li=None):
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
                pass
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
    input: original text(sub+body), translation text. if both same, only run once.
    main process
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
    if only_en == False:
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
        if (len(sent.split()) < 3 and only_en) or (len(set(sent)) < 4) or (len(sent.split()) == 2):  # 太短的句子
            pass
        take_sents.append(sent)
    return take_sents

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


# Replace the following path with the path to your model
model_path = "/path/to/your/model"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

category_model = MyBert()
loaded_state_dict = torch.load(model_path, map_location=device)
category_model.load_state_dict(loaded_state_dict)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Replace the following path with the path to your data
data_path = "/path/to/your/data"
df_comments = pd.read_csv(data_path)

xlmr_dataset = Dataset(df_comments, tokenizer)
dataloader = DataLoader(
    xlmr_dataset, batch_size=64, num_workers=int(os.cpu_count()), shuffle=False
)

sig_func = nn.Sigmoid().to(device)
category_model.to(device).eval()

# Replace the following path with the path to your output file
output_path = "/path/to/your/output/file"
pickle.dump(obj=(df_external_clean, labeled_df),
            file=open(output_path, 'wb'))

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


stop_words_file_path = '/path/to/your/stop_words_file'
with open(stop_words_file_path) as f:
    stop_words_list = f.read().splitlines()

# Preprocessing function
def preprocess(texts):
    # Your preprocessing code here
    pass


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Your __getitem__ code here
        pass

    def __len__(self):
        return len(self.df)


# Replace the following paths with your own paths
model_path = '/path/to/your/model'
labeled_df_path = '/path/to/your/labeled_df'
output_pickle_path = '/path/to/your/output_pickle'
output_excel_path = '/path/to/your/output_excel'

