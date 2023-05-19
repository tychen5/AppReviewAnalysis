
#!/usr/bin/env python
# coding: utf-8

import spacy
import pickle
import nltk
import stanza
import gc
import json
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline.multilingual import MultilingualPipeline
from nltk.tokenize import sent_tokenize

# Load parameters from JSON file
with open('./parameter.json', 'r') as f:
    param_dict = json.load(f)

# Download spacy models
prefix = ['ca', 'zh', 'da', 'nl', 'en', 'fr', 'de', 'el', 'it', 'ja', 'lt', 'mk', 'nb', 'pl', 'pt', 'ro', 'ru', 'es']
postfix = ['lg', 'md', 'sm', 'trf']
mid_text = ['_core_news_', '_dep_news_', '_core_web_']
spacy.cli.download("xx_ent_wiki_sm")
spacy.cli.download("xx_sent_ud_sm")
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

# Download stanza models
stanza_lang = "af ar be bg bxr ca cop cs cu da de el en es et eu fa fi fr fro ga gd gl got grc he hi hr hsb hu hy id it ja kk kmr ko la is lt lv lzh mr mt nl nn no olo orv pl pt ro ru sk sl sme sr sv swl ta te tr th ug uk ur vi wo zh-hans zh-hant"
stanza_lang = stanza_lang.split(" ")
print(len(stanza_lang))
stanza.download(lang="multilingual")

for langu in stanza_lang:
    try:
        stanza.download(langu)
    except ValueError:
        print(langu)
        continue

# Download nltk models
nltk.download('popular')
nltk.download('all')
