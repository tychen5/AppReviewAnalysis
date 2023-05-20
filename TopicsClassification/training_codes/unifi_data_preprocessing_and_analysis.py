import pandas as pd
import numpy as np
from googletrans import Translator
import functools
import operator
import ast
import time
import random
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as stats

# Replace the following paths with your own paths
data_path_1 = "/path/to/your/data/unifi_network_controller_survey.csv"
data_path_2 = "/path/to/your/data/unifi_network_firmware_setup.csv"
data_path_3 = "/path/to/your/data/df_android.pkl"
data_path_4 = "/path/to/your/data/df_ios.pkl"
data_path_5 = "/path/to/your/data/review_export_report_All.xlsx"
data_path_6 = "/path/to/your/data/review_export_report-Unifi-Network.csv"

data_df1 = pd.read_csv(data_path_1)
data_df2 = pd.read_csv(data_path_2)
data_df3 = pd.read_pickle(data_path_3)
data_df4 = pd.read_pickle(data_path_4)
data_df5 = pd.read_excel(data_path_5)
data_df6 = pd.read_csv(data_path_6)

all_dfs = [data_df1, data_df2, data_df3, data_df4, data_df5, data_df6]
all_take_dfs = []

for df in all_dfs:
    cols = df.columns
    take_col = []
    for col in cols:
        try:
            if len(df[col].unique()) > 1:
                take_col.append(col)
        except TypeError:
            take_col.append(col)
    all_take_dfs.append(df[take_col])

def clean_topics(textstr):
    """
    Cleansing the topics by removing unnecessary characters and replacing them with underscores.
    """
    text = textstr[1:-1]
    text = text.replace('"', '')
    text = text.replace(' ', '')
    text = text.replace(",", "_")
    text = text.replace("&", "_")
    return text

def clean_externaldata(strli):
    final_li = ast.literal_eval(strli)
    if len(final_li) < 1:
        return ['UNK']
    else:
        return final_li

df_external = all_take_dfs[-1].append(all_take_dfs[-2])
df_external["topics_clean"] = df_external.Topics.apply(clean_externaldata)
df_external_topics = df_external['topics_clean'].tolist()
df_external_topics = sorted(list(set(functools.reduce(operator.iconcat, df_external_topics, []))))

# Replace the following list with your own list of topics
topic_li = ['Topic1', 'Topic2', 'Topic3']

set(topic_li) - set(df_external_topics)
topic_li = sorted(df_external_topics)

df_android = all_take_dfs[2]
df_ios = all_take_dfs[3]

def reset_topic(oristr):
    take_label = []
    if oristr == '{}':
        return ['UNK']
    sent = oristr.replace("{", "")
    sent = sent.replace("}", "")
    firstli = sent.split('"')
    secondli = sent.split(',')
    for word in firstli:
        if (word not in topic_li) and (',' not in word):
            unk_wordli.append(word)
            unk_sent.append(sent)
        for topic in topic_li:
            if topic == word:
                take_label.append(topic)
    for word in secondli:
        if (word not in topic_li) and ('"' not in word):
            unk_wordli.append(word)
            unk_sent.append(sent)
        for topic in topic_li:
            if topic == word:
                take_label.append(topic)
    if len(take_label) < 1:
        return ['UNK']
    else:
        return sorted(list(set(take_label)))

df_android["topics_clean"] = df_android.topics.apply(reset_topic)
df_ios["topics_clean"] = df_ios.topics.apply(reset_topic)

def translate2en(textstr, country):
    try:
        time.sleep(random.randint(1, 60))
        return translator.translate(textstr).text
    except Exception as e:
        time.sleep(random.randint(30, 300))

df_android['message_clean'] = df_android.apply(lambda x: translate2en(x.message, x.country), axis=1)
df_ios['message_clean'] = df_ios.apply(lambda x: translate2en(x.message, x.country), axis=1)

# Save the cleaned dataframes to pickle files
df_ios.to_pickle("/path/to/your/data/df_ios_cleaned.pkl")
df_android.to_pickle("/path/to/your/data/df_android_cleaned.pkl")
df_external.to_pickle("/path/to/your/data/df_external_cleaned.pkl")

# Load the cleaned dataframes
df_ios = pd.read_pickle("/path/to/your/data/df_ios_cleaned.pkl")
df_android = pd.read_pickle("/path/to/your/data/df_android_cleaned.pkl")
df_external = pd.read_pickle("/path/to/your/data/df_external_cleaned.pkl").reset_index(drop=True)

# Further processing and analysis
