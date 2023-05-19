
import os
import pickle
import functools
import operator
import warnings
import random
import time
import gc
import requests
import json

import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from DBCV import DBCV
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from timeout_decorator import timeout
import hdbscan as hdbscanpkg
from scipy import stats
from scipy.stats import expon
from tqdm.auto import tqdm
from kneed import DataGenerator, KneeLocator
from sklearn.metrics.pairwise import cosine_similarity
from cuml.manifold import UMAP
from cuml.preprocessing import normalize as cunorm
from sklearn.preprocessing import normalize as sknorm
from cuml.cluster import HDBSCAN, DBSCAN

warnings.filterwarnings('ignore')

# Load labeled_df
labeled_df_path = "/path/to/your/labeled_df.pkl"
labeled_df = pickle.load(open(labeled_df_path, 'rb'))

# Load df_full_comments
df_full_comments_path = "/path/to/your/df_full_comments.pkl"
df_full_comments, _ = pickle.load(open(df_full_comments_path, 'rb'))

# Set paths and directories
param_path = "/path/to/your/used_params.pkl"
csv_output_dir = '/path/to/your/csv_output_dir/'
precompute_duplicate_path = '/path/to/your/precompute_duplicate_path.pkl'

# Other variables
used_params_li = []
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
topic_not_take_li = ['Dissatisfied users', 'Gaming', 'HDMI', 'Pricing', 'Frequency', 'Payment', 'Satisfied users',
                     'Social & Collaboration', 'Import Export', 'Camera & Photos', 'Audio', 'Streaming']

# Functions
def need_topic(topics):
    for to in topics:
        if to not in topic_not_take_li:
            return 1
    return 0


@timeout(4)
def fit_algo(algo, X):
    return algo.fit_predict(X)


def convert_str(li):
    final_li = []
    for i in li:
        i = str(i)
        if len(i) == 1:
            final_li.append(i + '.00')
        elif len(i) == 3:
            final_li.append(i + '0')
        else:
            final_li.append(i)
    return "_".join(final_li)





def add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df):
    def get_distance(sent_emb, cluster_id, topic_cluster_df):
        """
        Calculate distance to centroid.
        """
        if cluster_id.endswith('-1'):
            return np.nan
        centroid_emb = topic_cluster_df[topic_cluster_df['cluster_id'] == cluster_id].iloc[0]['centroid_vector'].reshape(1, 768)
        sent_emb = np.array(sent_emb).reshape(1, 768)
        dist_c = pairwise_distances(sent_emb, centroid_emb, metric='cosine')
        dist_e = pairwise_distances(sent_emb, centroid_emb, metric='euclidean')
        return (dist_c[0][0] + dist_e[0][0]) / 2

    sentence_table = pd.DataFrame()
    for topic in topic_df_dict.keys():
        topic_sentence_df = topic_df_dict[topic]
        topic_cluster_df = cluster_df_dict[topic]
        topic_sentence_df['centroid_distance'] = topic_sentence_df.apply(lambda x: get_distance(x.Embeddings, x.cluster_id, topic_cluster_df), axis=1)
        duplicate_idx_di = clean_topic_df_di[topic]
        sentence_table = pd.concat([sentence_table, topic_sentence_df], ignore_index=True)

    sentence_table = sentence_table.drop(['Embeddings', 'topics'], axis=1)
    sentence_table.to_csv(sentence_table_path, index=False)
    return sentence_table

import random
import pandas as pd
import numpy as np
import pickle
from IPython.display import clear_output
from tqdm.auto import tqdm
import requests
import json

def refine_statistics(sentence_table, cluster_table_path, topic_table_path):
    """
    Refine statistics by recalculating based on sentence groupings and adding centroid sentences.
    """
    cluster_table = pd.read_csv(cluster_table_path)
    topic_table = pd.read_csv(topic_table_path)
    sentence_table_part = sentence_table[['topic_id', 'cluster_id', 'sent_translation', 'comment_sent', 'centroid_distance']]
    groupby_df = sentence_table_part.groupby(['topic_id', 'cluster_id', 'sent_translation']).agg(list).reset_index()
    count_df = pd.DataFrame(groupby_df['cluster_id'].value_counts()).reset_index()
    count_df.columns = ['cluster_id', 'sentence_num']
    count_df = count_df.sort_values(['cluster_id']).reset_index(drop=True)

    def refine_loner(topic_id, count_df):
        tid = str(topic_id)
        if len(tid) < 2:
            tid = '0' + tid
        count_topic_df = count_df[count_df['cluster_id'].str.startswith(tid)]
        topic_sent_num = count_topic_df['sentence_num'].sum()
        topic_loner_num = count_topic_df[count_topic_df['cluster_id'].str.endswith('-1')]['sentence_num'].iloc[0]
        loner_ratio = topic_loner_num / topic_sent_num
        return str(topic_loner_num), loner_ratio

    topic_table[['loner_size', 'loner_ratio']] = topic_table.apply(lambda x: refine_loner(x.topic_id, count_df), axis=1, result_type='expand')
    topic_table.to_csv(topic_table_path, index=False)

    def refine_cluster_statistics(cluster_id, count_df):
        tid = cluster_id.split('_')[0]
        count_topic_df = count_df[count_df['cluster_id'].str.startswith(tid)]
        topic_sent_num = count_topic_df['sentence_num'].sum()
        topic_cluster_num = count_topic_df[count_topic_df['cluster_id'] == cluster_id]['sentence_num'].iloc[0]
        cluster_ratio = topic_cluster_num / topic_sent_num
        return str(topic_cluster_num), cluster_ratio

    cluster_table[['cluster_size', 'cluster_ratio']] = cluster_table.apply(lambda x: refine_cluster_statistics(x.cluster_id, count_df), result_type='expand', axis=1)
    groupby_df['centroid_distance'] = groupby_df['centroid_distance'].apply(lambda x: x[0])

    def find_nearest_centroid_sent(cid, groupby_df):
        cluster_sent_df = groupby_df[groupby_df['cluster_id'] == cid]
        min_dist = cluster_sent_df['centroid_distance'].min()
        return cluster_sent_df[cluster_sent_df['centroid_distance'] == min_dist]['sent_translation'].iloc[0]

    cluster_table['centroid_sentence'] = cluster_table['cluster_id'].apply(find_nearest_centroid_sent, args=(groupby_df,))
    cluster_table.to_csv(cluster_table_path, index=False)
    return topic_table, cluster_table

# Add other functions here

# Main code
# Replace the following variables with your own paths and values
param_tradmetric_li = []  # Add your list of traditional metric values
param_dbmetric_li = []  # Add your list of DB metric values
param_penaltyloner_li = []  # Add your list of penalty loner values
param_penaltybiggest_li = []  # Add your list of penalty biggest values
param_path = '/path/to/your/params.pkl'  # Replace with your params.pkl file path
csv_output_dir = '/path/to/your/csv_output_dir/'  # Replace with your CSV output directory path

random.shuffle(param_tradmetric_li)
random.shuffle(param_dbmetric_li)
random.shuffle(param_penaltyloner_li)
random.shuffle(param_penaltybiggest_li)
take_num = (len(param_tradmetric_li) * len(param_dbmetric_li) * len(param_penaltyloner_li) * len(param_penaltybiggest_li)) * 2

for r in tqdm(range(take_num)):
    # Add your main code logic here

    # Replace the following URL with your own API endpoint
    url = "http://your-api-url.com/api/v1/nlp/cluster/appreview"
    payload = json.dumps({
        "text": ["didnt get internet until rebooted udm pro", "why is it soooo slow??"]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)

    # Add more code logic here

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


def append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path):
    """Append filtered comments and sentences to the sentence table.

    Args:
        topic_statistic_df (DataFrame): DataFrame containing topic statistics.
        df_full_comments (DataFrame): DataFrame containing full comments.
        sentence_table (DataFrame): DataFrame containing sentence table.
        sentence_table_path (str): Path to save the updated sentence table.

    Returns:
        DataFrame: Updated sentence table.
    """
    topicmap_di = {}
    topic_name_li = topic_statistic_df['topic_name'].tolist()
    topic_id_li = topic_statistic_df['topic_id'].tolist()
    for name, idx in zip(topic_name_li, topic_id_li):
        topicmap_di[name] = idx
    diff_comm_idli = sorted(set(df_full_comments['comment_id'].unique()) - set(sentence_table['comment_id'].unique()))

    for cid in diff_comm_idli:
        missing_df = df_full_comments[df_full_comments['comment_id'] == cid]
        missing_df = missing_df[['App Store', 'App Name', 'version', 'published_at', 'language', 'comment_id',
                                 'rating', 'ori_comment', 'translation', 'sentiment_overalltype', 'sentiment_overallscore',
                                 ]]
        missing_df.columns = ['App Store', 'App Name', 'version', 'published_at', 'oricomment_lang', 'comment_id',
                              'rating', 'comment_sent', 'sent_translation', 'sentiment_overalltype', 'sentiment_overallscore']
        topic_id = topicmap_di['UNK']
        tid = str(topic_id)
        if len(tid) < 2:
            tid = '0' + tid
        cid = tid + '_-1'
        missing_df['topic_id'] = topicmap_di['UNK']
        missing_df['cluster_id'] = cid
        missing_df['varianceavg_distance'] = np.nan
        missing_df['centroid_distance'] = np.nan
        sentence_table = sentence_table.append(missing_df, ignore_index=True)

    sentence_table.to_csv(sentence_table_path, index=False)
    return sentence_table
