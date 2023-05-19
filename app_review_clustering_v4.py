
import os
import functools
import operator
import warnings
import pickle
import numpy as np
import pandas as pd
import cupy as cp
from cuml.manifold import UMAP
from cuml.preprocessing import normalize as cunorm
from sklearn.preprocessing import normalize as sknorm
from cuml.cluster import HDBSCAN, DBSCAN
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from DBCV import DBCV
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import gc
import requests
import json
import random
import time
import timeout_decorator
import hdbscan as hdbscanpkg
from scipy import stats
from scipy.stats import expon
from tqdm.auto import tqdm
from kneed import DataGenerator, KneeLocator
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
from sklearn.decomposition import PCA
import torch
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
from cuml.preprocessing import normalize
import cupy as cp
from cuml import PCA

# Load labeled_df and remove sensitive information
labeled_df_path = "/path/to/your/labeled_df.pkl"
labeled_df = pickle.load(open(labeled_df_path, 'rb'))

# Remove sensitive information from the code
used_params_li = []
param_path = "/path/to/your/used_params.pkl"
if os.path.exists(param_path):
    used_params_li = pickle.load(open(param_path, 'rb'))

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Update the following paths with your own paths
csv_output_dir = '/path/to/your/csv_output_dir/'
precompute_duplicate_path = '/path/to/your/precompute_duplicate_path.pkl'

# Add your own logic for processing labeled_df and other data


# Create a custom embedder class
class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        embeddings = normalize(embeddings)
        return embeddings

# Instantiate the custom embedder
custom_embedder = CustomEmbedder(embedding_model=sentence_model)

# Instantiate BERTopic with custom embedder
topic_model = BERTopic(embedding_model=custom_embedder, umap_model=umap_model, hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer_model, n_gram_range=(1, 3), diversity=0.85, nr_topics='auto',
                       calculate_probabilities=True)

# Fit and transform the topic model
docs = labeled_df['sent_translation'].tolist()
topics, probs = topic_model.fit_transform(docs)


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings


# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
custom_embedder = CustomEmbedder(embedding_model=embedding_model)
topic_model = BERTopic(umap_model=umap_model, embedding_model=custom_embedder)

# Fit and transform the documents
topics, probs = topic_model.fit_transform(docs)

# Find topics related to "access"
topic_model.find_topics("access")

# Load the 20 newsgroups dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs = data["data"]
targets = data["target"]
target_names = data["target_names"]
classes = [data["target_names"][i] for i in data["target"]]

# Initialize the topic dictionary
topic_di = {}

# Define a function to fit the algorithm with a timeout
@timeout_decorator.timeout(4)
def fit_algo(algo, X):
    return algo.fit_predict(X)

# Initialize the list of big algorithms
big_algo_li = []

# Define a function to search for the best metric weights
def search_metric_weight(weight1, weight2, weight3, weight4):
    """
    Search for the best metric weights in top N topics to find base parameter sets (in-topic).
    """
    # Code for searching metric weights goes here

# Define a function to search for the best metric weights in small topics
def search_metric_weight_smalltopics(algo_li, weight1, weight2, weight3, weight4):
    """
    Use big topics' algorithm to evaluate in small topics.
    """
    # Code for searching metric weights in small topics goes here

# Define a function to evaluate small clusters
def eval_small_cluster(topic_di, smallcluster_scoredf):
    """
    Evaluate small clusters and update the topic dictionary.
    """
    # Code for evaluating small clusters goes here

# Initialize the debug list
debug_li = []

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import pickle


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
        if cluster_id.endswith('-1'):
            return np.nan
        centroid_emb = topic_cluster_df[topic_cluster_df['cluster_id'] == cluster_id].iloc[0]['centroid_vector'].reshape(1, reduct_dim)
        sent_emb = np.array(sent_emb).reshape(1, reduct_dim)
        dist_c = pairwise_distances(sent_emb, centroid_emb, metric='cosine')
        dist_e = pairwise_distances(sent_emb, centroid_emb, metric='euclidean')
        return (dist_c[0][0] + dist_e[0][0]) / 2

    sentence_table = pd.DataFrame()
    for topic in topic_df_dict.keys():
        topic_sentence_df = topic_df_dict[topic]
        topic_cluster_df = cluster_df_dict[topic]
        topic_sentence_df['centroid_distance'] = topic_sentence_df.apply(lambda x: get_distance(x.Emb_reduct, x.cluster_id, topic_cluster_df), axis=1)
        duplicate_idx_di = clean_topic_df_di[topic]
        sentence_table = pd.concat([sentence_table, topic_sentence_df], ignore_index=True)
    sentence_table = sentence_table.drop(['Embeddings', 'topics', 'Emb_reduct'], axis=1)
    sentence_table.to_csv(sentence_table_path, index=False)
    return sentence_table

def refine_statistics(sentence_table, cluster_table_path, topic_table_path):
    """
    Recalculate statistics by grouping by sentence due to remapping.
    Add centroid sent.
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

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


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


# Replace the following paths with your own paths
topic_statistic_df_path = "/path/to/your/topic_statistic_df.csv"
df_full_comments_path = "/path/to/your/df_full_comments.csv"
sentence_table_path = "/path/to/your/sentence_table.csv"

# Load DataFrames from CSV files
topic_statistic_df = pd.read_csv(topic_statistic_df_path)
df_full_comments = pd.read_csv(df_full_comments_path)
sentence_table = pd.read_csv(sentence_table_path)

# Call the function
sentence_table = append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)