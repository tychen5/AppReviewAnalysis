
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
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
from scipy.stats import expon
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sknorm
from cuml.manifold import UMAP
from cuml.preprocessing import normalize as cunorm
from cuml.cluster import HDBSCAN, DBSCAN
from DBCV import DBCV
import hdbscan as hdbscanpkg
from tqdm.auto import tqdm
from kneed import DataGenerator, KneeLocator
import timeout_decorator

# Load used_params_li from a file
used_params_li = pickle.load(open("path/to/your/used_params_file.pkl", 'rb'))

# Load labeled_df from a file
labeled_df = pickle.load(open("path/to/your/labeled_df_file.pkl", 'rb'))

# Load df_full_comments from a file
df_full_comments, _ = pickle.load(open("path/to/your/df_full_comments_file.pkl", 'rb'))

# Set the output directory for CSV files
csv_output_dir = 'path/to/your/csv_output_directory/'

# Define the function need_topic
def need_topic(topics):
    for to in topics:
        if to not in topic_not_take_li:
            return 1
    return 0

# Define the function hdbscan_scorer
@timeout_decorator.timeout(2.9)
def hdbscan_scorer(eval_vec, eval_labels):
    try:
        hdbscan_score_c = DBCV(eval_vec, eval_labels, dist_function=cosine)
    except ValueError:
        hdbscan_score_c = 0
    try:
        hdbscan_score_e = DBCV(eval_vec, eval_labels, dist_function=euclidean)
    except ValueError:
        hdbscan_score_e = 0
    if np.isnan(hdbscan_score_c) and np.isnan(hdbscan_score_e):
        hdbscan_score_c = 0
        hdbscan_score_e = 0
    elif np.isnan(hdbscan_score_c):
        hdbscan_score_c = 0
    elif np.isnan(hdbscan_score_e):
        hdbscan_score_e = 0
    return hdbscan_score_c, hdbscan_score_e

# Define the function hdbscan_scorer2
@timeout_decorator.timeout(3.9)
def hdbscan_scorer2(eval_vec, eval_labels):
    try:
        hdbscan_score_c = hdbscanpkg.validity.validity_index(eval_vec, eval_labels, metric='cosine')
    except ValueError:
        hdbscan_score_c = 0
    try:
        hdbscan_score_e = hdbscanpkg.validity.validity_index(eval_vec, eval_labels, metric='euclidean')
    except ValueError:
        hdbscan_score_e = 0
    if np.isnan(hdbscan_score_c) and np.isnan(hdbscan_score_e):
        hdbscan_score_c = 0
        hdbscan_score_e = 0
    elif np.isnan(hdbscan_score_c):
        hdbscan_score_c = 0
    elif np.isnan(hdbscan_score_e):
        hdbscan_score_e = 0
    return hdbscan_score_c, hdbscan_score_e

# Define the function calc_scores
def calc_scores(vectors, labels):
    # Your implementation here
    pass

# Define the function fit_algo
@timeout_decorator.timeout(4)
def fit_algo(algo, X):
    return algo.fit_predict(X)

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import random
from tqdm.auto import tqdm
import pickle
from IPython.display import clear_output
from sklearn import datasets
from hdbscan import HDBSCAN
from cuml.cluster import HDBSCAN, DBSCAN


def add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df):
    def get_distance(sent_emb, cluster_id, topic_cluster_df):
        """
        Calculate distance to centroid
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
        duplicate_df = remapping_sentence_table(topic_sentence_df, labeled_df, duplicate_idx_di)
        sentence_table = pd.concat([sentence_table, topic_sentence_df], ignore_index=True)
        sentence_table = pd.concat([sentence_table, duplicate_df], ignore_index=True)
    sentence_table = sentence_table.drop(['Embeddings', 'topics'], axis=1)
    sentence_table.to_csv(sentence_table_path, index=False)
    return sentence_table

