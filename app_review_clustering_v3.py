
import os
import pickle
import functools
import operator
import warnings
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt
import timeout_decorator
import hnswlib
import hdbscan as hdbscanpkg
from kneed import DataGenerator, KneeLocator
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize as sknorm
from sklearn.metrics.pairwise import cosine_similarity
from cuml.manifold import UMAP
from cuml.preprocessing import normalize as cunorm
from cuml.cluster import HDBSCAN, DBSCAN
from DBCV import DBCV

# Set environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set paths
param_path = "/path/to/your/used_params.pkl"
labeled_df_path = "/path/to/your/labeled_df.pkl"
df_full_comments_path = "/path/to/your/full_comments.pkl"
csv_output_dir = "/path/to/your/csv_output_dir/"
precompute_duplicate_path = "/path/to/your/precompute_duplicate_path.pkl"

# Load data
used_params_li = []
if os.path.exists(param_path):
    used_params_li = pickle.load(open(param_path, 'rb'))

labeled_df = pickle.load(open(labeled_df_path, 'rb'))
df_full_comments, _ = pickle.load(open(df_full_comments_path, 'rb'))

# Preprocessing
warnings.filterwarnings('ignore')


def calc_scores(vectors, labels):
    """
    Calculate various clustering evaluation scores for the given vectors and labels.
    
    Args:
        vectors (numpy.ndarray): The feature vectors.
        labels (numpy.ndarray): The cluster labels.
        
    Returns:
        tuple: A tuple containing the calculated scores.
    """
    idx_take = np.argwhere(labels != -1).squeeze()
    stat = stats.mode(idx_take)
    group_num = len(set(labels))
    group_penalty = 0 if group_num <= 10 else 1 - expon.pdf((group_num - 10) / 10, 0, 2)

    try:
        noise_len = len(np.argwhere(labels == -1).squeeze())
    except TypeError:
        noise_len = 0

    try:
        most_label, most_count = stat.mode[0], stat.count[0]
        count_penalty = most_count / (len(labels) - noise_len)
    except IndexError:
        most_count = 0
        count_penalty = 0

    noise_score = noise_len / len(labels)
    eval_labels = labels[idx_take]
    eval_vec = vectors[idx_take, :]

    try:
        S_score = metrics.silhouette_score(eval_vec, eval_labels, metric='cosine')
        D_score = metrics.davies_bouldin_score(eval_vec, eval_labels)
        C_score = metrics.calinski_harabasz_score(eval_vec, eval_labels)
    except ValueError:
        S_score = 0
        D_score = 0
        C_score = 0

    try:
        hdbscan_score_c, hdbscan_score_e = hdbscan_scorer(eval_vec, eval_labels)
    except timeout_decorator.TimeoutError:
        hdbscan_score_c = 0
        hdbscan_score_e = 0

    try:
        hdbscan_score2_c, hdbscan_score2_e = hdbscan_scorer2(vectors, labels)
    except timeout_decorator.TimeoutError:
        hdbscan_score2_c = 0
        hdbscan_score2_e = 0

    return (S_score, D_score, C_score, hdbscan_score_c, hdbscan_score_e,
            hdbscan_score2_c, hdbscan_score2_e, noise_score, count_penalty, group_penalty)

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import pairwise_distances


def convert_str(li):
    # Your code here

    return "_".join(final_li)


def remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di):
    """
    Note: Only provides visualization remap, the actual stored embedding vector/centroid vector/eigen vector will still be deduplicated.
    labeled_df: Complete labeled_df (including duplicates)
    same_index_di: topic => key sent idx => list of sent idx the same as key
    """
    # Your code here

    return all_tmpdf_value


def map_sentiment(ori_sent, sent_table, fullcomment_table):
    # Your code here

    return sent_sent_type, sent_sent_score, fullcomment, comm_sent_type, comm_sent_score


def add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df):
    def get_distance(sent_emb, cluster_id, topic_cluster_df):
        """
        GOAL: distance to centroid
        """
        # Your code here

        return (dist_c[0][0] + dist_e[0][0]) / 2

    # Your code here

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
