
import cupy as cp
import numpy as np
import pandas as pd
import pickle
import os
from cuml.manifold import UMAP
from cuml.preprocessing import normalize as cunorm
from sklearn.preprocessing import normalize as sknorm
from cuml.cluster import HDBSCAN, DBSCAN
import functools
import operator
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
import numpy as np
from scipy import stats
from scipy.stats import expon
import warnings
import random
from tqdm.auto import tqdm
from kneed import DataGenerator, KneeLocator
from sklearn.metrics.pairwise import cosine_similarity

used_params_li = []  # Used weight ratios
param_path = "/path/to/your/notebooks/tmp_appreview_network_sentences_clustering_used_params_sentence_v0.10.pkl"

if os.path.exists(param_path):
    used_params_li = pickle.load(open(param_path, 'rb'))

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

labeled_df = pickle.load(open("/path/to/your/notebooks/app_reviews_sentence_final_v0.10_df.pkl", 'rb'))

try:
    labeled_df = labeled_df.drop(['sentiment_type', 'sentiment_type_xlm'], axis=1)
except KeyError:
    pass


def search_metric_weight(weight1, weight2, weight3, weight4):
    """
    Search in top N topics to find base parameter sets (in-topic).
    """
    topic_di = {}
    big_algo_li = []

    for topic in all_topics:
        if topic not in big_topic_li:
            topic_df = topic_df_di[topic]  # Get non-duplicate topic df
            if len(topic_df) < 16:  # Can optimize if length is less than model parameters
                continue

            idx_not_duplicate_li = topic_df.index.tolist()  # Only loc key
            vectors = np.array(topic_df['Embeddings'].tolist())
            distance_matrix = pairwise_distances(vectors, vectors, metric='cosine', n_jobs=15)
            algo_nameli = []
            algo_ansli = []
            algo_take = []

            for algo in algorithm_li:
                params = algo.get_params()
                metri = algo.metric
                algo_nameli.append(params)

                if metri == 'precomputed':  # dbscan
                    X = cp.array(distance_matrix)
                else:
                    X = cunorm(cp.array(vectors))

                try:
                    labels = fit_algo(algo, X)
                except timeout_decorator.TimeoutError:
                    algo_nameli.remove(params)

                algo_take.append(algo)
                algo_ansli.append(labels)
                del X, labels, algo
                gc.collect()

            algo_df = pd.DataFrame(algo_nameli)
            algo_df['cluster_id'] = algo_ansli
            algo_df['algo_obj'] = algo_take
            algo_df[['S_score', 'D_score', 'C_score', 'DBCVc_score', 'DBCVe_score', 'DBCV2c_score', 'DBCV2e_score', 'loner_score', 'toobig_score', 'group_num']] = algo_df.apply(lambda x: calc_scores(vectors, x.cluster_id),
                                                                                                      axis=1, result_type='expand')
            algo_df['D_norm'] = 2 * ((algo_df['D_score'] - algo_df['D_score'].min()) / (algo_df['D_score'].max() - algo_df['D_score'].min() + 1e-10)) - 1
            algo_df['C_norm'] = 2 * ((algo_df['C_score'] - algo_df['C_score'].min()) / (algo_df['C_score'].max() - algo_df['C_score'].min() + 1e-10)) - 1
            algo_df['loner_norm'] = 2 * ((algo_df['loner_score'] - algo_df['loner_score'].min()) / (algo_df['loner_score'].max() - algo_df['loner_score'].min() + 1e-10)) - 1
            algo_df['toobig_norm'] = 2 * ((algo_df['toobig_score'] - algo_df['toobig_score'].min()) / (algo_df['toobig_score'].max() - algo_df['toobig_score'].min() + 1e-10)) - 1
            algo_df['group_num'] = 2 * ((algo_df['group_num'] - algo_df['group_num'].min()) / (algo_df['group_num'].max() - algo_df['group_num'].min() + 1e-10)) - 1  # Smaller is better
            algo_df['final_score'] = (algo_df['S_score'].astype(float) * weight1 - algo_df['D_norm'].astype(float) * weight1 + algo_df['C_norm'].astype(float) * weight1 + algo_df['DBCVc_score'].astype(float) * weight2 + algo_df['DBCVe_score'].astype(float) * weight2 + algo_df['DBCV2c_score'].astype(float) * (weight2 + 0.2) + algo_df['DBCV2e_score'].astype(float) * (weight2 + 0.2) - algo_df['loner_norm'].astype(float) * weight3 - algo_df['toobig_norm'].astype(float) * weight4 - algo_df['group_num'].astype(float) * ((weight3 + weight4) / 3)) / (weight1 * 3 + weight2 * 2 + (weight2 + 0.2) * 2 + weight3 + weight4 + (weight3 + weight4) / 3)
            final_df = algo_df[algo_df['final_score'] == algo_df['final_score'].max()]
            final_df['emb'] = [vectors] * len(final_df)
            topic_di[topic] = final_df
            good_algo_li = final_df['algo_obj'].tolist()
            big_algo_li.extend(good_algo_li)
            del vectors, distance_matrix, topic_df
            gc.collect()

    return topic_di, list(set(big_algo_li))


small_topic_li = list(small_topic_di.keys())
take_score_li = []


def search_metric_weight_smalltopics(algo_li, weight1, weight2, weight3, weight4):
    """
    Use big topics' algo to evaluate in small topics.
    """
    topic_di = {}
    for topic in all_topics:
        if topic not in small_topic_li:
            topic_df = topic_df_di[topic]
            vectors = np.array(topic_df['Embeddings'].tolist())
            distance_matrix = pairwise_distances(vectors, vectors, metric='cosine', n_jobs=-1)
            algo_nameli = []
            algo_ansli = []
            algo_take = []
            algo_id_li = []
            algo_score_li = []

            for (algo_id, algo) in enumerate(algo_li):
                try:
                    if len(topic_df) <= max(algo.get_params()['min_samples'], algo.get_params()['min_cluster_size']):  # min
                        algo_id_li.append(algo_id)
                        algo_score_li.append(np.nan)
                except KeyError:
                    if len(topic_df) <= algo.get_params()['min_samples']:
                        algo_id_li.append(algo_id)
                        algo_score_li.append(np.nan)

                metri = algo.metric
                params = algo.get_params()
                algo_nameli.append(params)

                if metri == 'precomputed':
                    X = cp.array(distance_matrix)
                else:
                    X = cunorm(cp.array(vectors))

                try:
                    labels = fit_algo(algo, X)
                except timeout_decorator.TimeoutError:
                    continue

                algo_take.append(algo)
                algo_ansli.append(labels)
                algo_id_li.append(algo_id)
                del X, labels, algo
                gc.collect()

            if len(algo_score_li) == len(algo_li):  # too small group
                continue

            algo_df = pd.DataFrame(algo_nameli)
            algo_df['cluster_id'] = algo_ansli
            algo_df[['S_score', 'D_score', 'C_score', 'DBCVc_score', 'DBCVe_score', 'DBCV2c_score', 'DBCV2e_score', 'loner_score', 'toobig_score', 'group_num']] = algo_df.apply(lambda x: calc_scores(vectors, x.cluster_id),
                                                                                                          axis=1, result_type='expand')
            algo_df['D_norm'] = 2 * ((algo_df['D_score'] - algo_df['D_score'].min()) / (algo_df['D_score'].max() - algo_df['D_score'].min() + 1e-10)) - 1
            algo_df['C_norm'] = 2 * ((algo_df['C_score'] - algo_df['C_score'].min()) / (algo_df['C_score'].max() - algo_df['C_score'].min() + 1e-10)) - 1
            algo_df['loner_norm'] = 2 * ((algo_df['loner_score'] - algo_df['loner_score'].min()) / (algo_df['loner_score'].max() - algo_df['loner_score'].min() + 1e-10)) - 1
            algo_df['toobig_norm'] = 2 * ((algo_df['toobig_score'] - algo_df['toobig_score'].min()) / (algo_df['toobig_score'].max() - algo_df['toobig_score'].min() + 1e-10)) - 1
            algo_df['group_num'] = 2 * ((algo_df['group_num'] - algo_df['group_num'].min()) / (algo_df['group_num'].max() - algo_df['group_num'].min() + 1e-10)) - 1
            algo_df['final_score'] = (algo_df['S_score'].astype(float) * weight1 - algo_df['D_norm'].astype(float) * weight1 + algo_df['C_norm'].astype(float) * weight1 + algo_df['DBCVc_score'].astype(float) * weight2 + algo_df['DBCVe_score'].astype(float) * weight2 + algo_df['DBCV2c_score'].astype(float) * (weight2 + 0.5) + algo_df['DBCV2e_score'].astype(float) * (weight2 + 0.5) - algo_df['loner_norm'].astype(float) * weight3 - algo_df['toobig_norm'].astype(float) * weight4 - algo_df['group_num'].astype(float) * ((weight3 + weight4) / 10)) / (weight1 * 3 + weight2 * 2 + (weight2 + 0.5) * 2 + weight3 + weight4 + (weight3 + weight4) / 10)
            final_df = algo_df.copy()
            final_df['emb'] = [vectors] * len(final_df)
            final_df['weighted_score'] = final_df['final_score'] * small_topic_di[topic]
            topic_di[topic] = final_df
            score_li = final_df['weighted_score'].tolist()
            algo_score_li.extend(score_li)
            algo_id_li_, algo_score_li_ = zip(*sorted(zip(algo_id_li, algo_score_li)))
            take_score_li.append(list(algo_score_li_))
            del vectors, distance_matrix, topic_df
            gc.collect()

    score_df = pd.DataFrame(take_score_li, columns=[i for i in range(len(algo_li))])
    return topic_di, score_df


def eval_small_cluster(topic_di, smallcluster_scoredf):
    algo_idx = smallcluster_scoredf.mean()[smallcluster_scoredf.mean() == smallcluster_scoredf.mean().max()].index[0]

    for topic in all_topics:
        if topic not in small_topic_li:
            try:
                tmp = topic_di[topic]
            except KeyError:
                continue

            topic_di[topic] = tmp[tmp.index == algo_idx]

    return topic_di

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import pickle


def save_results(topic_di, output_dir_path, pickle_path):
    comment_topic_di = {}
    topic_cluster_di = {}
    cluster_stat_li = []
    calc_overall_score = []


    return (comment_table, cluster_table, topic_statistic_df, comment_table_path,
            cluster_table_path, topic_table_path, pickle_path + '_' + final_score + '.pkl')


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


def remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di):
    all_tmpdf_value = pd.DataFrame()

    return all_tmpdf_value


def map_sentiment(ori_sent, sent_table, fullcomment_table):
    # Your code here

    return sent_sent_type, sent_sent_score, fullcomment, comm_sent_type, comm_sent_score


def add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df):
    def get_distance(sent_emb, cluster_id, topic_cluster_df):
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
    """
    Append filtered comments and sentences to the sentence table.

    Args:
        topic_statistic_df (pd.DataFrame): Topic statistics dataframe.
        df_full_comments (pd.DataFrame): Full comments dataframe.
        sentence_table (pd.DataFrame): Sentence table dataframe.
        sentence_table_path (str): Path to save the updated sentence table.

    Returns:
        pd.DataFrame: Updated sentence table.
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


# Example usage
# sentence_table = append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)
