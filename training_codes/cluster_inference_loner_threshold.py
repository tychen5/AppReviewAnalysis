
#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

# Load data
pkl_path = '/path/to/your/pickle/file.pkl'
comment_topic_di, topic_cluster_di, topic_statistic_df, topic_di = pickle.load(open(pkl_path, 'rb'))

# Prepare data
all_centroid_eigen_vector_df = pd.DataFrame()
for k, topic_cluster_df in topic_cluster_di.items():
    topic_cluster_df = topic_cluster_df[['cluster_id', 'centroid_vector', 'eigen_vector']]
    all_centroid_eigen_vector_df = pd.concat([all_centroid_eigen_vector_df, topic_cluster_df], ignore_index=True)

all_comment_vector_df = pd.DataFrame()
for k, topic_sentence_df in comment_topic_di.items():
    topic_sentence_df = topic_sentence_df[['comment_id', 'cluster_id', 'Embeddings']]
    all_comment_vector_df = pd.concat([all_comment_vector_df, topic_sentence_df])

def calc_distance(key_cid, key_emb, value_df):
    if key_cid.endswith('-1'):
        return np.nan, np.nan, np.nan, np.nan
    target_df = value_df[value_df['cluster_id'] == key_cid]
    centroid_vector = target_df['centroid_vector'].iloc[0].reshape(1, 768)
    eigen_vector = target_df['eigen_vector'].iloc[0].reshape(1, 768)
    sent_emb = np.array(key_emb).reshape(1, 768)
    cen_dist_c = pairwise_distances(sent_emb, centroid_vector, metric='cosine')[0][0]
    cen_dist_e = pairwise_distances(sent_emb, centroid_vector, metric='euclidean')[0][0]
    eig_dist_c = pairwise_distances(sent_emb, eigen_vector, metric='cosine')[0][0]
    eig_dist_e = pairwise_distances(sent_emb, eigen_vector, metric='euclidean')[0][0]
    return cen_dist_e, cen_dist_c, eig_dist_e, eig_dist_c

all_comment_vector_df[['cent_euc', 'cent_cos', 'eign_euc', 'eign_cos']] = all_comment_vector_df.apply(
    lambda x: calc_distance(x.cluster_id, x.Embeddings, all_centroid_eigen_vector_df), axis=1, result_type='expand')

all_comment_vector_df['dist_avg'] = (all_comment_vector_df['cent_euc'] + all_comment_vector_df['cent_cos'] +
                                     all_comment_vector_df['eign_euc'] + all_comment_vector_df['eign_cos']) / 4

max_avg = (all_comment_vector_df['cent_euc'].max() + all_comment_vector_df['cent_cos'].max() +
           all_comment_vector_df['eign_euc'].max() + all_comment_vector_df['eign_cos'].max()) / 4

final_dist_thr = min(all_comment_vector_df['dist_avg'].max(), max_avg)
name = 'Network'
pickle.dump(obj=final_dist_thr, file=open('/path/to/your/output/file.pkl', 'wb'))

def calc_avg_dist(centroid_vector, eigen_vector, sent_emb, loner_dist_thr):
    centroid_vector = centroid_vector.reshape(1, 768)
    eigen_vector = eigen_vector.reshape(1, 768)
    sent_emb = np.array(sent_emb).reshape(1, 768)
    cen_dist_c = pairwise_distances(sent_emb, centroid_vector, metric='cosine')[0][0]
    cen_dist_e = pairwise_distances(sent_emb, centroid_vector, metric='euclidean')[0][0]
    eig_dist_c = pairwise_distances(sent_emb, eigen_vector, metric='cosine')[0][0]
    eig_dist_e = pairwise_distances(sent_emb, eigen_vector, metric='euclidean')[0][0]
    avg_dist = (cen_dist_c + cen_dist_e + eig_dist_c + eig_dist_e) / 4
    if avg_dist > loner_dist_thr:
        return loner_dist_thr + 1e-6
    else:
        return avg_dist
