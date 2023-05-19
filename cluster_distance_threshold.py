
#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

# Load data from pickle file
with open('/path/to/your/pickle/file.pkl', 'rb') as f:
    comment_topic_di, topic_cluster_di, topic_statistic_df, topic_di = pickle.load(f)

# Create DataFrame with centroid and eigen vectors for each topic cluster
all_centroid_eigen_vector_df = pd.concat([topic_cluster_df[['cluster_id', 'centroid_vector', 'eigen_vector']] for topic_cluster_df in topic_cluster_di.values()], ignore_index=True)

# Create DataFrame with comment vectors for each topic
all_comment_vector_df = pd.concat([topic_sentence_df[['comment_id', 'cluster_id', 'Embeddings']] for topic_sentence_df in comment_topic_di.values()])

# Calculate distances between comment vectors and centroid/eigen vectors
def calc_distance(row):
    if row['cluster_id'].endswith('-1'):
        return np.nan, np.nan, np.nan, np.nan
    target_df = all_centroid_eigen_vector_df[all_centroid_eigen_vector_df['cluster_id'] == row['cluster_id']]
    centroid_vector = target_df['centroid_vector'].iloc[0].reshape(1, 768)
    eigen_vector = target_df['eigen_vector'].iloc[0].reshape(1, 768)
    sent_emb = np.array(row['Embeddings']).reshape(1, 768)
    cen_dist_c = pairwise_distances(sent_emb, centroid_vector, metric='cosine')[0][0]
    cen_dist_e = pairwise_distances(sent_emb, centroid_vector, metric='euclidean')[0][0]
    eig_dist_c = pairwise_distances(sent_emb, eigen_vector, metric='cosine')[0][0]
    eig_dist_e = pairwise_distances(sent_emb, eigen_vector, metric='euclidean')[0][0]
    return cen_dist_e, cen_dist_c, eig_dist_e, eig_dist_c

all_comment_vector_df[['cent_euc', 'cent_cos', 'eign_euc', 'eign_cos']] = all_comment_vector_df.apply(
    lambda row: pd.Series(calc_distance(row)), axis=1)

# Calculate average distance and save threshold value
all_comment_vector_df['dist_avg'] = all_comment_vector_df[['cent_euc', 'cent_cos', 'eign_euc', 'eign_cos']].mean(axis=1)
max_avg = all_comment_vector_df[['cent_euc', 'cent_cos', 'eign_euc', 'eign_cos']].max().mean()
final_dist_thr = min(all_comment_vector_df['dist_avg'].max(), max_avg)
with open('/path/to/your/threshold/file.pkl', 'wb') as f:
    pickle.dump(final_dist_thr, f)