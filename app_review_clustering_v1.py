
import os
import functools
import operator
import warnings
import random
import time
import gc
import pickle
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from hdbscan import hdbscanpkg
from kneed import DataGenerator, KneeLocator
from timeout_decorator import timeout_decorator
from tqdm.auto import tqdm

# Load used parameters
used_params_li = []
used_params_file = "path/to/used_params_file.pkl"
used_params_li = pickle.load(open(used_params_file, 'rb'))

warnings.filterwarnings('ignore')

# Load labeled data
labeled_df_file = "path/to/labeled_df_file.pkl"
labeled_df = pickle.load(open(labeled_df_file, 'rb'))

# Remove unnecessary columns
try:
    labeled_df = labeled_df.drop(['sentiment_type', 'sentiment_type_xlm'], axis=1)
except KeyError:
    pass


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

    for t_id, topic in enumerate(topic_di.keys()):
        process_topic(topic_di, topic, t_id, comment_topic_di, topic_cluster_di, cluster_stat_li, calc_overall_score)

    comment_table, cluster_table, topic_statistic_df = create_tables(topic_di, comment_topic_di, topic_cluster_di)

    final_score = format_final_score(calc_overall_score)
    cluster_num = format_cluster_num(cluster_table)
    cluster_size_total = format_cluster_size_total(cluster_table)

    output_dir_path = f"{output_dir_path}_{cluster_num}_{cluster_size_total}_{final_score}"
    create_output_directory(output_dir_path)

    comment_table_path, cluster_table_path, topic_table_path = save_tables_to_csv(output_dir_path, comment_table, cluster_table, topic_statistic_df)

    pkl_dir = "/".join(pickle_path.split("/")[:-1])
    create_output_directory(pkl_dir)

    pickle.dump(obj=(comment_topic_di, topic_cluster_di, topic_statistic_df, topic_di),
                file=open(pickle_path + '_' + final_score + '.pkl', 'wb'))

    return comment_table, cluster_table, topic_statistic_df, comment_table_path, cluster_table_path, pickle_path + '_' + final_score + '.pkl'



def create_output_directory(output_dir_path):
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)


def convert_str(li):
    final_li = []
    for i in li:
        i = str(i)
        if len(i) == 1:
            final_li.append(i + '.0')
        else:
            final_li.append(i)
    return "_".join(final_li)


def remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di):
    """
    Note: Only provides visualization remap, the actual stored embedding vector/centroid vector/eigen vector will still be deduplicated.
    labeled_df: Complete labeled_df (including duplicates)
    same_index_di: topic => key sent idx => list of sent idx the same as key
    """
    # Your code logic here

import numpy as np
import pandas as pd
import random
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm
from IPython.display import clear_output

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
