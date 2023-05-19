This Python script performs clustering on app reviews to identify and analyze different topics and sentiments. The script processes the data, performs dimensionality reduction, computes various clustering evaluation scores, and saves the results in CSV files.

### Dependencies

- os
- pickle
- warnings
- functools
- operator
- itertools
- multiprocessing
- numpy
- pandas
- cupy
- matplotlib
- timeout_decorator
- hnswlib
- tqdm
- kneed
- sklearn
- cuml
- DBCV
- scipy
- hdbscan
- torch

### Overview

The script starts by loading a labeled DataFrame containing app reviews and performs preprocessing steps such as dropping unnecessary columns, filtering by app name, and computing statistics. It then applies dimensionality reduction using PCA and whitening techniques to reduce the size of the embeddings.

The script calculates various clustering evaluation scores for the given vectors and labels, such as silhouette score, Davies-Bouldin score, Calinski-Harabasz score, and HDBSCAN scores. It then saves the results in CSV files, including comment tables, cluster tables, and topic statistic tables.

Additional functions are provided to refine the statistics, remap sentences, map sentiments, add centroid distances, and append filtered comments and sentences to the sentence table.

### Key Functions

- `compute_kernel_bias(vecs, n_components)`: Computes kernel and bias for dimensionality reduction.
- `transform_and_normalize(vecs, kernel, bias)`: Applies transformation and normalization to the vectors.
- `whitening_torch(emb)`: Applies whitening transformation using PyTorch.
- `calc_scores(vectors, labels)`: Calculates various clustering evaluation scores.
- `save_results(topic_di, output_dir_path, pickle_path)`: Saves the results in CSV files.
- `remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di)`: Remaps sentences for visualization.
- `map_sentiment(ori_sent, sent_table, fullcomment_table)`: Maps sentiment scores and types.
- `add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df)`: Adds centroid distances to the sentence table.
- `refine_statistics(sentence_table, cluster_table_path, topic_table_path)`: Refines statistics by grouping sentences and adding centroid sentences.
- `append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)`: Appends filtered comments and sentences to the sentence table.

### Usage

To use this script, simply update the file paths for the input data (labeled DataFrame, full comments DataFrame, and used parameters) and the output directory. Then, run the script to perform clustering on the app reviews and save the results in CSV files.