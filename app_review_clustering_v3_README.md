## App Review Clustering

This Python script, is designed to perform clustering on app reviews using various clustering algorithms and evaluation metrics. The primary goal is to identify and group similar reviews, providing insights into user feedback and potential areas for improvement.

### Features

- Load and preprocess app review data
- Calculate various clustering evaluation scores
- Save clustering results to CSV files
- Remap and refine cluster statistics
- Append filtered comments and sentences to the sentence table

### Dependencies

- numpy
- pandas
- pickle
- os
- sklearn
- scipy
- matplotlib
- kneed
- tqdm
- timeout_decorator
- hnswlib
- hdbscan
- cuml
- DBCV

### Usage

1. Set the appropriate paths for input and output files in the script.
2. Run the script.

### Functions

#### calc_scores(vectors, labels)

Calculate various clustering evaluation scores for the given vectors and labels.

#### save_results(topic_di, output_dir_path, pickle_path)

Save clustering results to CSV files.

#### convert_str(li)

Convert a list of strings to a single string.

#### remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di)

Remap and refine cluster statistics.

#### map_sentiment(ori_sent, sent_table, fullcomment_table)

Map sentiment scores to sentences and comments.

#### add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df)

Add centroid distances to the sentence table.

#### refine_statistics(sentence_table, cluster_table_path, topic_table_path)

Recalculate statistics by grouping by sentence due to remapping and add centroid sentences.

#### append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)

Append filtered comments and sentences to the sentence table.