AppReviewClustering.py

# README.md

## AppReviewClustering

AppReviewClustering is a Python script that performs clustering on app reviews to identify and analyze various topics and sentiments. The script uses a combination of machine learning algorithms, including HDBSCAN and DBSCAN, along with UMAP for dimensionality reduction. The output is a set of CSV files containing detailed information about the clusters, topics, and sentiments.

### Features

- Load and preprocess app review data
- Perform clustering using HDBSCAN and DBSCAN algorithms
- Calculate various cluster evaluation metrics
- Visualize and analyze the results
- Save the results in CSV format

### Dependencies

- cupy
- numpy
- pandas
- matplotlib
- scikit-learn
- DBCV
- scipy
- timeout_decorator
- hdbscan
- tqdm
- kneed
- cuml

### Usage

1. Set the paths for the input data files (labeled_df.pkl, df_full_comments.pkl) and output directories (csv_output_dir, precompute_duplicate_path).
2. Set the list of topics to exclude from the analysis (topic_not_take_li).
3. Define the functions for clustering, scoring, and saving the results.
4. Set the parameters for the clustering algorithms and other variables.
5. Run the main code to perform clustering and save the results in CSV format.

### Functions

- `need_topic(topics)`: Check if a topic should be included in the analysis.
- `hdbscan_scorer(eval_vec, eval_labels)`: Calculate the HDBSCAN score for a given set of vectors and labels.
- `hdbscan_scorer2(eval_vec, eval_labels)`: Calculate the HDBSCAN score for a given set of vectors and labels using a different method.
- `calc_scores(vectors, labels)`: Calculate various cluster evaluation metrics for a given set of vectors and labels.
- `fit_algo(algo, X)`: Fit a clustering algorithm to a set of data.
- `save_results(topic_di, output_dir_path, pickle_path)`: Save the clustering results to CSV files.
- `convert_str(li)`: Convert a list of numbers to a string with a specific format.
- `remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di)`: Remap the sentence table for visualization purposes.
- `map_sentiment(ori_sent, sent_table, fullcomment_table)`: Map sentiment scores and types for original sentences and full comments.
- `add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df)`: Add centroid distances to the sentence table.
- `refine_statistics(sentence_table, cluster_table_path, topic_table_path)`: Refine statistics by recalculating based on sentence groupings and adding centroid sentences.
- `append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)`: Append filtered comments and sentences to the sentence table.

### Example

```python
import AppReviewClustering as arc

# Set paths and directories
labeled_df_path = "/path/to/your/labeled_df.pkl"
df_full_comments_path = "/path/to/your/df_full_comments.pkl"
csv_output_dir = '/path/to/your/csv_output_dir/'
precompute_duplicate_path = '/path/to/your/precompute_duplicate_path.pkl'

# Set parameters and variables
param_tradmetric_li = [0.1, 0.2, 0.3]
param_dbmetric_li = [0.4, 0.5, 0.6]
param_penaltyloner_li = [0.7, 0.8, 0.9]
param_penaltybiggest_li = [1.0, 1.1, 1.2]

# Run the main code
arc.main(labeled_df_path, df_full_comments_path, csv_output_dir, precompute_duplicate_path, param_tradmetric_li, param_dbmetric_li, param_penaltyloner_li, param_penaltybiggest_li)
```

This will perform clustering on the app reviews and save the results in the specified CSV output directory.