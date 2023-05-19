# App Review Clustering

This Python script, `app_review_clustering.py`, is designed to perform clustering on app reviews using various clustering algorithms and metrics. The primary goal is to identify and group similar reviews based on their content, sentiment, and other features. This can help developers and product managers to better understand user feedback and improve their applications.

### Features

- Import and preprocess app review data
- Perform clustering using HDBSCAN and DBSCAN algorithms
- Calculate various clustering metrics, such as Silhouette score, Davies-Bouldin score, and Calinski-Harabasz score
- Optimize clustering parameters using a search function
- Evaluate clustering results on small and large topics
- Save clustering results and statistics to files

### Dependencies

- cupy
- numpy
- pandas
- pickle
- cuml
- sklearn
- matplotlib
- collections
- scipy
- gc
- requests
- json
- random
- time
- timeout_decorator
- hdbscan
- tqdm
- kneed

### Usage

1. Import the required libraries and set the environment variables.
2. Load the app review data and preprocess it.
3. Define the clustering algorithms and their parameters.
4. Perform clustering using the `search_metric_weight` function.
5. Evaluate the clustering results on small and large topics using the `search_metric_weight_smalltopics` and `eval_small_cluster` functions.
6. Save the clustering results and statistics to files using the `save_results` function.
7. Refine the statistics and append filtered comments and sentences to the sentence table using the `refine_statistics` and `append_filtered_commsent` functions.

### Functions

- `search_metric_weight`: Searches for the best clustering parameters in top N topics.
- `search_metric_weight_smalltopics`: Evaluates the clustering results on small topics using the best algorithms found in the previous step.
- `eval_small_cluster`: Evaluates the clustering results on small topics and returns the best clustering parameters.
- `save_results`: Saves the clustering results and statistics to files.
- `convert_str`: Converts a list of numbers to a string with a specific format.
- `remapping_sentence_table`: Remaps the sentence table based on the clustering results.
- `map_sentiment`: Maps the sentiment scores to the clustered sentences and comments.
- `add_centroid_dist`: Adds the centroid distance to the sentence table.
- `refine_statistics`: Recalculates the statistics and adds the centroid sentence to the cluster table.
- `append_filtered_commsent`: Appends filtered comments and sentences to the sentence table.

### Example Usage

```python
import app_review_clustering as arc

# Load and preprocess app review data
labeled_df = arc.load_app_review_data()

# Perform clustering and evaluate results
topic_di, big_algo_li = arc.search_metric_weight(weight1, weight2, weight3, weight4)
topic_di, score_df = arc.search_metric_weight_smalltopics(big_algo_li, weight1, weight2, weight3, weight4)
topic_di = arc.eval_small_cluster(topic_di, score_df)

# Save clustering results and statistics
comment_table, cluster_table, topic_statistic_df, comment_table_path, cluster_table_path, topic_table_path, pickle_path = arc.save_results(topic_di, output_dir_path, pickle_path)

# Refine statistics and append filtered comments and sentences
topic_table, cluster_table = arc.refine_statistics(sentence_table, cluster_table_path, topic_table_path)
sentence_table = arc.append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)
```

This script provides a comprehensive solution for clustering app reviews and analyzing the results. By using this script, developers and product managers can gain valuable insights into user feedback and make data-driven decisions to improve their applications.