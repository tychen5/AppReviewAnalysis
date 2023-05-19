## App Review Clustering

This Python script is designed to process and analyze app reviews by clustering them into meaningful topics. The script uses various machine learning algorithms and natural language processing techniques to achieve this goal.

### Key Features

- Load and preprocess labeled app review data
- Create a custom embedder class for text embeddings
- Use BERTopic for topic modeling
- Evaluate and refine topic clusters
- Save and output results in CSV format

### Dependencies

- numpy
- pandas
- cupy
- cuml
- sklearn
- DBCV
- scipy
- timeout_decorator
- hdbscan
- kneed
- torch
- bertopic
- sentence_transformers

### Usage

1. Update the paths for `labeled_df_path`, `param_path`, `csv_output_dir`, and `precompute_duplicate_path` with your own file paths.
2. Add your own logic for processing `labeled_df` and other data.
3. Run the script to perform app review clustering and generate output files.

### Key Functions

- `CustomEmbedder`: A custom class for creating text embeddings using a specified embedding model.
- `search_metric_weight`: Searches for the best metric weights in top N topics to find base parameter sets (in-topic).
- `search_metric_weight_smalltopics`: Uses big topics' algorithm to evaluate in small topics.
- `eval_small_cluster`: Evaluates small clusters and updates the topic dictionary.
- `save_results`: Saves the results of the clustering process in CSV format.
- `remapping_sentence_table`: Remaps the sentence table based on the same index dictionary.
- `map_sentiment`: Maps sentiment scores and types to sentences and comments.
- `add_centroid_dist`: Adds centroid distances to the sentence table.
- `refine_statistics`: Recalculates statistics by grouping by sentence due to remapping and adds centroid sentences.
- `append_filtered_commsent`: Appends filtered comments and sentences to the sentence table.

### Output

The script generates the following output files:

- `comment_table.csv`: A table containing information about each comment and its associated topic and cluster.
- `cluster_table.csv`: A table containing information about each cluster, including size, ratio, and centroid sentence.
- `topic_statistic_df.csv`: A table containing statistics for each topic, such as loner size and loner ratio.
- `sentence_table.csv`: A table containing information about each sentence, including topic, cluster, and distances.

### Note

Please ensure that you have the necessary dependencies installed and that you update the file paths in the script before running it.