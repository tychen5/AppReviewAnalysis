# Cluster Inference Loner Threshold

This Python script, `cluster_inference_loner_threshold.py`, calculates the average distance between comment embeddings and their corresponding cluster centroids and eigen vectors. It then determines a threshold value to identify "loner" comments, which are comments that have a higher average distance than the calculated threshold.

### Dependencies

- pandas
- numpy
- sklearn

### Input

The script requires a pickle file containing the following data structures:

- `comment_topic_di`: A dictionary containing comment data with topic keys.
- `topic_cluster_di`: A dictionary containing cluster data with topic keys.
- `topic_statistic_df`: A DataFrame containing topic statistics.
- `topic_di`: A dictionary containing topic information.

### Output

The script outputs a pickle file containing the final distance threshold value.

### How it works

1. The script first loads the required data from the input pickle file.
2. It then prepares the data by extracting relevant columns from the input data structures and concatenating them into two DataFrames: `all_centroid_eigen_vector_df` and `all_comment_vector_df`.
3. The `calc_distance` function calculates the cosine and Euclidean distances between a comment's embedding and its corresponding cluster centroid and eigen vector.
4. The script applies the `calc_distance` function to all comments in `all_comment_vector_df` and stores the resulting distances in new columns.
5. It calculates the average distance for each comment and finds the maximum average distance across all comments.
6. The final distance threshold is determined as the minimum of the maximum average distance and the maximum of the individual distances.
7. The script saves the final distance threshold value to an output pickle file.
8. The `calc_avg_dist` function calculates the average distance between a comment's embedding and its corresponding cluster centroid and eigen vector, considering the loner distance threshold.

### Usage

To use this script, simply update the input and output file paths and run the script. The output pickle file will contain the final distance threshold value, which can be used to identify "loner" comments in the dataset.