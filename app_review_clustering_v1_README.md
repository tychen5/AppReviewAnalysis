## App Review Clustering

This Python script, is designed to perform clustering on app reviews using various clustering algorithms and techniques. The primary goal is to identify and group similar reviews, allowing for easier analysis and understanding of user feedback.

### Dependencies

The script requires the following libraries:

- os
- functools
- operator
- warnings
- random
- time
- gc
- pickle
- requests
- json
- numpy
- pandas
- matplotlib
- collections
- scipy
- sklearn
- cuml
- DBCV
- hdbscan
- kneed
- timeout_decorator
- tqdm

### Overview

The script consists of several functions that work together to perform clustering on app reviews. The main steps include:

1. Loading labeled data and used parameters.
2. Preprocessing the data by removing unnecessary columns.
3. Saving the results of the clustering process.
4. Processing each topic and creating tables for the results.
5. Formatting the final score, cluster number, and cluster size.
6. Creating output directories and saving tables to CSV files.
7. Remapping the sentence table for visualization purposes.
8. Mapping sentiment values to the original sentences.
9. Adding centroid distances to the final output.

### Functions

The script contains the following functions:

- `save_results()`: Saves the clustering results, including comment topics, topic clusters, cluster statistics, and overall scores.
- `process_topic()`: Processes each topic, updating comment topics, topic clusters, cluster statistics, and overall scores.
- `create_tables()`: Creates tables for comment topics, topic clusters, and topic statistics.
- `format_final_score()`: Formats the final score for output.
- `format_cluster_num()`: Formats the cluster number for output.
- `format_cluster_size_total()`: Formats the total cluster size for output.
- `create_output_directory()`: Creates an output directory if it does not already exist.
- `save_tables_to_csv()`: Saves comment, cluster, and topic statistic tables to CSV files.
- `convert_str()`: Converts a list of numbers to a string.
- `remapping_sentence_table()`: Remaps the sentence table for visualization purposes.
- `map_sentiment()`: Maps sentiment values to the original sentences.
- `add_centroid_dist()`: Adds centroid distances to the final output.

### Usage

To use this script, ensure that you have the required dependencies installed and provide the necessary input files, such as labeled data and used parameters. The script will then perform clustering on the app reviews and save the results in the specified output directories.

### Output

The output of this script includes:

- Comment table: A table containing the clustered comments.
- Cluster table: A table containing information about the clusters.
- Topic statistic table: A table containing statistics about the topics.
- CSV files: The tables are saved as CSV files in the specified output directory.
- Pickle file: A pickle file containing the comment topics, topic clusters, topic statistics, and topic dictionary.