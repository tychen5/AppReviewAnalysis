# App Review Clustering

This Python script is designed to perform clustering on app reviews using various clustering algorithms, such as HDBSCAN and DBSCAN. The script also calculates cluster validity scores and distances to centroids for each review. The main goal of this script is to group similar app reviews together, making it easier to analyze and understand user feedback.

### Dependencies

The script requires the following libraries:

- os
- pickle
- functools
- operator
- warnings
- random
- time
- gc
- requests
- json
- cupy
- numpy
- pandas
- matplotlib
- collections
- scipy
- sklearn
- cuml
- DBCV
- hdbscan
- tqdm
- kneed
- timeout_decorator

### Input Files

The script requires the following input files:

1. `used_params_file.pkl`: A pickle file containing the list of used parameters.
2. `labeled_df_file.pkl`: A pickle file containing the labeled DataFrame.
3. `df_full_comments_file.pkl`: A pickle file containing the full comments DataFrame.

### Output Files

The script generates a CSV file containing the sentence table with centroid distances.

### Functions

The script contains the following functions:

1. `need_topic(topics)`: Determines if a topic is needed based on a list of topics.
2. `hdbscan_scorer(eval_vec, eval_labels)`: Calculates the HDBSCAN score using DBCV for cosine and euclidean distances.
3. `hdbscan_scorer2(eval_vec, eval_labels)`: Calculates the HDBSCAN score using hdbscanpkg for cosine and euclidean distances.
4. `calc_scores(vectors, labels)`: Calculates various scores for the given vectors and labels.
5. `fit_algo(algo, X)`: Fits the given algorithm on the input data and returns the predicted labels.
6. `add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df)`: Adds centroid distances to the sentence table and saves it as a CSV file.

### Usage

To use this script, ensure that all required libraries are installed and input files are available. Update the file paths for input files and the output directory for the CSV file. Run the script, and the output CSV file containing the sentence table with centroid distances will be generated in the specified output directory.