# App Review Clustering
The main goal is to identify and group similar reviews based on their content, which can help developers and product managers understand user feedback and improve their applications accordingly.

### Features

- Loads and preprocesses app review data
- Removes unnecessary columns and filters topics
- Calculates various clustering evaluation scores
- Performs clustering using HDBSCAN and DBSCAN algorithms
- Saves clustering results to CSV files
- Refines and updates cluster statistics
- Appends filtered comments and sentences to the sentence table

### Dependencies

- Python 3.6+
- NumPy
- pandas
- CuPy
- matplotlib
- timeout_decorator
- hnswlib
- tqdm
- kneed
- scikit-learn
- SciPy
- cuML
- DBCV
- hdbscan
- torch

### Usage

1. Update the following paths in the script with your own paths:

   - `used_params_path`
   - `labeled_df_path`
   - `df_full_comments_path`
   - `csv_output_dir`
   - `precompute_duplicate_path`

2. Run the script:

   ```
   python AppReviewClustering_v0_11.py
   ```

3. The script will perform clustering on the app reviews and save the results to the specified output directory.

### Functions

- `need_topic(topics)`: Determines if a topic should be included in the analysis.
- `calc_scores(vectors, labels)`: Calculates various clustering evaluation scores for the given vectors and labels.
- `save_results(topic_di, output_dir_path, pickle_path)`: Saves clustering results to CSV files and a pickle file.
- `convert_str(li)`: Converts a list of numbers to a formatted string.
- `refine_statistics(sentence_table, cluster_table_path, topic_table_path)`: Recalculates statistics by grouping by sentence due to remapping and adds centroid sent.
- `append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)`: Appends filtered comments and sentences to the sentence table.

### Output

The script generates the following output files:

- `sentence_table.csv`: A table containing sentences and their associated topic and cluster IDs.
- `cluster_table.csv`: A table containing cluster statistics and information.
- `topic_table.csv`: A table containing topic statistics and information.
- `*.pkl`: A pickle file containing the clustering results and related data structures.