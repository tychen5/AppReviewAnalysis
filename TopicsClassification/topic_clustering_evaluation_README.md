# Topic Clustering Evaluation

This Python script, `topic_clustering_evaluation.py`, is designed to evaluate and save the results of topic clustering algorithms. The script contains two main functions: `eval_small_cluster` and `save_results`. These functions are used to update the topic dictionary with the best algorithm index and save the clustering results to CSV and pickle files, respectively.

### Functions

#### 1. eval_small_cluster(topic_di, smallcluster_scoredf)

This function evaluates small clusters and updates the topic dictionary with the best algorithm index.

**Arguments:**

- `topic_di` (dict): A dictionary containing topic information.
- `smallcluster_scoredf` (DataFrame): A DataFrame containing small cluster scores.

**Returns:**

- `dict`: The updated topic dictionary with the best algorithm index.

#### 2. save_results(topic_di, output_dir_path, pickle_path)

This function saves the results of clustering to CSV files and pickle files.

**Arguments:**

- `topic_di` (dict): A dictionary containing topic information.
- `output_dir_path` (str): The path to the output directory.
- `pickle_path` (str): The path to the pickle file.

**Returns:**

- `tuple`: A tuple containing `comment_table`, `cluster_table`, and `topic_statistic_df` DataFrames.

### Usage

To use this script, replace the paths and other sensitive information with your own values. Then, run the script to evaluate the topic clustering algorithms and save the results.

### Dependencies

- pandas
- tqdm

### Example

```python
from topic_clustering_evaluation import eval_small_cluster, save_results

# ... (load your data and preprocess it)

# Evaluate small clusters
updated_topic_di = eval_small_cluster(topic_di, smallcluster_scoredf)

# Save results
comment_table, cluster_table, topic_statistic_df = save_results(updated_topic_di, output_dir_path, pickle_path)
```

### Note

Please ensure that all necessary dependencies are installed and that the input data is properly formatted before running the script.