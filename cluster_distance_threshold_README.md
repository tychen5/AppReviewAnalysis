## Cluster Distance Threshold

This Python script, `cluster_distance_threshold.py`, calculates the distance threshold for comment vectors in relation to centroid and eigen vectors of topic clusters. The script is designed to work with pre-processed data stored in a pickle file, and it outputs the calculated threshold value in another pickle file.

### Dependencies

- Python 3
- pandas
- numpy
- scikit-learn

### Input

The script requires a pickle file containing the following data structures:

- `comment_topic_di`: A dictionary with topic keys and DataFrames containing comment vectors for each topic.
- `topic_cluster_di`: A dictionary with topic keys and DataFrames containing cluster information for each topic.
- `topic_statistic_df`: A DataFrame containing topic statistics.
- `topic_di`: A dictionary containing topic information.

The path to the input pickle file should be specified in the `pkl_path` variable.

### Output

The script outputs a pickle file containing the calculated distance threshold value. The path to the output pickle file should be specified in the `file` parameter of the `pickle.dump()` function.

### How it works

1. The script loads the data from the input pickle file.
2. It creates a DataFrame with centroid and eigen vectors for each topic cluster.
3. It creates a DataFrame with comment vectors for each topic.
4. It calculates the distances between comment vectors and centroid/eigen vectors using cosine and Euclidean metrics.
5. It calculates the average distance for each comment vector.
6. It saves the threshold value as the minimum of the maximum average distance and the maximum of the individual distances.
7. The calculated threshold value is saved in a pickle file.

### Usage

To use the script, simply update the `pkl_path` variable with the path to your input pickle file and the `file` parameter of the `pickle.dump()` function with the path to your desired output pickle file. Then, run the script using a Python interpreter.

```bash
python cluster_distance_threshold.py
```

After the script has finished executing, the calculated distance threshold value will be saved in the specified output pickle file.