def eval_small_cluster(topic_di, smallcluster_scoredf):
    """
    Evaluate small clusters and update topic_di with the best algorithm index.

    Args:
        topic_di (dict): Dictionary containing topic information.
        smallcluster_scoredf (DataFrame): DataFrame containing small cluster scores.

    Returns:
        dict: Updated topic_di with the best algorithm index.
    """
    algo_idx = smallcluster_scoredf.mean()[smallcluster_scoredf.mean() == smallcluster_scoredf.mean().max()].index[0]

    for topic in all_topics:
        if topic not in small_topic_li:
            try:
                tmp = topic_di[topic]
            except KeyError:
                topic_di[topic] = tmp[tmp.index == algo_idx]

    return topic_di


def save_results(topic_di, output_dir_path, pickle_path):
    """
    Save the results of clustering to CSV files and pickle files.

    Args:
        topic_di (dict): Dictionary containing topic information.
        output_dir_path (str): Path to the output directory.
        pickle_path (str): Path to the pickle file.

    Returns:
        tuple: Tuple containing comment_table, cluster_table, and topic_statistic_df DataFrames.
    """
    # Initialize dictionaries and lists
    comment_topic_di = {}
    topic_cluster_di = {}
    cluster_stat_li = []
    calc_overall_score = []



    return comment_table, cluster_table, topic_statistic_df


# Main code
# ... (rest of the code)

# Loop through different parameter combinations
for _ in tqdm(range(take_num)):
    # ... (rest of the code)

    # Save results
    comment_bct, cluster_bct, topic_bct = save_results(finalcluster_di, output_dir_path, pkl_path)
    used_params_li.append(params)

