# Comment Vector Feature Extraction

This Python script, `comment_vector_feature_extraction.py`, is designed to extract features from comments and perform multi-label classification using a custom BERT model. The script also includes functions to calculate various clustering scores for the extracted features.

### Dependencies

- Python 3.6+
- PyTorch
- Transformers
- NumPy
- Pandas
- Pickle
- HDBSCAN (for clustering scores)

### Overview

The script consists of three main parts:

1. A custom BERT model class (`MyBert`) for multi-label classification.
2. A custom dataset class (`CustomDataset`) for loading and processing the input data.
3. Functions for extracting features, calculating clustering scores, and saving the results.

#### MyBert

The `MyBert` class is a custom BERT model for multi-label classification. It is built on top of the `xlm-roberta-base` model from the Transformers library and includes additional layers for multi-label classification.

#### CustomDataset

The `CustomDataset` class is a custom dataset class for loading and processing the input data. It takes a DataFrame and a tokenizer as input and returns tokenized input data and attention masks for each comment.

#### Functions

The script includes several functions for extracting features, calculating clustering scores, and saving the results:

- `calc_scores`: Calculates various clustering scores for the given vectors and labels.
- `extract_features_emb`: Extracts features from the input data using the specified models and dataloaders.

### Usage

1. Replace the placeholders in the script with your own paths and information as needed.
2. Run the script to extract features and calculate clustering scores.
3. Save the results to the specified output paths.

### Output

The script generates the following output files:

- `comment_table.csv`: A table containing the extracted features for each comment.
- `cluster_table.csv`: A table containing the clustering scores for each comment.
- `topic_statistic.csv`: A table containing various topic statistics.