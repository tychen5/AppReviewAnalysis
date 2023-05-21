# Multilabel Topic Categorization

This Python code snippet, `multilabel_topic_categorization.py`, is designed to perform multilabel topic categorization on a dataset using a custom BERT model and other pre-trained models. The code is divided into two main sections: data preprocessing and model training/evaluation.

### Data Preprocessing

The data preprocessing section includes the following steps:

1. Load data from pickled files (iOS, Android, and external data).
2. Preprocess the data by removing 'UNK' (unknown) topics and appending a small number of 'UNK' samples.
3. Count the occurrences of each topic in the dataset.
4. Split the data into train and test sets.
5. Save the train and test data as pickled files.

### Model Training and Evaluation

The model training and evaluation section includes the following steps:

1. Import necessary libraries and set environment variables.
2. Count the occurrences of each topic in the train data.
3. Encode the topics using a dictionary.
4. Calculate positive weights for each topic.
5. Load the data and model.
6. Define a custom BERT model class, `MyBert`, which inherits from `nn.Module`.
7. Define a custom dataset class, `CustomDataset`, which inherits from `torch.utils.data.Dataset`.
8. Initialize tokenizers, datasets, and dataloaders for each model.
9. Load the custom BERT model and other pre-trained models.
10. Set device and weights for each model.
11. Extract features and embeddings from the models.
12. Calculate the final embeddings and ground truth labels.

Please ensure that you replace the placeholders (e.g., 'path/to/your/df_ios.pkl') with the actual paths to your data files and model names.

### Dependencies

- pandas
- numpy
- pickle
- random
- transformers
- torch
- functools
- operator
- collections
- os
- gc
- torchvision
- tqdm
- sklearn

### Usage

To use this code snippet, simply run the `multilabel_topic_categorization.py` script after replacing the placeholders with the appropriate paths and model names. The script will preprocess the data, train the custom BERT model, and evaluate the model using the extracted features and embeddings.