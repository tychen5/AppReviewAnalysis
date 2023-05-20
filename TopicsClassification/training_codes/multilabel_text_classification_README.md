# Multilabel Text Classification with XLM-RoBERTa

This Python script, `multilabel_text_classification.py`, is designed to perform multilabel text classification using the XLM-RoBERTa model. The code is structured to handle the preprocessing of text data, create a custom model architecture, and extract features from the model's hidden layers.

## Dependencies

The script requires the following libraries:

- pandas
- numpy
- pickle
- random
- transformers
- torch
- functools
- operator
- itertools
- os
- json
- glob
- gc
- math
- torchvision
- sklearn
- collections
- tqdm

## Custom Model Architecture

The `MyBert` class is a custom model architecture that inherits from `nn.Module`. It utilizes the pretrained XLM-RoBERTa model and adds a multilabel classification layer on top of it. The multilabel classification layer consists of a series of linear layers, activation functions (Mish), batch normalization, and dropout layers.

## Dataset Class

The `Dataset` class is a custom PyTorch dataset class that takes a DataFrame and a tokenizer as input. It processes the text data by tokenizing and encoding it, and returns the input tensors and one-hot encoded labels for each data point.

## Feature Extraction

The `extract_features_emb` function takes a list of dataloaders, models, devices, weights, and layer indices as input. It extracts features from the specified hidden layers of the models and computes a weighted average of the embeddings. The function returns the final embeddings and ground truth labels.

## Usage

To use this script, you need to provide a DataFrame containing the text data and labels, and instantiate the tokenizer and models. Then, create a custom dataset and dataloader using the provided `Dataset` class. Finally, call the `extract_features_emb` function to obtain the embeddings and ground truth labels.

Please note that you should replace the dummy values in the code with your actual values before running the script.

## License

This project is licensed under the MIT License.