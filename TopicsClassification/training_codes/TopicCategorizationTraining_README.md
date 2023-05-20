# Topic Categorization Training

This Python script, `TopicCategorizationTraining.py`, is designed to train and evaluate a custom BERT model for topic categorization using multiple pre-trained models and datasets. The script preprocesses the data, defines the custom BERT model, tokenizes the input, creates train and test datasets, and trains and evaluates the model.

## Features

- Load and preprocess data from multiple sources (iOS, Android, and external)
- Define a custom BERT model for topic categorization
- Tokenize input using the `transformers` library
- Create train and test datasets using a custom `Dataset` class
- Train and evaluate the model using PyTorch and the `transformers` library
- Extract features from multiple pre-trained models and combine them using weighted averaging

## Dependencies

- pandas
- numpy
- pickle
- random
- transformers
- torch
- torchvision
- tqdm
- sklearn

## Usage

1. Replace the placeholders (e.g., "path/to/df_ios.pkl") with the appropriate paths to your data files.
2. Replace the placeholders (e.g., 'path/to/your/model1') with your actual paths and model names.
3. Run the script using Python 3.6 or later.

```bash
python TopicCategorizationTraining.py
```

## Customization

You can customize the script by modifying the following parameters:

- `tokenizer_paths`: List of paths to the pre-trained tokenizers
- `model_paths`: List of paths to the pre-trained models
- `device_li`: List of devices to use for each model (e.g., 'cuda:0', 'cuda:1', 'cpu')
- `weight_li`: List of weights to use for combining the features from multiple models

## Output

The script outputs the combined embeddings and ground truth labels for the test dataset. These can be used for further analysis and evaluation of the model's performance.