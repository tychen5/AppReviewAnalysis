## Cluster Inference Pipeline

This Python script, `cluster_inference_pipeline.py`, is designed to perform cluster inferencing and sentiment analysis on a given text dataset. The script utilizes various libraries and pre-trained models to process the text, identify the language, and perform sentiment analysis. The main components of the script include language identification, sentiment analysis, and topic modeling.

This Python script, `cluster_inference_pipeline.py`, is designed to perform sentiment analysis, topic prediction, and clustering on input comments using various pre-trained models. The script processes input comments, extracts features, and computes distances to determine the most relevant topic and cluster for each comment.



### Dependencies

The script requires the following libraries:

- collections
- functools
- itertools
- json
- math
- operator
- os
- pickle
- random
- re
- requests
- string
- time

For NLP tasks, the following libraries are also required:
- nltk
- numpy
- pandas
- spacy
- stanza
- torch
- transformers
- scikit-learn
- pycountry
- spacy-langdetect

Python version 3.6 or higher is required.

### Key Features

1. **Language Identification**: The script uses the `pycountry`, `stanza`, and `spacy` libraries to identify the language of the input text. It employs a combination of methods to ensure accurate language identification.

2. **Sentiment Analysis**: The script uses the `transformers` library to perform sentiment analysis on the input text. It utilizes a pre-trained model (`xlm-roberta-base`) and an in-house API to analyze the sentiment of the text. The script calculates the average sentiment score from both methods.

3. **Topic Modeling**: The script employs a custom neural network model (`MyBert`) based on the `xlm-roberta-base` model to perform topic modeling. The model is trained to predict topics based on the input text.

4. **Text Preprocessing**: The script preprocesses the input text by tokenizing, removing stop words, and stemming the words using the `nltk` library.

5. **Parameter Loading**: The script loads various parameters from a JSON file, such as model paths, device configurations, and thresholds.

6. Preprocessing and sentiment analysis using XLM-RoBERTa and custom sentiment models.
7. Topic prediction and clustering using pre-trained models and custom algorithms.
8. Postprocessing and output formatting in JSON format.

### Usage

To use the `cluster_inference_pipeline.py` script, ensure that all required libraries are installed and the necessary files (such as the parameter JSON file and pre-trained models) are available.

1. Update the paths in the script to point to the correct locations of the required files.

2. Run the script using a Python interpreter (Python 3.6 or higher is recommended).

The script will process the input text, perform language identification, sentiment analysis, and topic modeling, and return the results.

To use the `cluster_inference_pipeline.py` script, import the necessary libraries and provide the input comments as a list. The script will return a JSON object containing the results of the analysis.

```python
from cluster_inference_pipeline import cluster_text_inference_inhouse

input_comment = ['bad application']
json_object = cluster_text_inference_inhouse(input_comment)
```

To use this script, simply import the required functions and call them with the appropriate input parameters. For example:

```python
from sentiment_analysis_pipeline import semtiment_analytic_inhouse1, cut_into_sentences

input_text = "Your input text here"
sentiment_results = semtiment_analytic_inhouse1(input_text)
sentences = cut_into_sentences(input_text, input_text, language_id)
```

To use this code snippet, you need to have the following dependencies installed:

- PyTorch
- Transformers
- NumPy
- Pandas
- Requests

You also need to provide the paths to your pre-trained models, tokenizers, and clustering data. Once you have set up the required dependencies and paths, you can use the `cluster_text_inference_inhouse` function to perform text clustering on your input data.
### Note

Please ensure that the paths to the required files (such as the parameter JSON file, pre-trained models, and stop words) are updated in the script before running it.


### Functions

The script contains several functions to perform various tasks:

- `extract_features_emb()`: Extracts features from the input comments using pre-trained models.
- `if_bad()`: Checks if a given numpy array is NaN.
- `calc_avg_dist()`: Calculates the average distance between the input comment's embeddings and the centroid and eigen vectors of a cluster.
- `compare_function()`: Compares the input comment's embeddings with the topic clusters and returns the most relevant topic and cluster.
- `cluster_text_inference_inhouse()`: Main function that performs sentiment analysis, topic prediction, and clustering on input comments.

1. `semtiment_analytic_inhouse1(body)`: This function takes an input text and performs sentiment analysis using the in-house API. It returns a tuple containing sentiment analysis results.

2. `nltk_cut(trans_text, ori_text=None, name_li=None)`: This function tokenizes the text using NLTK and returns the best number of sentences and the best sentences.

3. `cut_into_sentences(ori_text, trans_text, language_id)`: This function cuts the input text into sentences using different tokenizers and returns the best sentences.

4. `clean_text(text)`: This function cleans the input text by removing unnecessary characters and spaces.

5. `spacy_cut(trans_text, ori_text=None, name_li=None)`: This function tokenizes the text using spaCy and returns the best number of sentences and the best sentences.

6. `stanza_cut(trans_text, ori_text=None, name_li=None)`: This function tokenizes the text using Stanza and returns the best number of sentences and the best sentences.

7. `define_language(ori_text, language_id, spacy_multi2_doc)`: This function defines the language of the input text and returns the language information.

8. `xlm_sentiment_local(comment)`: This function calculates the XLM sentiment score for the input comment.

### Models

The script uses the following pre-trained models:

1. XLM-RoBERTa: A multilingual transformer model for sentiment analysis.
2. InfoXLM: A multilingual transformer model for feature extraction.
3. XLM-Align: A multilingual transformer model for feature extraction.
### Custom BERT Model

The `MyBert` class is a custom BERT model that inherits from `nn.Module`. It uses a pre-trained model and adds multilabel layers for classification.

### Output

The script returns a JSON object containing the results of the analysis, including sentiment scores, topic predictions, and cluster assignments for each input comment.
## Sentiment Analysis Pipeline

This Python script, is also designed to perform sentiment analysis on a given text using an in-house API and various NLP libraries such as NLTK, spaCy, and Stanza. The script tokenizes the input text into sentences, cleans the text, and processes it through the in-house API to obtain sentiment scores. It also calculates an average sentiment score using the in-house API and XLM sentiment scores.

## Multilingual Text Clustering Pipeline

This Python code snippet, is designed to perform text clustering on multilingual input data using pre-trained models and tokenizers. The code includes a custom BERT model, sentiment analysis, data preprocessing, feature extraction, and clustering functions.



### Sentiment Analysis

The code includes two sentiment analysis functions: `xlm_sentiment_local` and `sentiment_analytic_inhouse`. The first function is a placeholder for a local sentiment analysis implementation, while the second function sends a request to an external API for sentiment analysis.

### Data Preprocessing

The `preprocess` function tokenizes the input text, removes stop words, and applies stemming using the Porter Stemmer algorithm.

### Datasets

Two dataset classes, `DatasetTopic` and `Dataset`, are defined for handling input data. They inherit from `torch.utils.data.Dataset` and are used for processing and tokenizing the input text.

### Feature Extraction

The `extract_features_emb` function extracts features from the input text using multiple pre-trained models. It combines the embeddings from different models using a weighted average.

### Clustering

The `calc_avg_dist` and `compare_function` functions are used for clustering the input text based on the extracted features. The clustering is performed using a pre-defined set of topics and their associated centroids.

### Inference Function

The `cluster_text_inference_inhouse` function is the main entry point for the pipeline. It takes a list of input comments, applies sentiment analysis, preprocesses the text, extracts features, and performs clustering. The output is a JSON object containing the clustered text.

### Saving and Loading Models, Tokenizers, and Clustering Data

The code snippet also includes functions for saving and loading pre-trained models, tokenizers, and clustering data using the `pickle` module.

