# Text Cluster Inference

This Python script, `text_cluster_inference.py`, is designed to perform text clustering and sentiment analysis on a given input text. The script utilizes various natural language processing (NLP) libraries and techniques, such as Spacy, Stanza, NLTK, and Transformers, to process and analyze the input text.

## Features

- Language identification and processing using Spacy, Stanza, and Pycountry
- Sentiment analysis using Transformers and custom sentiment analysis functions
- Text preprocessing and tokenization using NLTK and custom functions
- Text clustering using pre-trained models and custom functions
- Feature extraction using pre-trained models and custom functions

## Dependencies

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
- collections
- nltk
- numpy
- pandas
- pycountry
- spacy
- stanza
- torch
- transformers

## Usage

To use the `text_cluster_inference.py` script, you need to provide the required paths and variables in the script. Replace the following paths and variables with your own values:

- sentiment_model_path
- sentiment_bound_path
- loner_bound_path
- spacy_libnames_path
- model1_path
- encoding_path
- xlmr_model_path
- xlmr_tok_path
- model2_path
- model2_tok
- model3_path
- model3_tok
- model4_path
- model4_tok
- stop_words_path
- base_cluster_path

After setting the required paths and variables, you can call the `cluster_text_inference_inhouse(input_comment)` function with your input text to perform text clustering and sentiment analysis.

## Output

The output of the `cluster_text_inference_inhouse(input_comment)` function is a JSON object containing the following information:

- Original input text
- Translated text (if applicable)
- Sentiment type and score
- Language of the text
- Cluster ID
- Centroid distance

This information can be used for further analysis and processing of the input text.