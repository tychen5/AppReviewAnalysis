# Text Analysis Pipeline

This Python script performs various natural language processing tasks on input text data, including sentiment analysis, text preprocessing, feature extraction, topic modeling, and clustering. It utilizes pre-trained models and tokenizers from popular NLP libraries such as Spacy, Stanza, NLTK, and Hugging Face Transformers, as well as custom models and functions for specific tasks.

The script provides customizable options for parameter settings and output formats, making it versatile for different use cases.
## Key Features

1. Language identification and processing using Spacy, Stanza, and Pycountry libraries.
2. Text preprocessing, including tokenization, stopword removal, and stemming using NLTK library.
3. Sentiment analysis using pre-trained models from the Transformers library.
4. Text clustering using a custom BERT model `MyBert` and XLM-RoBERTa tokenizer.
5. Calculation of various metrics, such as mean absolute error, precision-recall-fscore, etc.
6. Custom `Dataset` class for handling data and tokenization.
7. Pre-trained models from Hugging Face's Transformers library, including XLM-RoBERTa, InfoXLM, and XLM-Align.
8. Functions for sentiment analysis using XLM model and in-house API.
9. Text preprocessing and tokenization using NLTK, SpaCy, and Stanza libraries.
10. Feature extraction using pre-trained models and combining embeddings with weights.

## Dependencies
The following Python packages are required to use this code:
* collections
* functools
* itertools
* json
* math
* operator
* os
* pickle
* random
* re
* requests
* string
* time
### Natural Language Processing Dependencies
The following Python packages are used for natural language processing tasks:
* NLTK
* SpaCy
* Stanza
### Machine Learning Dependencies
The following Python packages are used for machine learning tasks:
* NumPy
* Pandas
* Scikit-learn
* PyTorch
* Transformers
* Hugging Face Transformers
### Other Dependencies
The following Python packages are also required:
* Pycountry
* Pickle
* JSON
* Note: This code requires Python 3.6 or later.

## Usage

Before running the 
cluster_inference_pipeline.py
 script, make sure that you have installed all the required dependencies and set up the necessary files and paths correctly. The following files and paths are needed:

* PARAMETER_JSON_PATH
: Path to the JSON file containing various parameters for the script.

* SPACY_LIBNAMES_PATH
: Path to the Spacy library names file.
* SENTIMENT_MODEL_PATH
: Path to the pre-trained sentiment model.
* SENTIMENT_BOUND_PATH
: Path to the sentiment inference threshold file.
* LONER_BOUND_PATH
: Path to the clustering inference threshold file.
* STOP_WORDS_PATH
: Path to the stop words file.
* BASE_CLUSTER_PATH
: Path to the base clusters file.
### Running the script
Once all the required files and paths have been set up, you can run the script to perform text clustering and sentiment analysis on your input dataset.

To use this script, you can either run it as a standalone Python script or import it as a module in other projects.

### Using the script
To use the script, you must have the required pre-trained models, tokenizers, and data files in the specified paths. To perform topic modeling and sentiment analysis on your input text, call the 
cluster_text_inference_inhouse()
 function with your input text:
```
import cluster_inference_pipeline as cip

input_comment = "Your input text here"
result = cip.cluster_text_inference_inhouse(input_comment)
print(result)
```
### Output

The script will output the following:

- A list of bad comments that could not be processed.
- Sentiment analysis results for each comment in the dataset.
- Cluster assignments for each comment in the dataset.

### Customization

To customize the script for your specific use case, you can modify the parameters in the JSON file or adjust the functions and variables within the script. For example, you can change the pre-trained sentiment model, the clustering model, or the text preprocessing steps.

### Functions

The script contains several functions for various NLP tasks:

- `extract_features_emb()`: Extracts features from the input data using pre-trained models and combines the embeddings with weights.
- `if_bad()`: Checks if a given numpy array is NaN.
- `xlm_sentiment_local()`: Performs sentiment analysis using the XLM model.
- `chk_null()`: Checks if a given message is null, empty, or NaN.
- `semtiment_analytic_inhouse1()`: Performs sentiment analysis using an in-house API.
- `nltk_cut()`: Tokenizes the text using NLTK and returns the best number of sentences and the best sentences.
- `cut_into_sentences()`: Cuts the text into sentences using different tokenizers and returns the best sentences.

### Example

```python
from cluster_inferencing_pipeline import Dataset, extract_features_emb, xlm_sentiment_local, chk_null, semtiment_analytic_inhouse1, nltk_cut, cut_into_sentences

# Load your data and tokenizer
data = ...
tokenizer = ...

# Create a Dataset object
dataset = Dataset(data, tokenizer)

# Perform feature extraction
features = extract_features_emb(dataloader_li, model_li, device_li, weight_li)

# Perform sentiment analysis using XLM model
label, score = xlm_sentiment_local(comment)

# Check if a message is null, empty, or NaN
flag, cmt, take = chk_null(msg)

# Perform sentiment analysis using in-house API
results = semtiment_analytic_inhouse1(body)

# Tokenize the text using NLTK
best_num, best_sent = nltk_cut(trans_text, ori_text, name_li)

# Cut the text into sentences using different tokenizers
sentences = cut_into_sentences(ori_text, trans_text, language_id)
```

### Note

Please make sure to update the file paths and API URLs in the script according to your project structure and requirements.
Please ensure that you have the required pre-trained models, tokenizers, and data files in the specified paths before running the script.

### Key Components

1. **MyBert**: A custom PyTorch model class that extends `nn.Module`. It uses a pre-trained model and adds multi-label layers for topic modeling.

2. **DatasetTopic**: A custom PyTorch dataset class for handling input text data.

3. **Preprocessing Functions**: Functions for cleaning and tokenizing input text, including `preprocess()` and `word_tokenize()`.

4. **Sentiment Analysis Functions**: Functions for performing sentiment analysis using both an in-house API and a local implementation, including `xlm_sentiment_local()` and `sentiment_analytic_inhouse()`.

5. **Feature Extraction**: Functions for extracting features from input text using pre-trained models, including `extract_features_emb()`.

6. **Clustering Functions**: Functions for clustering input text based on extracted features, including `calc_avg_dist()` and `compare_function()`.

7. **Main Inference Function**: The `cluster_text_inference_inhouse()` function, which takes an input text and performs topic modeling and sentiment analysis, returning a JSON object with the results.





