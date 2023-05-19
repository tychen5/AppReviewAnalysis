# Multilingual NLP Libraries Setup

This Python script, `multilingual_nlp_libraries_setup.py`, automates the process of downloading and setting up various Natural Language Processing (NLP) libraries and models for multiple languages. The script focuses on three popular NLP libraries: SpaCy, Stanza, and NLTK, as well as transformer models from the Hugging Face Transformers library.

### SpaCy

The script downloads and sets up SpaCy language models for the following languages:

- Catalan (ca)
- Chinese (zh)
- Danish (da)
- Dutch (nl)
- English (en)
- French (fr)
- German (de)
- Greek (el)
- Italian (it)
- Japanese (ja)
- Lithuanian (lt)
- Macedonian (mk)
- Norwegian Bokmål (nb)
- Polish (pl)
- Portuguese (pt)
- Romanian (ro)
- Russian (ru)
- Spanish (es)

It attempts to download models with different sizes (lg, md, sm, trf) and types (core_news, dep_news, core_web).

### Stanza

The script downloads and sets up Stanza language models for multiple languages, including a multilingual model. The full list of languages can be found in the `stanza_lang` variable within the script.

### NLTK

The script downloads and sets up popular NLTK data packages, as well as all available NLTK data packages.

### Hugging Face Transformers

The script loads transformer models and tokenizers from the Hugging Face Transformers library, including:

- xlm-roberta-base
- cardiffnlp/twitter-xlm-roberta-base
- microsoft/infoxlm-base
- microsoft/xlm-align-base

Additionally, it sets up a sentiment analysis pipeline using the `cardiffnlp/twitter-xlm-roberta-base-sentiment` model.

### Usage

To use this script, simply run it in your Python environment. Make sure to update the path to your `parameter.json` file, which should contain the `spacy_libnames_path` key with the appropriate value.

```python
python multilingual_nlp_libraries_setup.py
```

### Dependencies

This script requires the following Python libraries:

- spacy
- stanza
- nltk
- transformers
- torch
- numpy
- pandas
- json
- pickle
- gc (garbage collector)

Make sure to install these libraries before running the script.