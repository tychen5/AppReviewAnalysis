# Language Processing Libraries Downloader

This Python script, `language_processing_libraries_downloader.py`, automates the process of downloading and installing various language processing libraries and models for the following libraries:

- Spacy
- Stanza
- NLTK

The script is designed to be run once to set up the environment with the necessary libraries and models for further natural language processing tasks.

### Features

1. Downloads and installs Spacy models for multiple languages and variations (core news, dependency news, core web) in different sizes (large, medium, small, transformer).
2. Downloads and installs Stanza models for multiple languages.
3. Downloads and installs popular and all NLTK models.

### Usage

To use the script, simply run the `language_processing_libraries_downloader.py` file in your Python environment. The script will automatically download and install the required libraries and models.

### Dependencies

The script requires the following Python libraries to be installed:

- Spacy
- Stanza
- NLTK

You can install these libraries using pip:

```bash
pip install spacy stanza nltk
```

### Configuration

The script reads parameters from a JSON file named `parameter.json`. This file should be located in the same directory as the script. The JSON file should contain a dictionary with the necessary parameters.

### Output

The script will output the list of successfully installed Spacy libraries and print the number of Stanza languages downloaded. In case of any errors during the download process, the script will print the language code that caused the error and continue with the remaining languages.

### Note

Please ensure that you have a stable internet connection and sufficient disk space before running the script, as it downloads a large number of models and libraries.