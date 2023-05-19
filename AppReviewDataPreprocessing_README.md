# App Review Data Preprocessing

This Python script preprocesses app review data, cleans the text, and performs sentiment analysis and topic modeling. The script is designed to work with app review data from various sources, such as CSV files, and combines them into a single dataframe for further analysis.

### Features

- Load data from multiple CSV files and combine them into a single dataframe
- Remove duplicate reviews
- Clean and preprocess review text
- Perform sentiment analysis on review text
- Detect and handle multiple languages in reviews
- Perform topic modeling using pre-trained models
- Save and load dataframe checkpoints
- Utilize various NLP libraries and models for text processing

### Requirements

- Python 3.6 or higher
- pandas
- numpy
- transformers
- torch
- stanza
- spacy
- pycountry
- nltk
- gc
- requests
- json
- random
- time
- pickle
- os

### Usage

1. Update the file paths in the script to point to your input CSV files and desired output locations.
2. Ensure that all required libraries are installed.
3. Run the script to preprocess the app review data and perform sentiment analysis and topic modeling.

### Output

The script will output a cleaned and preprocessed dataframe containing the app review data, along with sentiment analysis and topic modeling results. This dataframe can be saved as a pickle file or exported to an Excel file for further analysis.

### Note

Please ensure that the input CSV files are formatted correctly and contain the necessary columns for the script to function properly. Additionally, make sure to update the file paths in the script to match your local environment.