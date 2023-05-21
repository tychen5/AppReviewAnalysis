# Sentiment Analysis

This Python script, `sentiment_analysis.py`, is designed to perform a basic sentiment analysis on a dataset containing sentiment values. The script reads in data from an external Excel file, processes the data using the pandas library, and outputs the count of occurrences for each sentiment value.

### Dependencies

The script requires the following Python packages:

- xlrd
- openpyxl
- numpy
- pandas

These packages can be installed using the following command:

```
pip install -U xlrd openpyxl numpy pandas
```

### Usage

1. Replace the `/path/to/your/data/review_export_report_All.xlsx` in the script with the actual path to your Excel file containing the sentiment data.
2. Run the script using the following command:

```
python sentiment_analysis.py
```

### Output

The script will display the column names of the dataset and the data itself. It will also output the count of occurrences for each sentiment value found in the 'Sentiment' column of the dataset.

### Code Overview

The script performs the following steps:

1. Installs the required packages (xlrd and openpyxl) if not already installed.
2. Imports the necessary libraries (numpy and pandas).
3. Sets display options for pandas to show all columns.
4. Reads in data from an external Excel file.
5. Prints the column names of the dataset.
6. Displays the dataset.
7. Counts the number of occurrences of each sentiment value in the 'Sentiment' column and outputs the result.

### Note

This script assumes that the input Excel file contains a column named 'Sentiment' with sentiment values. Please ensure that your dataset follows this format for the script to work correctly.