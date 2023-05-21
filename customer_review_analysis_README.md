# Customer Review Analysis

This Python script, `customer_review_analysis.py`, is designed to analyze and process customer reviews from various sources, including Android, iOS, and external platforms. The script cleans and translates the reviews, extracts relevant topics, and calculates satisfaction scores for further analysis and visualization.

### Dependencies

The script requires the following libraries:

- pandas
- numpy
- googletrans
- functools
- operator
- ast
- time
- random
- collections
- matplotlib
- scipy

### Functions

The script contains the following functions:

- `clean_topics(textstr)`: Cleans the text string by removing unnecessary characters and replacing them with appropriate ones.
- `clean_externaldata(strli)`: Cleans the external data by converting it into a list and handling empty lists.
- `reset_topic(oristr)`: Resets the topic by removing unnecessary characters and checking for unknown words.
- `translate2en(textstr, country)`: Translates the text string to English using the Google Translate API.
- `clean_time(t_obj)`: Cleans the time object by converting it to a string and keeping only the date part.
- `satis_func(topicli)`: Calculates the satisfaction score based on the presence of "Satisfied users" or "Dissatisfied users" in the topic list.

### Data Processing

The script performs the following data processing steps:

1. Load data from various sources (CSV, Excel, and Pickle files).
2. Filter out columns with only one unique value.
3. Clean and process the topics in the data.
4. Translate the messages to English using the Google Translate API.
5. Save the cleaned dataframes to pickle files for further analysis.

### Analysis and Visualization

After processing the data, the script appends the cleaned dataframes and filters out rows with unknown topics. It then calculates the number of topics, cleans the time data, and computes the satisfaction scores. Users can perform further analysis and visualization as needed.

### Usage

To use this script, replace the file paths in the code with your own data file paths. After running the script, you can load the cleaned dataframes from the saved pickle files and perform further analysis and visualization as needed.