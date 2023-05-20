# Unifi Data Preprocessing and Analysis

This Python script, `unifi_data_preprocessing_and_analysis.py`, is designed to preprocess and analyze data from various sources related to the Unifi Network Controller. The script imports data from multiple file formats, cleans and processes the data, and performs some basic analysis.

### Dependencies

The script requires the following Python libraries:

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

### Data Import

The script imports data from six different sources, including CSV, Pickle, and Excel files. The file paths for these data sources should be updated to match the user's local file paths.

### Data Preprocessing

The script performs several preprocessing steps on the imported data:

1. It removes columns with only one unique value.
2. It cleans the `Topics` column by removing unnecessary characters and replacing them with underscores.
3. It cleans the `ExternalData` column by converting it to a list.
4. It appends the external data to the main dataframes.
5. It resets the topics in the Android and iOS dataframes by removing any topics not found in the external data.
6. It translates the `message` column to English using the Google Translate API.

### Data Export

After preprocessing, the cleaned dataframes are saved as Pickle files for further analysis.

### Data Analysis

The script includes some basic data analysis, such as calculating the frequency of topics and visualizing the data using matplotlib. Users can expand on this section to perform more advanced analysis as needed.

### Usage

To use the script, simply update the file paths for the data sources and run the script. The cleaned dataframes will be saved as Pickle files, which can be loaded for further analysis.

**Note:** The Google Translate API may require an API key and may have usage limits. Be sure to check the API documentation for details on usage and limitations.