#!/usr/bin/env python
# coding: utf-8

# Install required packages
#!pip install -U xlrd openpyxl

# Import necessary libraries
import numpy as np
import pandas as pd

# Set display options for pandas
pd.set_option("display.max_columns", None)

# Read in data from external sources
all_outsource_data = pd.read_excel('/path/to/your/data/review_export_report_All.xlsx')

# Print column names and display data
print(all_outsource_data.columns)
all_outsource_data

# Count the number of occurrences of each sentiment value
all_outsource_data['Sentiment'].value_counts()