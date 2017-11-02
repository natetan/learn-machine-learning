#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yulong Tan
10.31.17
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Importing the datasets
dataset = pd.read_csv('Data.csv')

# Preparing the data
# We need to separate the independent (x) variables from the dependent (y) ones
# Rows are on the left of the comma, and cols are on the right
# : Means range. num:num (upper bound is excluded)
x = dataset.iloc[:, :-1].values # We want all the rows, and every column exept the last one
y = dataset.iloc[:, 3].values # We want all the tows, but only the 3rd column

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # looking for values that are NaN and replacing them with the mean
imputer = imputer.fit(x[:, 1:3]) # fixing all rows of cols 1 and 2
x[:, 1:3] = imputer.transform(x[:, 1:3]) # set the x data to the fixed table