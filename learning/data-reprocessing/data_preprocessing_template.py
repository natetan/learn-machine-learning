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
# looking for values that are NaN and replacing them with the mean
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 
imputer = imputer.fit(x[:, 1:3]) # fixing all rows of cols 1 and 2
x[:, 1:3] = imputer.transform(x[:, 1:3]) # set the x data to the fixed table

# Encoding categorical (nominal) data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_x = LabelEncoder()
x[:, 0] = le_x.fit_transform(x[:, 0])

le_y = LabelEncoder()
y = le_x.fit_transform(y)

ohe = OneHotEncoder(categorical_features = [0])
x = ohe.fit_transform(x).toarray()

# Splitting the dataset into a Training set and a Testing set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
## Recompute because we want it scaled
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.fit_transform(x_test)












