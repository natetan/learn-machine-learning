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
# : Means range. num:num
x = dataset.iloc[:, :-1].values # We want all the rows, and every column exept the last one
y = dataset.iloc[:, 3].values # We want all the tows, but only the 3rd column