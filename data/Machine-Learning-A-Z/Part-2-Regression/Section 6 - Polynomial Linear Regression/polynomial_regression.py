"""
Yulong Tan
12.01.17
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Use 1:2 instead of just 1, because we want our x to be a matrix, not a vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# For this one, we don't want to split the sets because we have very little data
# We want to make the most accurate prediction, therefore we need to use the
# whole dataset to train the model


 # We want to compare linear reg and polynomial reg
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the Linear Regression results
plot.scatter(X, y, color = 'red')
plot.plot(X, lin_reg.predict(X), color = 'blue')
plot.title('Position Salaries (Linear Regression)')
plot.xlabel('Position Level')
plot.ylabel('Salary')
plot.show()

# Visualizing the Polynomial Regression
plot.scatter(X, y, color = 'red')
plot.plot(X, lin_reg2.predict(X_poly), color = 'blue')
plot.title('Position Salaries (Polynomial Regression)')
plot.xlabel('Position Level')
plot.ylabel('Salary')
plot.show()