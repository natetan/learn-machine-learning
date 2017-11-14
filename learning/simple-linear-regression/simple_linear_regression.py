"""
Yulong Tan
11.14.17
"""

# Simple Linear Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling (Most of the time, a library will do this for us)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting a Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_prediction = regressor.predict(X_test)

# Visualizing the training set results
plot.scatter(X_train, y_train, color = 'red')
plot.plot(X_train, regressor.predict(X_train), color = 'blue')
plot.title('Salary vs Experience (Training Set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

# Visualizing the test set results
plot.scatter(X_test, y_test, color = 'red')
plot.plot(X_train, regressor.predict(X_train), color = 'blue')
plot.title('Salary vs Experience (Test Set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()