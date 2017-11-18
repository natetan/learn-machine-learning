# Yulong Tan
# 11.16.17

# Multiple Linear Regression

# Importing the dataset
dataset <- read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State <- factor(dataset$State,
                        levels = c('New York', 'California', 'Florida'),
                        labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split <- sample.split(dataset$Profit, SplitRatio = 0.8)
training.set <- subset(dataset, split == TRUE)
test.set <- subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Multiple Linear Regression to the training set
# regressor <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
regressor <- lm(formula = Profit ~ .,
                data = training.set) 

# Predicting the test set results
y.prediction <- predict(regressor, newdata = test.set)

# Building the optimal model using Backward Elimination
regressor <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regressor)

regressor <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regressor)

regressor <- lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regressor)

regressor <- lm(formula = Profit ~ R.D.Spend,
                data = dataset)
summary(regressor)









