# Yulong Tan
# 11.14.17

# Simple Linear Regression

# Importing the dataset and libraries
library(ggplot2)
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
regressor <- lm(formula = Salary ~ YearsExperience,
                data = training_set)

# Predicting the test set results
y_prediction <- predict(regressor, newdata = test_set)

# Visualizing the training set results
ggplot() + 
  geom_point(mapping = aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') +
  geom_line(mapping = aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualizing the test set results
ggplot() + 
  geom_point(mapping = aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(mapping = aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')