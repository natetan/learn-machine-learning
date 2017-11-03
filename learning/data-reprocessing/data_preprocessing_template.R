# Yulong Tan
# 10.31.17

# Using libraries that are installed
# install.package('packagename') <- this is only done once per machine, and can be done within the console

# Importing the dataset
dataset <- read.csv('Data.csv')

# Taking care of missing data
ReplaceColWithAverage <- function(col) {
  # ifelse is basically a ternary operator
  new.col <- ifelse(is.na(col),
                    ave(col, FUN = function(x) mean(x, na.rm = TRUE)),
                    col)
  return(new.col)
}

FactorizeCol <- function(col, givenLevels, givenLabels) {
  new.col <- factor(col, levels = givenLevels, labels = givenLabels)
  return(new.col)
}

# $ symbol grabs the column by name
dataset$Age <- ReplaceColWithAverage(dataset$Age)
dataset$Salary <- ReplaceColWithAverage(dataset$Salary)

# Encoding categorical (nominal) data
dataset$Country <- FactorizeCol(dataset$Country, c('France', 'Spain', 'Germany'), c(1, 2, 3))
dataset$Purchased <- FactorizeCol(dataset$Purchased, c('No', 'Yes'), c(0, 1))

# Splitting the training set and the testing set
# Writing this in the console ensures that you don't write it here
# You install this once per machine
# install.packages('caTools') 

library(caTools)
set.seed(123)

# This uses the train set as the ratio (Python uses the test one, so we did 0.2)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
training.set <- subset(dataset, split == TRUE)
test.set <- subset(dataset, split == FALSE)

# Feature Scaling
## Need to exclude the 'factors', because they're not numeric.
## The factors are the categorical data that we transformed earlier
training.set[, 2:3] <- scale(training.set[, 2:3])
test.set[, 2:3] <- scale(test.set[, 2:3])

