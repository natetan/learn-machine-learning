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

# $ symbol grabs the column by name
dataset$Age <- ReplaceColWithAverage(dataset$Age)
dataset$Salary <- ReplaceColWithAverage(dataset$Salary)