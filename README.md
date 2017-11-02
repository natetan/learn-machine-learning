# Learning Machine Learning
A repo dedicated to learning machine learning in both Python and R.

## Contents
- [Setup](#setup)
- [Importing the libraries](#importing-the-libraries)

### Setup
- Python
  - [Python Language](https://www.python.org/downloads/)
  - [Anaconda-Navigator (Spyder is the IDE)](https://www.anaconda.com/download/)
- R
  - [R programming language](https://cran.r-project.org/mirrors.html)
  - [R Studio](https://www.rstudio.com/products/rstudio/download/)

### General Code Styles
- R
  - Use `<-` for assignment
  - Use periods `.` to separate words in variable names
    - `my.variable <- 5`
- Python
  - Use `=` for assignment
  - Using underscores `_` to separate words in functions and variable names
    - `my_variable = 5`

### Running code in the console
This is the same for both R and Python. **Highlight** (or the entire file) the selected portion of 
the code you want to run, then hit `Command + Enter`

### Importing the libraries
**Python**  

```Python
import numpy as np                # Matematical tools
import matplotlib.pyplot as plot  # Plotting tools 
import pandas as pd               # Import and manage datasets
```

The keyword `as` lets us use custom shortcut names for out packages.

**R**  
```R
# Quotes required. This is only done once per machine, and is usually done in the console rather than the R script.
install.packages('packagename') 

# This calls the package being used and imports all the functions for that package.
library(packagename) 
```
R has most of these already built in, but there will be some packages in the future that we're going to need. This is how we're going to install everything.

### Setting the current working directory
This is a really important step. The file reader needs to know which directory to start as a reference. You can change that
in the settings. In **Spyder**, the default current working directory is set to whatever directory the file explorer is in. You can switch into the folder where the data is by clicking into it. You can manually set a folder location in the settings as well.

### Reading a Dataset
**Python**
```Python
# We have our working directory set where this data file is. Your directory may look different 
# depending on how you set yours up
dataset = pd.read_csv('Data.csv')
```

**R**
```R
dataset <- read.csv('Data.csv')
```

### Preparing Data
In Python, we have to distinguish between the matrix of features and the dependent variable vector. In our case, we need
the `Country`, `Age`, and `Salary` observations (our x variables).

```Python
# Preparing the data
# We need to separate the independent (x) variables from the dependent (y) ones
# Rows are on the left of the comma, and cols are on the right
# : Means range. num:num
x = dataset.iloc[:, :-1].values # We want all the rows, and every column exept the last one
y = dataset.iloc[:, 3].values # We want all the tows, but only the 3rd column
```
### Dealing With Missing Data
**Python**
```Python
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # looking for values that are NaN and replacing them with the mean
imputer = imputer.fit(x[:, 1:3]) # fixing all rows of cols 1 and 2
x[:, 1:3] = imputer.transform(x[:, 1:3]) # set the x data to the fixed table
```

**R**
```R
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
```
