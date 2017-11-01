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

### Running code in the console
This is the same for both R and Python. **Highlight** (or the entire file) the selected portion of 
the code you want to run, then hit `Command + Enter`

