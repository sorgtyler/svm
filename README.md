# svm
I used scikit-learn's support vector machine library to classify spammy-sounding emails. For training and testing, I used [Spambase] (https://archive.ics.uci.edu/ml/datasets/Spambase) from the UCI machine learning repository.  

##**Usage:**
To run the project, use `python svm.py` with the spambase.data file in the same path.

##**Dependencies:**
- [NumPy] (http://www.numpy.org) for calculations
- [sklearn] (http://scikit-learn.org/stable/) for their svm, cross_validation, and shuffle modules, as well as their metrics library for computing ROC and AUC.
- [SciPy] (https://www.scipy.org) for calculations
