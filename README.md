# Bosch Production Line Performance 

### Introduction  
This repository contains teh source code and other helper files for the group project of course CZ4041 - Machine Learning at Nanyang Technological University, Singapore. 

It is about a closed kaggle competition - Bosch Production Line Performance. 

### Prerequisite  
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [XGBoost](https://github.com/dmlc/xgboost)
* [Seaborn](https://seaborn.pydata.org/index.html)

### Pre-processing 
The pre-processing of features are in `clean_slate.py`, `count_dup.py`, `datatime_feat.py`, `id_feat.py`, `kurtosis.py` and `station_marker.py` files. 

### Running the tests 
The best performance is generated from XGBoost model in `XGBoostModel.py` file. 

### Acknowledgement 
All models were tested on kaggle [datasets](https://www.kaggle.com/c/bosch-production-line-performance/data). 
