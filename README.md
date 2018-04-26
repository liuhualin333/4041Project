# Bosch Production Line Performance 

### Introduction  
This repository contains the source code and other helper files for the group project of course CZ4041 - Machine Learning at Nanyang Technological University, Singapore. 

It is about a closed kaggle competition - [Bosch Production Line Performance](https://www.kaggle.com/c/bosch-production-line-performance). Datasets are obtained [here](https://www.kaggle.com/c/bosch-production-line-performance/data). 

### Pre-requisite
Please make sure you have the following packages before running scripts
The Python Environment we are using is Python 3.6
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [XGBoost](https://github.com/dmlc/xgboost)
* [Seaborn](https://seaborn.pydata.org/index.html) 

### Feature Engineering (FINAL SOLUTION)
The script that generates features used in our final model is `code/features_for_xgboost_model_final.py`. Please put all data files in data folder alongside the code folder.

### Predictive Model (FINAL SOLUTION)
Final model script is in `code/xgboost_model_final.py` file.

### Other Feature Scripts and Models Attempted Mentioned in Report (Not in FINAL SOLUTION)
These scripts are placed in `Others` folder

#### Other Features Attempted
`Others/count_dup.py` Duplicate count
`Others/datetime_feat.py` Compressed datetime features
`Others/kurtosis.py` Kurtosis
`Others/id_feat.py` Generates Id_based features
`Others/station_marker.py` Marker for station 32, 33, 34

#### Other Models Attempted
`Others/random_forest.py` Random Forest model
`Others/XGBoostModel.py` Another XGBoost model using different feature set
`Others/gbm.py` LightGBM model

### Acknowledgement 
All models' performances were measured by submission to Kaggle competition.
 
