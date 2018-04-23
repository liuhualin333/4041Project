# Bosch Production Line Performance 

### Introduction  
This repository contains teh source code and other helper files for the group project of course CZ4041 - Machine Learning at Nanyang Technological University, Singapore. 

It is about a closed kaggle competition - Bosch Production Line Performance. 

### Dependencies 
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [XGBoost](https://github.com/dmlc/xgboost)
* [Seaborn](https://seaborn.pydata.org/index.html)

### Source code 

There are a total of five machine learning models implemented in this reposiroty for Matthews correlation coefficient calculation.
1. `XGBoostModel.py`: XGBoost 
2. ``: lightGBM, 
3. `random_forest.py`: Random Forest, 
4. ``: SVM 
5. ``: Neural Network. 

First three of them works on given dataset, the last two were abandoned. 
The above models were tested on kaggle [datasets](https://www.kaggle.com/c/bosch-production-line-performance/data). 
The performance of the models on these datasets can be found [here]().
