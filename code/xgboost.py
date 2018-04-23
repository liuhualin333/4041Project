from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost
    
def mathews_corr_coef(true_pos, true_neg, false_pos, false_neg):
    # Calculate numerator
    numerator = true_pos * true_neg - false_pos * false_neg
    # Calculate denominator
    denominator = (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)
    # If denominator is 0, then MCC is 0
    if denominator == 0:
        return 0
    else:
    # If denominator is not 0, then use the formula
        return numerator / np.sqrt(denominator)
    
def calulate_mcc(y_true, y_prob, show=False):
    ids = np.argsort(y_prob)
    # Sort the responses.
    y_true_sort = y_true[ids]
    num_items = y_true_sort.shape[0]
    # Calculate the number of postives identified.
    pos_num = 1.0 * np.sum(y_true)  
    # Calculate the number of negatives identified.
    neg_num = num_items - pos_num
    # True positives
    true_pos = pos_num
    # True negatives
    true_neg = 0.0
    # False positives
    false_pos = neg_num
    # False negatives
    false_neg = 0.0
    # Index for best MCC score so far
    best_id = -1
    # Best MCC score so far
    best_mcc = 0.0
    # List to store MCC scores
    mcc_list = np.zeros(num_items)
    for i in range(num_items):
        if y_true_sort[i] == 1:
            false_neg += 1.0
            true_pos -= 1.0
        else:
            true_neg += 1.0
            false_pos -= 1.0
        # Calculate MCC score
        new_mcc = mathews_corr_coef(true_pos, true_neg, false_pos, false_neg)
        # Store MCC score
        mcc_list[i] = new_mcc
        # Check for the best MCC score
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    # If verbose, return multiple items for evaluation
    if show:
        best_proba = y_prob[ids[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
    # If not verbose, just return the MCC score
        return best_mcc

def mcc_score_for_classifier(y_prob, dtrain):
    y_true = dtrain.get_label()
    # Calculate the MCC score for the given predictions probabilities
    best_mcc = calulate_mcc(y_true, y_prob)
    return 'MCC', best_mcc

# Import station 32 features. 
train_station32 = pd.read_csv('../data/train_date_features.csv')
test_station32 = pd.read_csv('../data/test_date_features.csv')

train_station32 = train_s32_pattern.merge(pd.get_dummies(train_s32_pattern.Pattern), left_index = True, right_index = True, how = 'inner').drop('Pattern',1)
test_station32 = test_s32_pattern.merge(pd.get_dummies(test_s32_pattern.Pattern), left_index = True, right_index = True, how = 'inner').drop('Pattern',1)

# Import the faron features for training data
train_faron = pd.read_csv('../data/faron_train.csv')  
# Import the faron features for test data
test_faron = pd.read_csv('../data/faron_test.csv')

# Import consecutive response features
consecutive_response = pd.read_csv('../data/consecutive_response.csv') 

# Import important numeric indices having high feature importance score
important_numeric_indices = pd.read_csv('../data/numeric_indices.csv')
important_numeric_indices = important_numeric_indices.iloc[:,0]

# Import the traning and test numeric data, but only with important columns
train_numeric = pd.read_csv("../data/train_numeric.csv", usecols = np.concatenate([[0], important_numeric_indices +1]) 
                         ,dtype=np.float32).fillna(9999999)
test_numeric = pd.read_csv("../data/test_numeric.csv", usecols = np.concatenate([[0], important_numeric_indices +1]) ,
                       dtype=np.float32).fillna(9999999)

# Import important datetime indices having high feature importance score
important_datetime_indices = pd.read_csv('xgb_features/datetime_indices.csv')
important_datetime_indices = important_datetime_indices.iloc[:,0]

# Import the traning and test datetime data, but only with important columns
train_datetime = pd.read_csv('../data/train_date.csv', usecols = np.concatenate([[0], important_datetime_indices +1]))
test_date = pd.read_csv('../data/test_date.csv', usecols = np.concatenate([[0], important_datetime_indices +1]))

# Get responses from the training data
y_train = pd.read_csv('../data/train_numeric.csv', usecols=[0, 969], dtype=np.float32)
# Merge faron features with the numeric features for the training data.
X_train = train_numeric.merge(train_faron, left_index=True, right_on = 'Id',left_on = 'Id', how = 'inner')
# Merge datetime features with the training data.
X_train = X_train.merge(train_date, left_on = 'Id', right_on = 'Id', how ='inner')
# Merge the station32 features with the training data.
X_train = X_train.merge(train_station32, left_on = 'Id', right_on = 'Id', how = 'inner')
# Merge the consecutive response features with the training data.
X_train = X_train.merge(consecutive_response, left_on = 'Id', right_on = 'Id', how = 'inner')
X_train['Id'] = np.float32(X_train['Id'])
# Create the XGBoost DMatrix for the training data.
training_data = xgboost.DMatrix(X_train,y_train['Response'])

# Merge faron features with the numeric features for the test data.
X_test = test_numeric.merge(test_faron, left_index=True, right_on = 'Id', left_on = 'Id', how = 'inner')
# Merge datetime features with the test data.
X_test = X_test.merge(test_date, left_on = 'Id', right_on = 'Id', how ='inner')
# Merge the station32 features with the test data.
X_test = X_test.merge(test_station32, left_on = 'Id', right_on = 'Id', how = 'inner')
# Merge the consecutive response features with the test data.
X_test = X_test.merge(consecutive_response, left_on = 'Id', right_on = 'Id', how = 'inner')
X_test['Id'] = np.float32(X_test['Id'])
# Create the XGBoost DMatrix for the test data.
test_data = xgboost.DMatrix(X_test)

# Create a train-test split for cross validation.
X_training, X_testing, y_training, y_testing = train_test_split(X_train, y_train['Response'],test_size = 0.2)

# Create multiple DMatrices for testing and cross-validation.
check_one = xgboost.DMatrix(X_testing, y_testing)
check_two = xgboost.DMatrix(X_testing)
check_three = xgboost.DMatrix(X_training, y_training)
watchlist = [(set_three,'train'),(set_one,'val')]
# Parameters for training the xgboost classifier. Obtained from Grid-Search CV.
params = {
    'booster':'gbtree',
    'objective' : 'reg:logistic',
    'colsample_bytree' : 0.2,
    'min_child_weight':4,
    'subsample' : 1,
    'learning_rate': 0.1,
    'max_depth':6,
    'gamma': 0.05,
    'seeds' : 1234
}

# Create the classifier. Have 300 iteration rounds.
classifier = xgboost.train(params, training_data, 300,  feval=mcc_score_for_classifier,  maximize=True) 
# Get predictions for the test split.
y_pred_test = classifier.predict(check_two)
# Try out multiple threshold values to find the optimal threshold.
threshold_list = np.linspace(0.2, 0.5, 1000)
mcc_score = np.array([matthews_corrcoef(y_testing, y_pred_test>thresh) for thresh in threshold_list])
best_threshold = threshold_list[mcc_score.argmax()]

predictions_real = model.predict(test_data)
# Threshold that maximises MCC score for the testing data in the train-test split.
y_pred = (predictions_real > 0.365).astype(int)
submission = pd.read_csv("../data/sample_submission.csv", index_col=0)
submission["Response"] = y_pred
submission.to_csv("../data/final_submission.csv")