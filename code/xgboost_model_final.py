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


# Import all the generated features.
important_datetime_features = pd.read_csv('../data/datetime_features.csv')
important_datetime_features = (important_datetime_features.iloc[:, 0]).tolist()

train_date = pd.read_csv('../data/train_date.csv', usecols = ['Id'] + important_datetime_features)
test_date = pd.read_csv('../data/test_date.csv', usecols = ['Id'] + important_datetime_features)    

train_station32 = pd.read_csv('../data/train_station32_features.csv')
test_station32 = pd.read_csv('../data/test_station32_features.csv')

train_station32 = train_station32.merge(pd.get_dummies(train_station32.Pattern), left_index = True, right_index = True, how = 'inner').drop('Pattern',1)
test_station32 = test_station32.merge(pd.get_dummies(test_station32.Pattern), left_index = True, right_index = True, how = 'inner').drop('Pattern',1)

train_order = pd.read_csv('../data/train_order.csv')
test_order = pd.read_csv('../data/test_order.csv')

train_path = pd.read_csv('../data/train_path.csv')
test_path = pd.read_csv('../data/test_path.csv')

consecutive_response = pd.read_csv('../data/consecutive_response.csv')

important_numeric_indices = pd.read_csv('../data/numeric_indices.csv')
important_numeric_indices = important_numeric_indices.iloc[:,0]

train_numeric = pd.read_csv("../data/train_numeric.csv", usecols = np.concatenate([[0], important_numeric_indices + 1])
                         ,dtype=np.float32).fillna(9999999)
test_numeric = pd.read_csv("../data/test_numeric.csv", usecols = np.concatenate([[0], important_numeric_indices +1]) ,
                       dtype=np.float32).fillna(9999999)

# Get responses from the training data
y_train = pd.read_csv('../data/train_numeric.csv', usecols=[0, 969], dtype=np.float32)


# Merge all the features with the numeric features for the training data.
X_train = train_numeric.merge(train_order, left_index=True, right_on = 'Id',left_on = 'Id', how = 'inner')
X_train = X_train.merge(train_date, left_on = 'Id', right_on = 'Id', how ='inner')
X_train = X_train.merge(train_path, left_on = 'Id', right_on = 'Id', how ='inner')
X_train = X_train.merge(train_station32, left_on = 'Id', right_on = 'Id', how = 'inner')
X_train = X_train.merge(consecutive_response, left_on = 'Id', right_on = 'Id', how = 'inner')
X_train['Id'] = np.float32(X_train['Id'])
training_data = xgboost.DMatrix(X_train,y_train['Response'])

# Merge all the features with the numeric features for the test data.
X_test = test_numeric.merge(test_order, left_index=True, right_on = 'Id', left_on = 'Id', how = 'inner')
X_test = X_test.merge(test_date, left_on = 'Id', right_on = 'Id', how ='inner')
X_test = X_test.merge(test_path, left_on = 'Id', right_on = 'Id', how ='inner')
X_test = X_test.merge(test_station32, left_on = 'Id', right_on = 'Id', how = 'inner')
X_test = X_test.merge(consecutive_response, left_on = 'Id', right_on = 'Id', how = 'inner')
X_test['Id'] = np.float32(X_test['Id'])
test_data = xgboost.DMatrix(X_test)

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

# Create the classifier. We Have 300 iteration rounds.
classifier = xgboost.train(params, training_data, 300,  feval=mcc_score_for_classifier,  maximize=True)

# Predictions for the test data.
# Threshold obtained from cross-validation
predictions_real = classifier.predict(test_data)
y_pred = (predictions_real > 0.365).astype(int)
submission = pd.read_csv("../data/sample_submission.csv", index_col=0)
submission["Response"] = y_pred
submission.to_csv("../data/final_submission.csv")
