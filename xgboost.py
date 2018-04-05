from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost
import gc

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)

def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc

def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc

important_indices_num = pd.read_csv('xgb_features/important_indices_num_0.06.csv')
important_indices_num = important_indices_num.iloc[:,0]

date_cols = ['Id', 'L3_S30_D3496', 'L3_S30_D3506', 'L3_S30_D3501', 'L3_S30_D3516', 'L3_S30_D3511']

train_date = pd.read_csv('data/train_date.csv', usecols = date_cols)
test_date = pd.read_csv('data/test_date.csv', usecols = date_cols)

train_numeric = pd.read_csv("data/train_numeric.csv", usecols = np.concatenate([[0], important_indices_num +1]) 
                         ,dtype=np.float32).fillna(9999999)
test_numeric = pd.read_csv("data/test_numeric.csv", usecols = np.concatenate([[0], important_indices_num +1]) ,
                       dtype=np.float32).fillna(9999999)

train_faron = pd.read_csv('xgb_features/faron_train.csv')  
test_faron = pd.read_csv('xgb_features/faron_test.csv')

consecutive_response = pd.read_csv('xgb_features/consecutive_response.csv') 

train_s32_pattern = pd.read_csv('xgb_features/train_date_features.csv')
test_s32_pattern = pd.read_csv('xgb_features/test_date_features.csv')
train_s32_pattern = train_s32_pattern.merge(pd.get_dummies(train_s32_pattern.Pattern), left_index = True, right_index = True, how = 'inner').drop('Pattern',1)
test_s32_pattern = test_s32_pattern.merge(pd.get_dummies(test_s32_pattern.Pattern), left_index = True, right_index = True, how = 'inner').drop('Pattern',1)

train_date_47 = pd.read_csv('data/train_date.csv', usecols = ['Id','L3_S47_D4150']).fillna(0) 
test_date_47 = pd.read_csv('data/test_date.csv', usecols = ['Id','L3_S47_D4150']).fillna(0) 

train_date_0 = pd.read_csv('data/train_date.csv', usecols = ['Id','L0_S0_D1']).fillna(0) 
test_date_0 = pd.read_csv('data/test_date.csv', usecols = ['Id','L0_S0_D1']).fillna(0)
 
train_date_13 = pd.read_csv('data/train_date.csv', usecols = ['Id','L0_S13_D355']).fillna(0) 
test_date_13 = pd.read_csv('data/test_date.csv', usecols = ['Id','L0_S13_D355']).fillna(0) 

train_date_38 = pd.read_csv('data/train_date.csv', usecols = ['Id','L3_S38_D3953']).fillna(0)
test_date_38 = pd.read_csv('data/test_date.csv', usecols = ['Id','L3_S38_D3953']).fillna(0)

train_date_47['L3_S47_D4150']  = np.where(train_date_47['L3_S47_D4150'] > 0, 1, 0)
train_date_0['L0_S0_D1']       = np.where(train_date_0['L0_S0_D1'] > 0, 1, 0)
train_date_13['L0_S13_D355']   = np.where(train_date_13['L0_S13_D355'] > 0, 1, 0)
train_date_38['L3_S38_D3953']  = np.where(train_date_38['L3_S38_D3953'] > 0, 1, 0)

test_date_47['L3_S47_D4150']  = np.where(test_date_47['L3_S47_D4150'] > 0, 1, 0)
test_date_0['L0_S0_D1']       = np.where(test_date_0['L0_S0_D1'] > 0, 1, 0)
test_date_13['L0_S13_D355']   = np.where(test_date_13['L0_S13_D355'] > 0, 1, 0)
test_date_38['L3_S38_D3953']  = np.where(test_date_38['L3_S38_D3953'] > 0, 1, 0)

y_train = pd.read_csv('data/train_numeric.csv', usecols=[0, 969], dtype=np.float32)
X_train = train_numeric.merge(train_faron, left_index=True, right_on = 'Id',left_on = 'Id', how = 'inner')
X_train = X_train.merge(train_date, left_on = 'Id', right_on = 'Id', how ='inner')
X_train = X_train.merge(train_s32_pattern, left_on = 'Id', right_on = 'Id', how = 'inner')
X_train = X_train.merge(consecutive_response, left_on = 'Id', right_on = 'Id', how = 'inner')
X_train = X_train.merge(train_date_47, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_train = X_train.merge(train_date_0, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_train = X_train.merge(train_date_13, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_train = X_train.merge(train_date_38, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_train['Id'] = np.float32(X_train['Id'])
training_data = xgboost.DMatrix(X_train,y_train['Response'])

X_test = test_numeric.merge(test_faron, left_index=True, right_on = 'Id', left_on = 'Id', how = 'inner')
X_test = X_test.merge(test_date, left_on = 'Id', right_on = 'Id', how ='inner')
X_test = X_test.merge(test_s32_pattern, left_on = 'Id', right_on = 'Id', how = 'inner')
X_test = X_test.merge(consecutive_response, left_on = 'Id', right_on = 'Id', how = 'inner')
X_test = X_test.merge(test_date_47, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test = X_test.merge(test_date_0, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test = X_test.merge(test_date_13, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test = X_test.merge(test_date_38, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test['Id'] = np.float32(X_test['Id'])
test_data = xgboost.DMatrix(X_test)


X_training, X_testing, y_training, y_testing = train_test_split(X_train, y_train['Response'],test_size = 0.2)

set_one = xgboost.DMatrix(X_testing, y_testing)
set_two = xgboost.DMatrix(X_testing)
set_three = xgboost.DMatrix(X_training, y_training)
watchlist = [(set_three,'train'),(set_one,'val')]
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


#model_test = xgboost.train(params, training_data, 1000,  watchlist, feval=mcc_eval, early_stopping_rounds=60,  maximize=True)
#cv = xgboost.cv(params, training_data, 1000, nfold = 5, stratified=True, maximize=True )
#print(cv)
#print('Modelling Started')
model = xgboost.train(params, training_data, 230,  feval=mcc_eval,  maximize=True) 
y_pred_test = model.predict(set_two)
thresholds = np.linspace(0.2, 0.5, 1000)
mcc_one = np.array([matthews_corrcoef(y_testing, y_pred_test>thr) for th in thresholds])
plt.plot(thresholds, mcc_one)
best_threshold = thresholds[mcc_one.argmax()]
print(mcc_one.max())
print(best_threshold)

predictions_real = model.predict(test_data)
y_pred = (predictions_real > 0.365).astype(int)
sub = pd.read_csv("data/sample_submission.csv", index_col=0)
sub["Response"] = y_pred
sub.to_csv("final_submission.csv")