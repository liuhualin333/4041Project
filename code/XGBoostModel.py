import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
# import seaborn as sns

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


DATA = "../data"
num_chunks = pd.read_csv(DATA+"/train_numeric.csv", index_col=0, chunksize=300000, usecols=list(range(969)), dtype=np.float32)
datetime_chunks = pd.read_csv(DATA+"/datetime_train.csv", index_col=0, chunksize=300000, dtype=np.float32)
y = pd.read_csv(DATA+"/train_numeric.csv", usecols=[0,969], dtype=np.float32)
X_train, X_test, y_train, y_test = train_test_split(y.iloc[:,0], y.iloc[:,1], test_size=0.2, stratify=y.iloc[:,1])
X = pd.concat([pd.concat([dchunk.loc[dchunk.index.isin(X_test),:],nchunk.loc[nchunk.index.isin(X_test),:]],axis=1) for dchunk,nchunk in zip(datetime_chunks,num_chunks)]).sort_index()
y = pd.read_csv(DATA+"/train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[X.index]

clf = xgb.XGBClassifier(max_depth=6, min_child_weight=4, subsample=1.0, colsample_bytree=0.2, learning_rate=0.1, base_score=0.005)
clf.fit(X.values, y.values.ravel())
important_indices = np.where(clf.feature_importances_>=0.006)[0]
print(important_indices)

# split points to load the correct features from read_csv
n_date_features = 116

concecutive_response = pd.read_csv(DATA+"/consecutive_response.csv", dtype=np.float32)
y_id = pd.read_csv(DATA+"/train_numeric.csv", usecols=[0], dtype=np.float32)
concecutive_response = y_id.merge(concecutive_response[['Id', 'is_previous_defective', 'is_next_defective']], on='Id').set_index('Id')
X = np.concatenate([
    pd.read_csv(DATA+"/datetime_train.csv", index_col=0, dtype=np.float32,usecols=np.concatenate([[0], important_indices[(important_indices < n_date_features)] + 1])).fillna(9999999).values,
    pd.read_csv(DATA+"/id_feat_train.csv", index_col=0, dtype=np.float32, usecols=[1,3,4,5,6,7]).fillna(9999999).values,
    pd.read_csv(DATA+"/train_dup_int.csv", index_col=0, dtype=np.float32).fillna(9999999).values,
    pd.read_csv(DATA+"/station_marker_train.csv", index_col=0, dtype=np.float32).fillna(9999999).values,
    pd.read_csv(DATA+"/kurtosis_feat_train.csv", index_col=0, dtype=np.float32).fillna(9999999).values,
    concecutive_response.fillna(9999999).values,
    pd.read_csv(DATA+"/train_numeric.csv", index_col=0, dtype=np.float32, usecols=np.concatenate([[0], important_indices[(important_indices >= n_date_features)] + 1 - n_date_features])).fillna(9999999).values
], axis=1)
y = pd.read_csv(DATA+"/train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
preds = np.ones(y_test.shape[0])
# Get from cross validation
params = {
    'booster':'gbtree',
    'objective' : 'reg:logistic',
    'colsample_bytree' : 0.2,
    'min_child_weight':4,
    'subsample' : 1,
    'learning_rate': 0.1,
    'max_depth':6,
    'gamma': 0.05
}
training_data = xgb.DMatrix(X_train, y_train)
model = xgb.train(params, training_data, 230,  feval=mcc_eval,  maximize=True)

preds = model.predict(xgb.DMatrix(X_test))

# pick the best threshold out-of-fold
thresholds = np.linspace(0.01, 0.99, 99)
mcc = np.array([matthews_corrcoef(y_test, preds>thr) for thr in thresholds])
plt.plot(thresholds, mcc)
plt.show()
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())
print(best_threshold)

model = xgb.train(params, xgb.DMatrix(X, y), 230,  feval=mcc_eval,  maximize=True)

concecutive_response = pd.read_csv(DATA+"/consecutive_response.csv", dtype=np.float32)
y_id = pd.read_csv(DATA+"/test_numeric.csv", usecols=[0], dtype=np.float32)
concecutive_response = y_id.merge(concecutive_response[['Id', 'is_previous_defective', 'is_next_defective']], on='Id').set_index('Id')
X = np.concatenate([
    pd.read_csv(DATA+"/datetime_test.csv", index_col=0, dtype=np.float32,usecols=np.concatenate([[0], important_indices[(important_indices < n_date_features)] + 1])).fillna(9999999).values,
    pd.read_csv(DATA+"/id_feat_test.csv", index_col=0, dtype=np.float32, usecols=[1,3,4,5,6,7]).fillna(9999999).values,
    pd.read_csv(DATA+"/test_dup_int.csv", index_col=0, dtype=np.float32).fillna(9999999).values,
    pd.read_csv(DATA+"/station_marker_test.csv", index_col=0, dtype=np.float32).fillna(9999999).values,
    pd.read_csv(DATA+"/kurtosis_feat_test.csv", index_col=0, dtype=np.float32).fillna(9999999).values,
    concecutive_response.fillna(9999999).values,
    pd.read_csv(DATA+"/test_numeric.csv", index_col=0, dtype=np.float32, usecols=np.concatenate([[0], important_indices[(important_indices >= n_date_features)] + 1 - n_date_features])).fillna(9999999).values
], axis=1)

# generate predictions
preds = (model.predict(xgb.DMatrix(X)) > best_threshold).astype(np.int8)

# prepare submission
sub = pd.read_csv(DATA+"/sample_submission.csv", index_col=0)
sub["Response"] = preds
sub.to_csv("submission.csv.gz", compression="gzip")