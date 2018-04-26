import lightgbm
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def mcc(tp, tn, fp, fn):
    sub = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    return sub / np.sqrt(inf)

def eval_mcc(y_ground, y_pred, show=False):
    idx = np.argsort(y_pred)
    y_ground_sort = y_ground[idx]
    n = y_ground.shape[0]
    nump = 1.0 * np.sum(y_ground)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_ground_sort[i] == 1:
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
        best_pred = y_pred[idx[best_id]]
        y_pred = (y_pred > best_pred).astype(int)
        return best_pred, best_mcc, y_pred

    return best_mcc


def mcc_eval(preds, train_data):
    y_true = train_data.get_label()
    best_mcc = eval_mcc(y_true, preds)
    return 'MCC', best_mcc, True

cols =   ['Id',
             'L0_S4_F109', 'L0_S15_F403', 'L0_S13_F354',
             'L1_S24_F1846', 'L1_S24_F1695', 'L1_S24_F1632', 'L1_S24_F1604',
             'L1_S24_F1723', 'L1_S24_F1844', 'L1_S24_F1842',
             'L2_S26_F3106', 'L2_S26_F3036', 'L2_S26_F3113', 'L2_S26_F3073',
             'L3_S29_F3407', 'L3_S29_F3376', 'L3_S29_F3324', 'L3_S29_F3382', 'L3_S29_F3479',
             'L3_S30_F3704', 'L3_S30_F3774', 'L3_S30_F3554',
             'L3_S32_F3850', 'L3_S32_F3850',
             'L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3865',
             'L3_S37_F3944', 'L3_S37_F3946', 'L3_S37_F3948', 'L3_S37_F3950',
             'L3_S38_F3956', 'L3_S38_F3960', 'L3_S38_F3952',
             'L3_S30_F3604', 'L3_S30_F3749', 'L0_S0_F20', 'L3_S30_F3559', 'L3_S30_F3819', 'L3_S29_F3321', 'L3_S29_F3373',
             'L3_S30_F3569', 'L3_S30_F3569', 'L3_S30_F3579', 'L3_S30_F3639', 'L3_S29_F3449', 'L3_S36_F3918', 'L3_S30_F3609',
             'L3_S30_F3574', 'L3_S29_F3354', 'L3_S30_F3759', 'L0_S6_F122', 'L3_S30_F3664', 'L3_S30_F3534', 'L0_S1_F24', 'L3_S29_F3342',
             'L0_S7_F138', 'L2_S26_F3121', 'L3_S30_F3744', 'L3_S30_F3799', 'L3_S33_F3859', 'L3_S30_F3784', 'L3_S30_F3769', 'L2_S26_F3040',
             'L3_S30_F3804', 'L0_S5_F114', 'L0_S12_F336', 'L0_S9_F170', 'L3_S29_F3330', 'L3_S29_F3351', 'L3_S29_F3339', 'L3_S29_F3427', 'L3_S30_F3829',
             'L0_S0_F22', 'L3_S30_F3589', 'L3_S30_F3494', 'L3_S29_F3421', 'L3_S29_F3327', 'L0_S5_F116', 'L3_S29_F3318', 'L3_S30_F3524', 'L3_S29_F3379',
             'L3_S29_F3333', 'L3_S29_F3455', 'L3_S29_F3430', 'L3_S30_F3529', 'L0_S0_F0', 'L3_S30_F3754', 'L3_S36_F3920', 'L0_S3_F96', 'L3_S29_F3407',
             'L3_S29_F3473', 'L3_S29_F3476', 'L3_S30_F3674']

temp = cols
DATA = "../data"

#training
#load feature
concecutive_response = pd.read_csv(DATA+"/consecutive_response.csv", dtype=np.float32)
y_id = pd.read_csv(DATA+"/station_marker_train.csv", usecols=[0], dtype=np.float32)
concecutive_response = y_id.merge(concecutive_response[['Id', 'is_previous_defective', 'is_next_defective']], on='Id').set_index('Id')
X = np.concatenate([
    pd.read_csv(DATA + "/train_numeric.csv", index_col=0, usecols=cols, dtype=np.float32).values,
    pd.read_csv(DATA+"/id_feat_train.csv", index_col=0, usecols=[1,3,4,5,6,7], dtype=np.float32).values,
    pd.read_csv(DATA+"/train_dup_int.csv", index_col=0, dtype=np.float32).values,
    pd.read_csv(DATA+"/station_marker_train.csv", index_col=0, dtype=np.float32).values,
    concecutive_response.values
], axis=1)
#load response
y = pd.read_csv(DATA+"/train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()
train = X
#train/test split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42, stratify=y)

train_data = lightgbm.Dataset(X_train, label = y_train)
test_data = lightgbm.Dataset(X_test, label = y_test)

parameters = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 100,
    'max_depth':6,
    'min_child_weight': 1.25,
    'is_unbalance': 'false',
    'scale_pos_weight':175,
    'num_leaves': 70,
    'feature_fraction': 0.9,
}


gbm = lightgbm.fit(parameters,
                      train_data,
                       valid_sets=test_data,feval=mcc_eval,
                       early_stopping_rounds=50)


print (gbm.feature_importance())
#save model
filename = './gbm.pkl'
joblib.dump(gbm, filename)

preds = np.ones(y_test.shape[0])
preds = gbm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, preds))
#calculate best threshold
thresholds = np.linspace(0.1, 0.99, 50)
mcc = np.array([matthews_corrcoef(y_test, preds>thr) for thr in thresholds])
plt.plot(thresholds, mcc)
plt.show()
best_threshold = thresholds[mcc.argmax()]
print(best_threshold)
print(mcc.max())


#output prediction results
filename = './gbm.pkl'
gbm = joblib.load(filename)
concecutive_response = pd.read_csv(DATA+"/consecutive_response.csv", dtype=np.float32)
y_id = pd.read_csv(DATA+"/station_marker_test.csv", usecols=[0], dtype=np.float32)
concecutive_response = y_id.merge(concecutive_response[['Id', 'is_previous_defective', 'is_next_defective']], on='Id').set_index('Id')
X = np.concatenate([
    pd.read_csv(DATA + "/test_numeric.csv", index_col=0, usecols=cols, dtype=np.float32).values,
    pd.read_csv(DATA+"/id_feat_test.csv", index_col=0, usecols=[1,3,4,5,6,7], dtype=np.float32).values,
    pd.read_csv(DATA+"/test_dup_int.csv", index_col=0, dtype=np.float32).values,
    pd.read_csv(DATA+"/station_marker_test.csv", index_col=0, dtype=np.float32).values,
    concecutive_response.values
], axis=1)
print('Start prediction...')
submission = X
y = gbm.predict_proba(submission)[:,1]
count = 0
for i in range(0,len(y)):
    if y[i]>=best_threshold:
       y[i]=1
       count+=1
    else:
       y[i]=0
y = y.astype(np.int8)
print (count)
output = pd.read_csv(DATA+"/sample_submission.csv", index_col=0)
output["Response"] = y
output.to_csv("gbmsubmission.csv")
