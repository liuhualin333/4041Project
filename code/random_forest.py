"""
This script includes the  training and testing process for random forest classifier

Following are required packages:
1. numpy
2. pandas
3. scikit-learn

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import matthews_corrcoef
from collections import Counter


# trail run on numeric data and date time data 
# to reduce the number of feature used for training
chunk_size = 100000

train_raw_numeric_dir = '../data/train_numeric.csv'
train_datetime_dir = '../data/datetime_train.csv'

# get response column and split out 5% of the total samples for trail run
y = pd.read_csv(train_raw_numeric_dir, usecols=[0, 969], dtype=np.float32)
tr_x, te_x, tr_y, te_y = train_test_split(y['Id'], y['Response'].values, test_size=0.05, random_state=42)

# get index of sub-samples for trail 
sample_index = te_x.values.ravel()

# read trail samples of nemuric data
chunk1 = pd.read_csv(train_raw_numeric_dir, chunksize=chunk_size, dtype=np.float32, usecols=list(range(969)))
X1 = pd.concat([chunk.loc[chunk['Id'].isin(sample_index)] for chunk in chunk1])
X1 = X1.fillna(X1.mean())
X1 = X1.set_index('Id')

# read trail samples of datetime data
chunk2 = pd.read_csv(train_datetime_dir, chunksize=chunk_size, dtype=np.float32)
X2 = pd.concat([chunk.loc[chunk['Id'].isin(sample_index)] for chunk in chunk2])
X2 = X2.fillna(X2.mean())
X2 = X2.set_index('Id')

# deal with columns with all NaN value and 
# concat numeric data with datetime data
X = pd.concat([X1, X2], axis=1)
X = X.fillna(0)
X = X.values


# trail run 
rf_clf = RandomForestClassifier(n_estimators=600, max_depth=30, max_leaf_nodes=500, max_features='auto',
                             n_jobs=-1, oob_score=True, random_state=42, verbose=1)

# show the feature importances in the chart for threshold
# plt.hist(rf_clf.feature_importances_[rf_clf.feature_importances_>0])
# plt.show()

# select columns with feature importance larger than 0.003
impt_idx = np.where(rf_clf.feature_importances_>0.003)[0]



#--------------------------parameter tuning-------------------------------------------
# """
# parameter tunning using GridSearchCV for parameter range
# """

# # function takes a RF parameter and a ranger and 
# # produces a plot and dataframe of CV scores for parameter values

# clf = RandomForestClassifier(n_jobs=-1, random_state=42)

# _, X_gs, _, y_gs = train_test_split(X, y, test_size=0.4, random_state=42) 

# def evaluate_param(parameter, num_range, index):
#     grid_search = GridSearchCV(clf, param_grid = {parameter: num_range}, verbose=1, n_jobs=-1)
#     grid_search.fit(X_gs, y_gs)
    
#     df = {}
#     for i, score in enumerate(grid_search.grid_scores_):
#         df[score[0][parameter]] = score[1]
       
    
#     df = pd.DataFrame.from_dict(df, orient='index')
#     df.reset_index(level=0, inplace=True)
#     df = df.sort_values(by='index')
 
#     plt.subplot(3,2,index)
#     plot = plt.plot(df['index'], df[0])
#     plt.title(parameter)
#     return plot, df

# # parameters and ranges to plot
# param_grid = {"n_estimators": np.arange(50, 400, 50),
#               "max_depth": np.arange(5, 40, 5),
#               "min_samples_split": np.arange(2, 30, 5),
#               "min_samples_leaf": np.arange(5, 100, 20),
#               "max_leaf_nodes": np.arange(10, 160, 30),
#               "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}

# index = 1
# plt.figure(figsize=(16,12))
# for parameter, param_range in dict.items(param_grid):  
#     evaluate_param(parameter, param_range, index)
#     index += 1


# """
# find best parameters set using GridSearchCV
# """
# from operator import itemgetter

# # Utility function to report best scores
# def report(grid_scores, n_top):
#     top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
#     for i, score in enumerate(top_scores):
#         print("Model with rank: {0}".format(i + 1))
#         print("Mean validation score: {0:.4f})".format(
#               score.mean_validation_score,
#               np.std(score.cv_validation_scores)))
#         print("Parameters: {0}".format(score.parameters))
#         print("")

# param_grid2 = {"n_estimators": [250, 325, 400],
#               "max_depth": [15, 20, 25],
#               "max_leaf_nodes": [250, 300, 350]}

# grid_search = GridSearchCV(clf, param_grid=param_grid2, verbose=1, n_jobs=-1)
# grid_search.fit(X_resampled, y_resampled)

# report(grid_search.grid_scores_, 4)

#-----------------------parameter tuning finished-------------------------------


"""
random forest  training 
"""

# read data into different dataframes and then concate into one
train_count_dup_dir = '../data/train_dup_int.csv'
train_id_feat_dir = '../data/id_feat_train.csv'
train_station_dir = '../data/station_maker_train.csv'

concecutive_response = pd.read_csv('../data/consecutive_response.csv', dtype=np.float32)
y_id = pd.read_csv(train_raw_numeric_dir, usecols=[0], dtype=np.float32)
concecutive_response = y_id.merge(concecutive_response[['Id', 'is_previous_defective', 'is_next_defective']], on='Id').set_index('Id')

df1 = pd.read_csv(train_raw_numeric_dir, index_col=0, dtype=np.float32, 
                  usecols=np.concatenate([[0], impt_idx[impt_idx<969]+1]))
df2 = pd.read_csv(train_datetime_dir, index_col=0, dtype=np.float32, 
                  usecols=np.concatenate([[0], impt_idx[impt_idx>= 969]+1-969]))
df3 = pd.read_csv(train_count_dup_dir, index_col=0, dtype=np.float32)
df4 = pd.read_csv(train_id_feat_dir, index_col=0, dtype=np.float32, usecols=[1,3,4,5,6,7])
df5 = pd.read_csv(train_station_dir, index_col=0, dtype=np.float32)

X = pd.concat([df1, df2, df3, df4, df5, concecutive_response], axis=1)

# read labels
y = pd.read_csv(train_raw_numeric_dir, index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()

# deal with NaN value: fill with column mean
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.values


# using three fold cross validation method to get threshold for testing
skf = StratifiedKFold(n_splits=3)
preds = np.ones(y.shape[0])
rfc = RandomForestClassifier(n_estimators=600, max_depth=30, max_leaf_nodes=500, max_features=0.4,
                             n_jobs=-1, oob_score=True, random_state=42, verbose=1)
m = []
t = []

thresholds = np.linspace(0.01, 0.99, 50)

# downsampling did not help
#cc = RandomUnderSampler(random_state=42) 

cnt = 1
for train, test in skf.split(X, y):
    X_resampled, y_resampled = X[train], y[train]
    
    print('start cv fold ', cnt)
    
    model = rfc.fit(X_resampled, y_resampled)
    pred_cv = model.predict_proba(X[test])[:,1]
    print("fold {}, ROC AUC: {:.3f}".format(cnt, roc_auc_score(y[test], pred_cv)))
    mcc = np.array([matthews_corrcoef(y[test], pred_cv>thr) for thr in thresholds])
    best_threshold = thresholds[mcc.argmax()]
    print('max mcc: ', mcc.max())
    print('threshold: ', best_threshold)
    m.append(mcc.max())
    t.append(best_threshold)
    
    cnt+=1

mean_mcc = sum(m)/3
mean_threshold = sum(t)/3
print("cv mean mcc: ", mean_mcc)
print('cv mean threshold: ', mean_threshold)



"""
random forest testing
"""

# read data into different dataframes and then concate into one
raw_numeric_dir = '../data/test_numeric.csv'
count_dup_dir = '../data/test_dup_int.csv'
datetime_dir = '../data/datetime_test.csv'
id_feat_dir = '../data/id_feat_test.csv'
station_dir = '../data/station_maker_test.csv'

concecutive_response = pd.read_csv('../data/consecutive_response.csv', dtype=np.float32)
y_id = pd.read_csv(raw_numeric_dir, usecols=[0], dtype=np.float32)
concecutive_response = y_id.merge(concecutive_response[['Id', 'is_previous_defective', 'is_next_defective']], on='Id').set_index('Id')


df_test1 = pd.read_csv(raw_numeric_dir, index_col=0, dtype=np.float32, 
                       usecols=np.concatenate([[0], impt_idx[impt_idx < 969]+1]))
df_test2 = pd.read_csv(datetime_dir, index_col=0, dtype=np.float32, 
                       usecols=np.concatenate([[0], impt_idx[impt_idx >= 969]+1-969]))
df_test3 = pd.read_csv(count_dup_dir, index_col=0, dtype=np.float32)
df_test4 = pd.read_csv(id_feat_dir, index_col=0, dtype=np.float32, usecols=[1,4,5,6,7])
df_test5 = pd.read_csv(station_dir, index_col=0, dtype=np.float32)

X_test = pd.concat([df_test1, df_test2, df_test3, df_test4, df_test5, concecutive_response], axis=1)


# deal with column with NaN value: fill with column mean
X_test = X_test.apply(lambda x: x.fillna(x.mean()), axis=0)

# in case of columns with all NaN value: fill 0 
X_test = X_test.fillna(0)
X_test = X_test.values

# create random forest model 
preds = np.ones(X_test.shape[0])
rfc_test = RandomForestClassifier(n_estimators=600, max_depth=30, max_leaf_nodes=500, max_features=0.4,
                             n_jobs=-1, oob_score=True, random_state=42, verbose=1)

# fit the training data
model = rfc_test.fit(X, y)
print('finish fit, start prediction\n')

# make prediction on test data
preds = model.predict_proba(X_test)[:,1]
print('finish prediction: ', preds.shape)


# set test threshold as the mean threshold from training
test_threshold = mean_threshold

final_preds = preds[:]
final_preds[final_preds>=test_threshold]=1
final_preds[final_preds<test_threshold]=0

# save the results into csv file for submission
df_final = pd.DataFrame(index=df_test3.index)
df_final['Response'] = final_preds
df_final.index = df_final.index.map(int)
df_final['Response'] = df_final['Response'].astype(int)

df_final.to_csv('submission.csv')

