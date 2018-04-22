from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import gc

# Examples in training data : 1183747
# Examples in test data: 1183748
NUM_TRAINING = 1183747
NUM_TEST = 1183748

def consecutive_response_feature():
    train_df = pd.read_csv('data/train_numeric.csv', usecols=['Id', 'Response'])
    test_df = pd.read_csv('data/test_numeric.csv', usecols=['Id'])
    test_df['Response'] = 0
    
    concat_df = pd.concat([train_numeric, test_numeric]).sort_values(by='Id').reset_index(drop=True)
    
    concat_df['IsPreviousDefective'] = 0
    concat_df['IsNextDefective'] = 0
    
    for index in range(concat_df.shape[0]):
        if concat_df.Response[index] == 1:
            concat_df.IsPreviousDefective[index + 1] = 1
            concat_df.IsNextDefective[index - 1] = 1
    
    concat_df = concat_numeric.drop(labels='Response', axis=1)
    concat_df.to_csv('xgb_features/consecutive_response.csv', index=False)
    
def faron_features():
    train_df = pd.read_csv('data/train_numeric.csv', usecols=['Id', 'Response'])
    test_df = pd.read_csv('data/test_numeric.csv', usecols=['Id'])
    
    train_df['StartTime'] = -1
    test_df['StartTime'] = -1
    
    train_df['EndTime'] = -1
    test_df['EndTime'] = -1
    
    total_rows = max(NUM_TRAINING, NUM_TEST)
    rows_parsed = 0
    chunk_size = 50000
    
    # Last training chunk will have one row less than the last test chunk
    for train_date, test_date in zip(pd.read_csv('data/train_date.csv', chunksize=chunk_size), pd.read_csv('data/test_date.csv', chunksize=chunk_size)):
        features = list(train_date.columns)
        features.remove('Id')
        
        # StartTime is min timestamp
        train_start_time = train_date[features].min(axis=1).values
        test_start_time = test_date[features].min(axis=1).values
        
        # EndTime is max timestamp
        train_end_time = train_date[features].max(axis=1).values
        test_end_time = test_date[features].max(axis=1).values
        
        # Set StartTime
        train_df.loc[train_df.Id.isin(train_date.Id), 'StartTime'] = train_start_time
        test_df.loc[test_df.Id.isin(test_date.Id), 'StartTime'] = test_start_time
        
        # Set EndTime
        train_df.loc[train_df.Id.isin(train_date.Id), 'EndTime'] = train_end_time
        test_df.loc[test_df.Id.isin(test_date.Id), 'EndTime'] = test_end_time
        
        rows_parsed += chunk_size
        print('Rows parsed: ' + str(rows_parsed))
        if rows_parsed >= total_rows:
            break
    
    concat_df = pd.concat([train_df, test_df]).reset_index(drop=True).reset_index(drop=False)
    concat_df['Duration'] = concat_df['EndTime'] - concat_df['StartTime']
    concat_df['FeatureOne'] = concat_df['Id'].diff().fillna(9999999).astype(int)
    concat_df['FeatureTwo'] = concat_df['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
    concat_df = concat_df.sort_values(by=['StartTime', 'Id'], ascending=True)
    concat_df['FeatureThree'] = concat_df['Id'].diff().fillna(9999999).astype(int)
    concat_df['FeatureFour'] = concat_df['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
    concat_df = concat_df.sort_values(by='index').drop(labels='index', axis=1)
    
    train_df = concat_df.iloc[:NUM_TRAINING, :].drop(labels='Response', axis=1)
    test_df = concat_df.iloc[NUM_TRAINING:, :].drop(labels='Response', axis=1)
    
    train_df.to_csv('xgb_features/faron_train.csv', index=False)
    test_df.to_csv('xgb_features/faron_test.csv', index=False)
    
def numeric_features():
    series_list = []
    skip_rows = 0
    n_rows = 300000
    index = 0
    while index < 4:
        train_numeric = pd.read_csv('data/train_numeric.csv', index_col=0, usecols=list(range(969)),
                                    nrows=n_rows, skiprows=skip_rows).fillna(9999999)
        y = pd.read_csv("data/train_numeric.csv", index_col=0, usecols=[0,969]).loc[train_numeric.index].values.ravel()
        X = train_numeric.values
        
        print('Fitting Begins for iteration ' + str(index))
        clf = XGBClassifier(base_score= 0.0058, max_depth = 6, learning_rate = 0.05, seed = 1234, nthread=4)
        clf.fit(X, y, eval_metric = 'auc')
        print('Fitting Ends for iteration ' + str(index))
        
        plt.hist(clf.feature_importances_[clf.feature_importances_>0])
        important_indices_num = np.where(clf.feature_importances_>0.006)[0]
        series_list.append(important_indices_num)
#        important_indices_num = pd.DataFrame(important_indices_num)
#        important_indices_num.columns = ['Important Features']
#        important_indices_num.to_csv('xgb_features/important_indices_num' + str(index) + '.csv', index=False)
        
        index += 1
        skip_rows += n_rows
    
    important_indices_num = np.concatenate(series_list, axis=0)
    important_indices_num = np.unique(important_indices_num)
    important_indices_num = pd.DataFrame(important_indices_num)
    important_indices_num.columns = ['ImportantFeatures']
    important_indices_num.to_csv('xgb_features/important_indices_num_0.06.csv', index=False)
    
#def categorical_features():
#    series_list = []
#    skip_rows = 0
#    n_rows = 100000
#    index = 0
#    while index < 12:
#        train_cat = pd.read_csv('data/train_categorical.csv', index_col=0, 
#                                nrows=n_rows, skiprows=skip_rows, dtype=str)
#        train_cat = train_cat.replace(np.nan, 'T0', regex=True)
#        train_cat = train_cat.apply(LabelEncoder().fit_transform)
#        y = pd.read_csv("data/train_numeric.csv", index_col=0, usecols=[0,969],
#                        nrows=n_rows, skiprows=skip_rows).loc[train_cat.index].values.ravel()
#    
#        print('Fitting Begins for iteration ' + str(index))    
#        clf = XGBClassifier(base_score= 0.0058, max_depth = 6, learning_rate = 0.05, seed = 1234, n_jobs=4)
#        clf.fit(train_cat, y , eval_metric = 'auc')
#        print('Fitting Ends for iteration ' + str(index))
#        y = pd.read_csv('data/train_numeric.csv', index_col=0, usecols=[0, 969], dtype=np.float32).loc[train_cat.index].values.ravel()
#        
#        plt.hist(clf.feature_importances_[clf.feature_importances_>0])
#        important_indices_cat = np.where(clf.feature_importances_>0.01)[0]
#        series_list.append(important_indices_cat)    
##        important_indices_cat.to_csv('xgb_features\important_indices_cat' + str(index) + '.csv', index=False)
#        
#        index += 1
#        skip_rows += n_rows
#    
#    important_indices_cat = np.concatenate(series_list, axis=0)
#    important_indices_cat = np.unique(important_indices_cat)
#    important_indices_cat = pd.DataFrame(important_indices_cat)
#    important_indices_cat.columns = ['ImportantFeatures']
#    important_indices_cat.to_csv('xgb_features/important_indices_cat_0.01.csv', index=False)
    
    
def station_date_features():
    # Get those features for station 32, 33 and 34 which have the maximum number of non-NA values
    useful_stations = ['S32', 'S33', 'S34']
    
    train_date_temp = pd.read_csv('data/train_date.csv', nrows=1)
    
    date_cols = list(train_date_temp.columns)    
    date_cols.remove('Id')
    
    useful_cols = [x for x in date_cols if x.split('_')[1] in useful_stations]
    useful_cols.append('Id')
    
    df = pd.read_csv('data/train_date.csv', usecols=useful_cols).drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    df['Station'] = df['index'].apply(lambda x: x.split('_')[1])

    cols = df.drop_duplicates('Station', keep='first')['index'].tolist()
    cols.append('Id')
    
    train_date = pd.read_csv('data/train_date.csv', usecols=cols)
    train_date.columns = ['Id'] + useful_stations
    
    for station in useful_stations:
        train_date[station] = 1 * (train_date[station] >= 0)
    
    pattern = []
    
    for index, row in train_date.iterrows():
        if row['S32'] ==  1 and row['S33'] == 0 and row['S34'] == 0:
            pattern.append(1)
        elif row['S32'] == 1 and row['S33'] == 0 and row['S34'] == 1:
            pattern.append(2)
        else:
            pattern.append(0)
            
    train_date['Pattern'] = pattern
    train_date = train_date.drop(labels=['S33', 'S34'], axis=1)
    train_date.to_csv('xgb_features/train_date_features.csv', index=False)

    test_date = pd.read_csv('data/test_date.csv', usecols=cols)
    test_date.columns = ['Id'] + useful_stations
    
    for station in useful_stations:
        test_date[station] = 1 * (test_date[station] >= 0)
    
    pattern = []
    
    for index, row in test_date.iterrows():
        if row['S32'] ==  1 and row['S33'] == 0 and row['S34'] == 0:
            pattern.append(1)
        elif row['S32'] == 1 and row['S33'] == 0 and row['S34'] == 1:
            pattern.append(2)
        else:
            pattern.append(0)
            
    test_date['Pattern'] = pattern
    test_date = test_date.drop(labels=['S33', 'S34'], axis=1)
    test_date.to_csv('xgb_features/test_date_features.csv', index=False)
    
def run():
    start = time.time()
#    consecutive_response_feature()
#    faron_features()
#    numeric_features()
#    categorical_features()
    station_date_features()
    print(time.time() - start)

run()