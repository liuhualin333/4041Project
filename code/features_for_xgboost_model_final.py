import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import time

# Examples in training data : 1183747
# Examples in test data: 1183748
NUM_TRAINING = 1183747
NUM_TEST = 1183748

def consecutive_response_features():
    train_df = pd.read_csv('../data/train_numeric.csv', usecols=['Id', 'Response'])
    test_df = pd.read_csv('../data/test_numeric.csv', usecols=['Id'])

    # Set Response variable for samples in test data to 0.
    test_df['Response'] = 0

    # Concatenate training and test samples and sort by Id.
    concat_df = pd.concat([train_df, test_df]).sort_values(by='Id').reset_index(drop=True)

    # Create column IsPreviousDefective to check if the previous Id was defective i.e. Response = 1
    concat_df['IsPreviousDefective'] = 0
    # Create column IsNextDefective to check if the next Id was defective i.e. Response = 1
    concat_df['IsNextDefective'] = 0

    # Iterate through all rows
    for index in range(concat_df.shape[0]):
        if concat_df.Response[index] == 1:
            concat_df.IsPreviousDefective[index + 1] = 1
            concat_df.IsNextDefective[index - 1] = 1

    # Drop the Response variable because we don't need it anymore.
    concat_df = concat_df.drop(labels='Response', axis=1)
    # Write to csv.
    concat_df.to_csv('../data/consecutive_response.csv', index=False)

def order_features():
    train_df = pd.read_csv('../data/train_numeric.csv', usecols=['Id', 'Response'])
    test_df = pd.read_csv('../data/test_numeric.csv', usecols=['Id'])

    train_df['StartTime'] = -1
    test_df['StartTime'] = -1

    train_df['EndTime'] = -1
    test_df['EndTime'] = -1

    total_rows = max(NUM_TRAINING, NUM_TEST)
    rows_parsed = 0
    chunk_size = 50000

    # Last training chunk will have one row less than the last test chunk
    for train_date, test_date in zip(pd.read_csv('../data/train_date.csv', chunksize=chunk_size), pd.read_csv('../data/test_date.csv', chunksize=chunk_size)):
        features = list(train_date.columns)
        features.remove('Id')

        # StartTime is min timestamp
        train_start_time = train_date[features].min(axis=1).values
        test_start_time = test_date[features].min(axis=1).values

        # Set StartTime
        train_df.loc[train_df.Id.isin(train_date.Id), 'StartTime'] = train_start_time
        test_df.loc[test_df.Id.isin(test_date.Id), 'StartTime'] = test_start_time

        # EndTime is max timestamp
        train_end_time = train_date[features].max(axis=1).values
        test_end_time = test_date[features].max(axis=1).values

        # Set EndTime
        train_df.loc[train_df.Id.isin(train_date.Id), 'EndTime'] = train_end_time
        test_df.loc[test_df.Id.isin(test_date.Id), 'EndTime'] = test_end_time

        rows_parsed += chunk_size
        print('Rows parsed: ' + str(rows_parsed))
        if rows_parsed >= total_rows:
            break

    # Concatenate train and test dataframes to perform operations on both the datasets.
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

    train_df.to_csv('../data/train_order.csv', index=False)
    test_df.to_csv('../data/test_order.csv', index=False)
    

def datetime_features():
    # From the shopfloor visualization, if we look at Station 30, it is connected to stations 31 and 32, 
    # which respectively have the smallest and largest error rates
    # This makes station 30 suspicious
    # We take those datetime features from 
    # which had the maximum number of non-NA values.
    
    train_date_temp = pd.read_csv('../data/train_date.csv', nrows=1)
    cols = list(train_date_temp.columns)
    station30_cols = [c for c in cols if "_S30_" in c]
    train_date = pd.read_csv('../data/train_date.csv', usecols = station30_cols)
    # Choose those columns which have the maximum number of non-NA values.
    important_cols = (train_date.count() == train_date.count().max()).reset_index()
    important_cols = important_cols.loc[important_cols[0] == True]
    important_cols = important_cols.drop(labels=0, axis=1)
    important_cols.columns = ['ImportantFeatures']
    important_cols.to_csv('../data/datetime_features.csv', index=False)

def numeric_features():
    series_list = []

    skip_rows = 0
    n_rows = 300000
    index = 0
    while index < 4:
        train_numeric = pd.read_csv('../data/train_numeric.csv', index_col=0, usecols=list(range(969)),
                                    nrows=n_rows, skiprows=skip_rows).fillna(9999999)
        y = pd.read_csv("../data/train_numeric.csv", index_col=0, usecols=[0,969]).loc[train_numeric.index].values.ravel()
        X = train_numeric.values

        print('Fitting Begins for iteration ' + str(index))
        clf = XGBClassifier(base_score= 0.0058, max_depth = 6, learning_rate = 0.05, seed = 1234, nthread=4)
        clf.fit(X, y, eval_metric = 'auc')
        print('Fitting Ends for iteration ' + str(index))

        plt.bar(range(len(clf.feature_importances_[clf.feature_importances_>0.006])), clf.feature_importances_[clf.feature_importances_>0.006])
        plt.xlabel('Feature Indices')
        plt.ylabel('Feature Importance Score')

        plt.plot()
        numeric_indices = np.where(clf.feature_importances_>0.006)[0]
        print(numeric_indices)
        print(len(numeric_indices))

        series_list.append(numeric_indices)

        index += 1
        skip_rows += n_rows

    numeric_indices = np.concatenate(series_list, axis=0)
    numeric_indices = np.unique(numeric_indices)
    numeric_indices = pd.DataFrame(numeric_indices)
    numeric_indices.columns = ['ImportantFeatures']
    numeric_indices.to_csv('../data/numeric_indices.csv', index=False)

def production_path_features():
    # From the shop visualization,
    # There are four different path flows,
    # One production path goes through Station 0,
    # One production path goes from Station 13,
    # One production path goes from Station 38,
    # One production path goes from Station 47.

    # In Station 0, L0_S0_D1 has largest number of non-NA values.
    # In Station 13, L0_S13_D355 has largest number of non-NA values.
    # In Station 38, L3_S38_D3953 has largest number of non-NA values.
    # In Station 47, L3_S47_D4150 has largest number of non-NA values.

    # This was found by loading the dataset with only those columns that had
    # features belonging to a particular station.
    # To find the column/feature in a particular station with maximum non-NA values
    # we simply looked at the output of df.count()

    train_path = pd.read_csv('../data/train_date.csv', usecols=['Id', 'L0_S0_D1', 'L0_S13_D355', 'L3_S38_D3953', 'L3_S47_D4150']).fillna(0)
    train_path['through_path_one'] = np.where(train_path['L0_S0_D1'] > 0, 1, 0)
    train_path['through_path_two'] = np.where(train_path['L0_S13_D355'] > 0, 1, 0)
    train_path['through_path_three'] = np.where(train_path['L3_S38_D3953'] > 0, 1, 0)
    train_path['through_path_four'] = np.where(train_path['L3_S47_D4150'] > 0, 1, 0)
    train_path = train_path.drop(labels=['L0_S0_D1', 'L0_S13_D355', 'L3_S38_D3953', 'L3_S47_D4150'], axis=1)

    test_path = pd.read_csv('../data/test_date.csv', usecols=['Id', 'L0_S0_D1', 'L0_S13_D355', 'L3_S38_D3953', 'L3_S47_D4150']).fillna(0)
    test_path['through_path_one'] = np.where(test_path['L0_S0_D1'] > 0, 1, 0)
    test_path['through_path_two'] = np.where(test_path['L0_S13_D355'] > 0, 1, 0)
    test_path['through_path_three'] = np.where(test_path['L3_S38_D3953'] > 0, 1, 0)
    test_path['through_path_four'] = np.where(test_path['L3_S47_D4150'] > 0, 1, 0)
    test_path = test_path.drop(labels=['L0_S0_D1', 'L0_S13_D355', 'L3_S38_D3953', 'L3_S47_D4150'], axis=1)

    train_path.to_csv('../data/train_path.csv', index=False)
    test_path.to_csv('../data/test_path.csv', index=False)

def station32_features():
    # Get those features for station 32, 33 and 34 which have the maximum number of non-NA values
    useful_stations = ['S32', 'S33', 'S34']

    train_date_temp = pd.read_csv('../data/train_date.csv', nrows=1)

    date_cols = list(train_date_temp.columns)
    date_cols.remove('Id')

    useful_cols = [x for x in date_cols if x.split('_')[1] in useful_stations]
    useful_cols.append('Id')

    df = pd.read_csv('../data/train_date.csv', usecols=useful_cols).drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    df['Station'] = df['index'].apply(lambda x: x.split('_')[1])

    cols = df.drop_duplicates('Station', keep='first')['index'].tolist()
    cols.append('Id')

    train_date = pd.read_csv('../data/train_date.csv', usecols=cols)
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
    train_date.to_csv('../data/train_station32_features.csv', index=False)

    test_date = pd.read_csv('../data/test_date.csv', usecols=cols)
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
    test_date.to_csv('../data/test_station32_features.csv', index=False)

def run():
    start = time.time()
    consecutive_response_features()
    order_features()
    datetime_features()
    numeric_features()
    production_path_features()
    station32_features()
    print(time.time() - start)

run()
#l = pd.read_csv('../data/train_date.csv', nrows=1)
#cols = list(l.columns)
#cols = [c for c in cols if "_S30_" in c]
#df = pd.read_csv('../data/train_date.csv', usecols=cols)
#q = df.count() == df.count().max()
#q = q.reset_index()
