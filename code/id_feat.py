# Script to generate id based features
import pandas as pd
import numpy as np


DATA_DIR = "../data"

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'

CHUNKSIZE = 50000

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    print("File length: ",i+1)
    return i + 1

TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)

TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

NROWS = file_len(TRAIN_NUMERIC)

TRAIN_FILENAME = "./id_feat_train.csv"
TEST_FILENAME = "./id_feat_test.csv"

train = pd.read_csv(TRAIN_NUMERIC, usecols=[ID_COLUMN, TARGET_COLUMN], nrows=NROWS)
test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN], nrows=NROWS)

train["StartTime"] = -1
test["StartTime"] = -1


nrows = 0
for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])
    # Generate starttime for later use
    stime_tr = tr[feats].min(axis=1).values
    stime_te = te[feats].min(axis=1).values

    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te

    nrows += CHUNKSIZE
    if nrows >= NROWS:
        break


ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)

train_test['feat1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)# Id based features, get the difference of the job id in original order
train_test['feat2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)# Id based features, get the difference of the job id in original descending order

train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)

train_test['feat3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int) # Id based features, get the difference of the job id in starttime order
train_test['feat4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int) # Id based features, get the difference of the job id in starttime descending order

train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)
train = train_test.iloc[:ntrain, :]
test = train_test.iloc[ntrain:, :]


train.to_csv(TRAIN_FILENAME, index=False)
test.to_csv(TEST_FILENAME, index=False)