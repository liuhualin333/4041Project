# Kurtosis feature generation script

import pandas as pd
import numpy as np

DATA_DIR = "."

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'
CHUNKSIZE = 100000

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    print("File length: ",i+1)
    return i + 1

TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)
TRAIN_FILENAME = "{0}/kurtosis_feat_train.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)
TEST_FILENAME = "{0}/kurtosis_feat_test.csv".format(DATA_DIR)

NROWS = file_len(TRAIN_DATE)

# Result dataframe initialization
train_result = pd.read_csv(TRAIN_DATE, usecols=[ID_COLUMN])
test_result = pd.read_csv(TEST_DATE, usecols=[ID_COLUMN])
nrows = 0

features = pd.read_csv(TRAIN_DATE, nrows=1).drop(['Id'], axis=1).columns.values

# Get line name and station name
def orgainize(features):
    line_features = {}
    lines = set([f.split('_')[0] for f in features])
    
    for l in lines:
        line_features[l] = [f for f in features if l+'_' in f]
            
    return line_features

line_features = orgainize(features)

for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])
    # Whole path Kurtosis
    train_result.loc[train_result.Id.isin(tr.Id), 'Kurtosis'] = tr[feats].kurtosis(axis=1)
    test_result.loc[test_result.Id.isin(te.Id), 'Kurtosis'] = te[feats].kurtosis(axis=1)
    # Kurtosis by line
    for key in line_features.keys():
    	train_result.loc[train_result.Id.isin(tr.Id), 'Kurtosis_'+key] = tr[line_features[key]].kurtosis(axis=1)
    	test_result.loc[test_result.Id.isin(te.Id), 'Kurtosis_'+key] = te[line_features[key]].kurtosis(axis=1)
    nrows += CHUNKSIZE
    if nrows >= NROWS:
        break
# Output to file
train_result.to_csv(TRAIN_FILENAME, index=False)
test_result.to_csv(TEST_FILENAME,index=False)