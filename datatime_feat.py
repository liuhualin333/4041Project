import pandas as pd
import numpy as np
import multiprocessing

def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	print("File length: ",i+1)
	return i + 1

def traverse_dataframe(filename,df,output_filename,mode,feat_set,feat_names, num_rows):
	CHUNKSIZE = 100000
	for feat in feat_set:
		print(mode,feat)
		# Get corresponding cols
		col_list = [name for name in feat_names if feat in name]
		col_list.append('Id')
		# Use relevant cols
		_df = pd.read_csv(filename, usecols=col_list, chunksize=CHUNKSIZE)
		nrows = 0
		if(feat not in ['L0', 'L1', 'L2', 'L3']):
			df[feat] = np.NaN
			df[feat+"_max-min"] = np.NaN
		else:
			df[feat+"_start"] = np.NaN
			df[feat+"_end"] = np.NaN
			df[feat+"_max-min"] = np.NaN
		for frame in _df:
			feats = np.setdiff1d(col_list, ['Id'])
			if(feat not in ['L0', 'L1', 'L2', 'L3']):
				stime = frame[feats].min(axis=1).values
				etime = frame[feats].max(axis=1).values
				df.loc[df.Id.isin(frame.Id),feat+"_max-min"] = etime - stime
				df.loc[df.Id.isin(frame.Id),feat] = frame[feats].mean(axis=1).values
			else:
				stime = frame[feats].min(axis=1).values
				etime = frame[feats].max(axis=1).values
				df.loc[df.Id.isin(frame.Id),feat+"_start"] = stime
				df.loc[df.Id.isin(frame.Id),feat+"_end"] = etime
				df.loc[df.Id.isin(frame.Id),feat+"_max-min"] = etime - stime
			nrows += CHUNKSIZE
			print(feat,nrows)
			if nrows >= num_rows:
				break

	with open(output_filename, 'a') as f:
		df.to_csv(f, index=False)

if __name__ == "__main__":
	DATA_DIR = '.'

	TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
	TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)

	TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
	TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

	PROCESSED_TRAIN_FILENAME = "{0}/datetime_train.csv".format(DATA_DIR)
	PROCESSED_TEST_FILENAME = "{0}/datetime_test.csv".format(DATA_DIR)

	NROWS_TRAIN = file_len(TRAIN_DATE)
	NROWS_TEST = file_len(TEST_DATE)
	# NROWS = 200

	# Get feat_set
	sample = pd.read_csv(TRAIN_DATE, nrows=1)
	train = pd.read_csv(TRAIN_DATE, usecols=['Id'], nrows=NROWS_TRAIN)
	test = pd.read_csv(TEST_DATE, usecols=['Id'], nrows=NROWS_TEST)
	feat_names = list(sample.columns.values)
	feat_set = []
	for feat in feat_names:
		feat_compo = feat.split("_")
		if len(feat_compo) == 3:
			if (not feat_compo[0] in feat_set):
				feat_set.append(feat_compo[0])
			if (not feat_compo[1] in feat_set):
				feat_set.append(feat_compo[1])

	p1 = multiprocessing.Process(target=traverse_dataframe, args=(TRAIN_DATE,train,PROCESSED_TRAIN_FILENAME,'tr',feat_set,feat_names,NROWS_TRAIN,))
	p1.start()
	p1 = multiprocessing.Process(target=traverse_dataframe, args=(TEST_DATE,test,PROCESSED_TEST_FILENAME,'te',feat_set,feat_names,NROWS_TEST,))
	p1.start()