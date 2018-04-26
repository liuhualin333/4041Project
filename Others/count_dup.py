import pandas as pd
import numpy as np

def train_process():

    csv_file = 'train_numeric.csv'

    temp_df = pd.read_csv('train_numeric.csv', nrows=1)

    cols = temp_df.columns.values
    sub_cols = cols[1:-1].tolist()

    chunk_cnt = 0

    cnt = 0
    end_cnt = 0
    max_dups = 0

    # dataframe for recording rows beside splitting postions for further checking
    # cut_df = pd.DataFrame()
    # end_dup_df = pd.DataFrame()

    reader = pd.read_csv(csv_file, chunksize=99967)
    for chunk in reader:
        chunk_cnt+=1
        # count total number of rows
        cnt+=chunk.shape[0]
        print(chunk.iloc[5806].isnull().sum())
        break


        # find duplicates within chunk
        chunk['is_dup'] = chunk.duplicated(sub_cols, keep=False)
        chunk.loc[chunk['is_dup']==False, 'count_dup'] = 1



        max_index = max(chunk.index.values)

        # rows at splitting positions
        # cut_df = cut_df.append(chunk.loc[min(chunk.index.values)])
        # cut_df = cut_df.append(chunk.loc[max_index])

        # get duplicated rows and find the number of duplicates after first occurrence
        # assume duplicated sets are not consecutive
        # assume max dup number is 30
        dup_df = chunk.loc[chunk['is_dup'] == True]
        dup_df = dup_df.drop_duplicates(subset=sub_cols, keep='first')
        dup_df = dup_df.reset_index()
        no_dup = dup_df.shape[0]
        print('chunk no. %d, find dups %d'%(chunk_cnt, no_dup))

        for row in range(no_dup):

            first_dup = dup_df.loc[row]['index']
            if row == no_dup-1:
                next_dup = min(max_index, first_dup+40)
            else:
                next_dup = dup_df.loc[row+1]['index']

            dups = [first_dup]

            dup_range = min(next_dup-first_dup+1, max_index-first_dup+1)

            count = 1
            for i in range(1, dup_range):
                if chunk.loc[first_dup+i, 'is_dup'] == False:
                    break
                dups.append(first_dup+i)
                count+=1

            chunk.loc[dups, 'count_dup'] = count

            if first_dup+count==max_index:
                # end_dup_df.append(chunk.loc[max_index])
                end_cnt += 1

            if count>max_dups:
                print(dup_df.loc[row]['index'], count)
                max_dups = count


        dup_chunk = chunk.loc[:, ['Id', 'count_dup']]

        dup_chunk.to_csv('train_dup_int.csv', index=False, mode='a')


    # cut_df.to_csv('cut_sides_2.csv')
    # end_dup_df.to_csv('end_dup_2.csv')
    #
    # print(cnt)
    # print(max_dups)
    # print(end_cnt)

    return 'train_dup_int.csv'



def test_process():

    csv_file = 'test_numeric.csv'

    temp_df = pd.read_csv('test_numeric.csv', nrows=1)

    cols = temp_df.columns.values
    sub_cols = cols[1:-1].tolist()

    chunk_cnt = 0

    cnt = 0
    end_cnt = 0
    max_dups = 0

    # dataframe for recording rows beside splitting postions for further checking
    # cut_df = pd.DataFrame()
    # end_dup_df = pd.DataFrame()

    reader = pd.read_csv(csv_file, chunksize=99967)
    for chunk in reader:
        chunk_cnt+=1
        # count total number of rows
        cnt+=chunk.shape[0]


        # find duplicates within chunk
        chunk['is_dup'] = chunk.duplicated(sub_cols, keep=False)
        chunk.loc[chunk['is_dup']==False, 'count_dup'] = 1

        print(chunk.iloc[5806:5809])

        break

        max_index = max(chunk.index.values)

        # rows at splitting positions
        # cut_df = cut_df.append(chunk.loc[min(chunk.index.values)])
        # cut_df = cut_df.append(chunk.loc[max_index])

        # get duplicated rows and find the number of duplicates after first occurrence
        # assume duplicated sets are not consecutive
        # assume max dup number is 30
        dup_df = chunk.loc[chunk['is_dup'] == True]
        dup_df = dup_df.drop_duplicates(subset=sub_cols, keep='first')
        dup_df = dup_df.reset_index()
        no_dup = dup_df.shape[0]
        print('chunk no. %d, find dups %d'%(chunk_cnt, no_dup))

        for row in range(no_dup):

            first_dup = dup_df.loc[row]['index']
            if row == no_dup-1:
                next_dup = min(max_index, first_dup+40)
            else:
                next_dup = dup_df.loc[row+1]['index']

            dups = [first_dup]

            dup_range = min(next_dup-first_dup+1, max_index-first_dup+1)

            count = 1
            for i in range(1, dup_range):
                if chunk.loc[first_dup+i, 'is_dup'] == False:
                    break
                dups.append(first_dup+i)
                count+=1

            chunk.loc[dups, 'count_dup'] = count

            if first_dup+count==max_index:
                # end_dup_df.append(chunk.loc[max_index])
                end_cnt += 1

            if count>max_dups:
                print(dup_df.loc[row]['index'], count)
                max_dups = count


        dup_chunk = chunk.loc[:, ['Id', 'count_dup']]

        dup_chunk.to_csv('test_dup_int.csv', index=False, mode='a')


    # cut_df.to_csv('cut_sides_2.csv')
    # end_dup_df.to_csv('end_dup_2.csv')

    # print(cnt)
    # print(max_dups)
    # print(end_cnt)

    return 'test_dup_int.csv'



file = train_process()

df = pd.read_csv(file)

print(df.shape)
df = df.drop(df[df["Id"] == 'id'].index)
df['count_dup'].fillna(1, inplace=True)
df = df.drop(df[pd.to_numeric(df['count_dup'], errors='coerce').isnull()].index)
print(df.shape)
df[['Id']] = df[['Id']].astype(float)
df[['Id']] = df[['Id']].astype(int)

df[['count_dup']] = df[['count_dup']].astype(float)
df[['count_dup']] = df[['count_dup']].astype(int)

print(df.dtypes)

df.to_csv(file, index=False, mode='w')
