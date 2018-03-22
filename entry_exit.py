import pandas as pd

def run(train_date):
    station_times = get_station_times(train_date, withId=True).sort_values(by=['Id', 'station'])
    path = station_times.groupby(['Id', 'station']).min()['time'].to_frame().reset_index()
    #print(path)
    entry_exit = get_entry_exit(path[['Id','station']].copy())
    print(entry_exit) 
    entry_exit.to_csv('entry_exit.csv', index=True, sep=',', mode='a')
    
for chunk in pd.read_csv('train_date.csv', chunksize = 10000): 
    run(chunk) 

# Let's get job with station and time
def get_station_times(dates, withId=False):
    times = []
    cols = list(dates.columns)
    if 'Id' in cols:
        cols.remove('Id')
    for feature_name in cols:
        if withId:
            df = dates[['Id', feature_name]].copy()
            df.columns = ['Id', 'time']
        else:
            df = dates[[feature_name]].copy()
            df.columns = ['time']
        df['station'] = int(feature_name.split('_')[1][1:])
        df = df[['Id', 'station', 'time']]
        df = df.dropna()
        times.append(df)
    return pd.concat(times)

def get_entry_exit(station):
    df = station.groupby('Id').agg(['first', 'last'])
    df.rename(columns={'first': 'Entry', 'last': 'Exit'}, inplace=True) 
    return df