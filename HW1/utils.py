import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Data_preprocess():
    data = pd.read_csv('thsr_raw0.csv')
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m').dt.strftime('%Y%m').astype(int)
    # print(data)
    # raise NotImplementedError()

    # 疏運期間天數
    special = {
        202201: 8,
        202202: 11,
        202203: 1,
        202204: 8,
        202205: 7,
        202206: 5,
        202207: 0,
        202208: 0,
        202209: 5,
        202210: 5,
        202211: 0,
        202212: 2,
        202301: 16,
        202302: 5,
        202303: 2,
        202304: 9,
        202305: 6,
        202306: 6,
        202307: 0,
        202308: 0,
        202309: 4,
        202310: 8,
        202311: 0,
        202312: 3,
        202401: 2,
        202402: 10,
        202403: 0,
        202404: 6,
        202405: 4,
        202406: 5,
        202407: 0,
        202408: 0,
        202409: 6,
        202410: 6,
        202411: 0,
        202412: 0,
        202501: 9
    }
    data['special days'] = data['date'].map(special)
    # print(data.head())

    # split data
    # train data = 2022-01 - 2023-12
    # validate data = 2024-01 - 2024-06
    # test data = 2024-07 - 2025-01
    train_data = data[data['date'] <= 202312].reset_index(drop=True)
    validate_data = data[(data['date'] >= 202401) & (data['date'] <= 202406)].reset_index(drop=True)
    test_data = data[data['date'] >= 202407].reset_index(drop=True)

    # 車站編號 = 每月平均人數 (from train data)
    station_avg = train_data.groupby(['station id'])['crowd number'].mean().reset_index()
    station_avg.columns = ['station id', 'avg']

    train_data = pd.merge(train_data, station_avg, on='station id', how='left')
    train_data['station id'] = train_data['avg']
    train_data.drop(columns=['avg'], inplace=True)

    validate_data = pd.merge(validate_data, station_avg, on='station id', how='left')
    validate_data['station id'] = validate_data['avg']
    validate_data.drop(columns=['avg'], inplace=True)

    test_data = pd.merge(test_data, station_avg, on='station id', how='left')
    test_data['station id'] = test_data['avg']
    test_data.drop(columns=['avg'], inplace=True)

    # record avg -> station name mapping
    mp = dict(zip(station_avg['station id'], station_avg['avg']))
    avg_to_name = {v:k for k, v in mp.items()}
    # print(avg_to_name)

    print(train_data)

    return train_data, validate_data, test_data, avg_to_name

# Data_preprocess()

def Data_analyze():
    data = pd.read_csv('thsr_exp.csv')
    # data['date'] = pd.to_datetime(data['date'], format='%Y-%m').dt.strftime('%Y%m').astype(int)
    mp = {'南港': 0, '台中': 1, '台北': 2, '台南': 3, '嘉義': 4, '左營': 5, '彰化': 6, '新竹': 7, '板橋': 8, '桃園': 9, '苗栗': 10, '雲林': 11}
    data['station id'] = {v:k for k,v in mp.items()}

    plt.rc('font', family='Microsoft JhengHei')     # show chinese in plt

    # different month's crowd flow
    month_sum = data.groupby(['date'])['crowd number'].sum().reset_index()
    # print(month_sum['date'], month_sum['crowd number'])
    plt.figure(1)
    plt.plot(month_sum['crowd number'])
    plt.xticks(range(len(month_sum['date'])), month_sum['date'], rotation=45)
    plt.title("crowd flow in different month")
    plt.show()

    # different station crowd flow
    station_sum = data.groupby(['station id'])['crowd number'].sum().reset_index()
    # print(station_sum)
    plt.figure(2)
    plt.plot(station_sum['crowd number'])
    plt.xticks(range(len(station_sum['station id'])), station_sum['station id'])
    plt.title("crowd flow in different station")
    plt.show()

# Data_analyze()

def Covid(data):
    cov = pd.read_csv('19CoV.csv')
    cov['date'] = pd.to_datetime(cov['date']).dt.strftime('%Y%m').astype(int)
    cov_sum = cov.groupby(['date'])['disease number'].sum().reset_index()
    new_row = [
        {'date': 202310, 'disease number': 0},
        {'date': 202402, 'disease number': 0},
        {'date': 202406, 'disease number': 0},
        {'date': 202407, 'disease number': 0},
        {'date': 202408, 'disease number': 0},
        {'date': 202409, 'disease number': 0},
        {'date': 202411, 'disease number': 0},
        {'date': 202412, 'disease number': 0},
        {'date': 202501, 'disease number': 0}
    ]
    for r in new_row:
        cov_sum.loc[len(cov_sum)] = r
    data = pd.merge(data, cov_sum[['date', 'disease number']], on='date', how='left')
    # print(data)
    return data

# train_data, validate_data, test_data, _ = Data_preprocess()
# Covid(train_data)