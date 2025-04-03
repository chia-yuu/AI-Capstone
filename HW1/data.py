# get raw data from website and save as csv
import requests
import pandas as pd
from lxml import etree
import numpy as np

station_name = ['南港', '台北', '板橋', '桃園', '新竹', '苗栗', '台中', '彰化', '雲林', '嘉義', '台南', '左營']

url = 'https://www.thsrc.com.tw/corp/9571df11-8524-4935-8a46-0d5a72e6bc7c'
req = requests.get(url)
if(req.status_code == 200):
    # get content
    html = etree.HTML(req.text)
    passenger = html.xpath('//table[@id="fixTable"]//td/text()')
    month = html.xpath('//table[@id="fixTable"]//th/text()')

    num = []
    date = []
    station_id = []
    schedule_n = []     # 高鐵每周總班次
    j = 14
    id = 0
    k = 4
    for i in range(len(passenger)):
        # from 2022-1 to 2025-1
        if (month[j] == '2021-12'):
            break
        # only get each station's data, no total
        if(i + 1) % 13 == 0:
            j += 1
            id = 0
            continue

        # number of schedule
        if(k == 0):
            schedule_n.append(1016)
        elif(k == 1):
            schedule_n.append(1025)
            if(month[j] == '2023-06'):
                k -= 1
        elif(k == 2):
            schedule_n.append(1039)
            if(month[j] == '2023-09'):
                k -= 1
        elif(k == 3):
            schedule_n.append(1060)
            if(month[j] == '2023-12'):
                k -= 1
        else:
            schedule_n.append(1103)
            if(month[j] == '2024-06'):
                k -= 1

        tmp = int(passenger[i].replace(",", ""))
        num.append(tmp)
        date.append(month[j])
        station_id.append(station_name[id])
        id += 1

else:
    print('request error!')

# save as csv
data = list(zip(date, station_id, num, schedule_n))
df = pd.DataFrame(data, columns=['date', 'station id', 'crowd number', 'schedule number'])
df.to_csv('thsr_exp.csv', index=False)
df = pd.read_csv('thsr_exp.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m').dt.strftime('%Y%m').astype(int)

# COVID-19
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
df = pd.merge(df, cov_sum[['date', 'disease number']], on='date', how='left')
df.to_csv('thsr_exp.csv', index=False)