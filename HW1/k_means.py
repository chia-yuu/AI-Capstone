import os
os.environ["OMP_NUM_THREADS"] = "2"     # avoid warning: KMeans has a mem leak

import utils
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":
    k = 100
    train_data, validate_data, test_data, avg_to_name = utils.Data_preprocess() # data type =  pd.df
    # train_data = utils.Covid(train_data)
    # validate_data = utils.Covid(validate_data)
    # test_data = utils.Covid(test_data)
    # train_data = train_data.sort_values(by=['station id', 'date'])
    # validate_data = validate_data.sort_values(by=['station id', 'date'])
    # test_data = test_data.sort_values(by=['station id', 'date'])
    # print(test_data)
    # raise NotImplemented

    # train
    train_x = train_data.drop(columns=['crowd number'])
    scaler = StandardScaler()
    train_x_scale = scaler.fit_transform(train_x)

    # print(train_x)
    # print("\n")
    # print(train_x_scale)

    kmeans = KMeans(n_clusters=k)
    train_data['station group'] = kmeans.fit_predict(train_x_scale)
    # print(train_data)

    group_mean = train_data.groupby('station group')['crowd number'].mean().reset_index()
    # print(group_mean)

    # pred
    fs = scaler.transform(test_data[["date", "station id", "special days"]])
    test_data['station group'] = kmeans.predict(fs)
    pred = test_data['station group'].map(group_mean.set_index('station group')['crowd number'])
    print(test_data)

    # save result to csv
    test_data['station id'] = test_data['station id'].map(avg_to_name)
    data = list(zip(test_data['date'], test_data['station id'], pred, test_data['crowd number']))
    df = pd.DataFrame(data, columns=['date', 'station', 'predict', 'actual'])
    df.to_csv('k means result.csv', index=False)

    # MSE / RMSE
    mse = mean_squared_error(test_data['crowd number'], pred)
    rmse = np.sqrt(mse)
    mse_base = 14129571424.870806   # from random forest
    rmse_base = 118867.87381319987  # from random forest
    print(f"\n### k means result (k = {k}) ###")
    print("MSE / RMSE")
    print(f"model: MSE = {mse}, RMSE = {rmse}")
    print(f"baseline: MSE = {mse_base}, RMSE = {rmse_base}")
    print(f"improvement(base line rmse - predict rmse): {rmse_base - rmse}")

    # R square
    test_np = np.array(test_data)
    tot_mean = np.mean(test_np[:, 2])
    ss_tot = np.sum((test_np[:, 2] - tot_mean) ** 2)
    ss_res = np.sum((test_np[:, 2] - pred) ** 2)
    r_square = 1 - (ss_res / ss_tot)
    r_base = 0.8668243182542058
    print("\nR square")
    print(f"model: R square = {r_square}")
    print(f"baseline: R square = {r_base}")
    print(f"improvement(predict - base line): {r_square - r_base}")
    
    # MAPE
    mape = mean_absolute_percentage_error(test_data['crowd number'], pred)
    mape_base = 0.21791360470522392
    print("\nMAPE")
    print(f"model: MAPE = {mape}")
    print(f"baseline: MAPE = {mape_base}")
    print(f"improvement(base line - predict): {mape_base - mape}")
    # raise NotImplementedError()

    # visualize
    plt.rc('font', family='Microsoft JhengHei')     # show chinese in plt

    # all station (line)
    plt.figure(1)
    plt.plot(np.array(test_data['crowd number']), label='actual', color='blue', alpha=0.8)
    # plt.scatter(range(len(pred)), pred, label='predict', color='red')
    plt.plot(np.array(pred), label='predict', color='red', alpha=0.8)
    # tic_range = range(3, len(test_data['station id']), 7)
    # tic_label = test_data['station id'][0:len(test_data['station id'])].iloc[tic_range]
    # plt.xticks(tic_range, tic_label)
    t = np.array(df['date'])
    t = np.round(t/100, 2)
    tic_range = range(6, len(t), 12)
    tic_label = t[tic_range]
    plt.xticks(tic_range, tic_label)
    plt.title("k means (k = 150) - all station 2024-07 ~ 2025-01")
    plt.xlabel('station')
    plt.ylabel('crowd number')
    plt.legend()
    plt.savefig('k means result.png')
    plt.show()

    # hsinchu station (point)
    plt.figure(2)
    draw_x = df[df['station'] == '新竹']
    draw_x = np.array(draw_x)
    plt.plot(draw_x[:, 3], label='actual', color='blue', alpha=0.6)
    # plt.plot(draw_x[:, 2], label='predict', color='red', alpha=0.6)
    plt.scatter(range(len(draw_x[:, 2])), draw_x[:, 2], s=10, color='red', label='predict')
    plt.xticks(range(len(draw_x[:, 0])), draw_x[:, 0])   # date as x label
    plt.ylim(300000, 1000000)
    plt.title("k means - Hsinchu station 2024-07 ~ 2025-01")
    plt.xlabel("date")
    plt.ylabel("crowd number")
    plt.legend()
    plt.savefig('k means result (新竹).png')
    plt.show()