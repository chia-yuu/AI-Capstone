import utils
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_data, validate_data, test_data, avg_to_name = utils.Data_preprocess() # data type =  pd.df
    # train_data = utils.Covid(train_data)
    # validate_data = utils.Covid(validate_data)
    # test_data = utils.Covid(test_data)
    # train_data = train_data.sort_values(by=['station id', 'date'])
    # validate_data = validate_data.sort_values(by=['station id', 'date'])
    # test_data = test_data.sort_values(by=['station id', 'date'])
    # print("train data:")
    # print(train_data.head())
    # raise NotImplementedError()

    train_y = train_data['crowd number']
    train_x = train_data.drop(columns=['crowd number'])
    val_y = validate_data['crowd number']
    val_x = validate_data.drop(columns=['crowd number'])
    test_y = test_data['crowd number']
    test_x = test_data.drop(columns=['crowd number'])
    baseline = test_data['station id']      # baseline = avg of each station

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    baseline = np.array(baseline)

    # train
    print("start training...")
    model = RandomForestRegressor()
    model.fit(train_x, train_y)

    # pred
    print("pred...")
    pred = model.predict(test_x)


    # MSE / RMSE
    mse = mean_squared_error(test_y, pred)
    rmse = math.sqrt(mse)
    mse_base = mean_squared_error(baseline, pred)
    rmse_base = math.sqrt(mse_base)
    print("\n### random forest result ###")
    print("MSE / RMSE")
    print(f"model: MSE = {mse}, RMSE = {rmse}")
    print(f"baseline: MSE = {mse_base}, RMSE = {rmse_base}")
    print(f"improvement(base line rmse - predict rmse): {rmse_base - rmse}")

    # R square
    r_square = model.score(test_x, test_y)
    test_np = np.array(baseline)
    tot_mean = np.mean(baseline)
    ss_tot = np.sum((baseline - tot_mean) ** 2)
    ss_res = np.sum((baseline - pred) ** 2)
    r_base = 1 - (ss_res / ss_tot)
    print("\nR square")
    print(f"model: R square = {r_square}")
    print(f"baseline: R square = {r_base}")
    print(f"improvement(predict - base line): {r_square - r_base}")

    # MAPE
    mape = mean_absolute_percentage_error(test_y, pred)
    mape_base = mean_absolute_percentage_error(baseline, pred)
    print("\nMAPE")
    print(f"model: MAPE = {mape}")
    print(f"baseline: MAPE = {mape_base}")
    print(f"improvement(base line - predict): {mape_base - mape}")
    # raise NotImplementedError()

    # print("test x")
    # print(test_x[:, 1])
    # raise NotImplementedError()

    # save result to csv & visualize
    test_date = test_x[:, 0]
    data = list(zip(pd.Series(test_date), pd.Series(test_x[:, 1]).map(avg_to_name), pred, test_y))
    # data = list(zip(pred, test_y))
    df = pd.DataFrame(data, columns=['date', 'station', 'predict', 'actual'])
    df.to_csv('random forest result.csv', index=False)

    plt.rc('font', family='Microsoft JhengHei')     # show chinese in plt

    # 只畫新竹
    # df_test_x = pd.DataFrame(test_x, columns=['date', 'station id', 'special days'])
    # test_date = df_test_x[df_test_x['station id'] == 491219.5416666667]['date']   # 選出新竹站
    # draw_x = df[df_test_x['station id'] == 491219.5416666667]       # station | predict | actual
    # test_date = np.array(test_date)
    # draw_x = np.array(draw_x)

    # 畫全部
    plt.figure(1)
    test_station = pd.Series(test_x[:, 1]).map(avg_to_name)
    draw_x = df
    test_station = np.array(test_station)
    draw_x = np.array(draw_x)
    plt.plot(draw_x[:, 3], label='actual', color='blue', alpha=0.8)
    # plt.scatter(range(len(draw_x[:, 2])), draw_x[:, 2], label='predict', color='red', s=10)
    plt.plot(draw_x[:, 2], label='predict', color='red', alpha=0.8)
    # tic_range = range(3, len(test_station), 7)
    # tic_label = test_station[tic_range]
    # plt.xticks(tic_range, tic_label)
    t = np.array(df['date'])
    t = np.round(t/100, 2)
    tic_range = range(6, len(t), 12)
    tic_label = t[tic_range]
    plt.xticks(tic_range, tic_label)
    # plt.xticks(range(len(test_station)), test_station, rotation=90)
    plt.title("random forest - all station 2024-07 ~ 2025-01")
    plt.xlabel("station")
    plt.ylabel("crowd number")
    plt.legend()
    plt.savefig('random forest result.png')
    plt.show()

    # 只畫新竹
    plt.figure(2)
    draw_x = df[df['station'] == '新竹']
    draw_x = np.array(draw_x)
    plt.plot(draw_x[:, 3], label='actual', color='blue', alpha=0.6)
    # plt.plot(draw_x[:, 2], label='predict', color='red', alpha=0.6)
    plt.scatter(range(len(draw_x[:, 2])), draw_x[:, 2], s=10, color='red', label='predict')
    plt.xticks(range(len(draw_x[:, 0])), draw_x[:, 0])   # date as x label
    plt.ylim(300000, 1000000)
    plt.title("random forest - Hsinchu station 2024-07 ~ 2025-01")
    plt.xlabel("date")
    plt.ylabel("crowd number")
    plt.legend()
    plt.savefig('random forest result (新竹).png')
    plt.show()

    # 柱狀圖
    # plt.bar(range(len(df)), df['actual'], width=0.8, label='actual')
    # # plt.bar(range(len(df)), df['predict'], width=0.8, alpha=0.5)
    # plt.scatter(range(len(df)), df['predict'], label='predict')
    # plt.title("random forest - all station 2024-07 ~ 2025-01")
    # plt.xticks(range(len(test_station)), test_station, rotation=90)
    # plt.xlabel("station")
    # plt.ylabel("crowd number")
    # plt.legend()
    # plt.savefig('random forest result.png')
    # plt.show()

'''
### random forest result ###
MSE / RMSE
model: MSE = 1774875998.4185035, RMSE = 42129.27721215382
baseline: MSE = 14129571424.870806, RMSE = 118867.87381319987
improvement(base line rmse - predict rmse): 76738.59660104604

R square
model: R square = 0.9891164362846216

==============================================================

### random forest result ###
MSE / RMSE
model: MSE = 1501070503.2141552, RMSE = 38743.65113427173
baseline: MSE = 14464895772.74277, RMSE = 120270.0950891067
improvement(base line rmse - predict rmse): 81526.44395483498

R square
model: R square = 0.9907954152979908
'''