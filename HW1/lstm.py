import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seq_sz = 5
    ep = 100
    bs = 32
    train_data, validate_data, test_data, avg_to_name = utils.Data_preprocess() # data type =  pd.df
    train_data = train_data.sort_values(by=['station id', 'date'])
    validate_data = validate_data.sort_values(by=['station id', 'date'])
    test_data = test_data.sort_values(by=['station id', 'date'])

    train_y = train_data['crowd number']
    train_x = train_data.drop(columns=['crowd number'])
    val_y = validate_data['crowd number']
    val_x = validate_data.drop(columns=['crowd number'])
    test_y = test_data['crowd number']
    test_x = test_data.drop(columns=['crowd number'])
    baseline = test_data['station id']

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    baseline = np.array(baseline)

    scaler_x = StandardScaler()
    train_x_scale = scaler_x.fit_transform(train_x)
    val_x_scale = scaler_x.transform(val_x)
    test_x_scale = scaler_x.transform(test_x)

    scaler_y = StandardScaler()
    train_y_scale = scaler_y.fit_transform(train_y.reshape(-1, 1))
    val_y_scale = scaler_y.transform(val_y.reshape(-1, 1))
    test_y_scale = scaler_y.transform(test_y.reshape(-1, 1))

    # time window
    def creat_seq(x, y, sz=3):
        seq_x, seq_y = [], []
        for i in range(len(x) - sz):
            seq_x.append(x[i:i+sz])
            seq_y.append(y[i+sz])
        return np.array(seq_x), np.array(seq_y)
    
    # prepare for LSTM
    train_x_seq, train_y_seq = creat_seq(train_x_scale, train_y_scale, seq_sz)
    val_x_seq, val_y_seq = creat_seq(val_x_scale, val_y_scale, seq_sz)
    test_x_seq, _ = creat_seq(test_x_scale, test_y_scale, seq_sz)

    # print("train y seq")
    # print(train_y_seq)
    # print(train_y_seq.shape)
    # raise NotImplementedError()

    # (samples, time steps, features)
    # train_x_seq = train_x_seq.reshape(train_x_seq.shape[0], train_x_seq.shape[1], 3)
    # val_x_seq = val_x_seq.reshape(val_x_seq.shape[0], val_x_seq.shape[1], 3)
    # test_x_seq = test_x_seq.reshape(test_x_seq.shape[0], test_x_seq.shape[1], 3)

    train_x_seq = train_x_seq.reshape(train_x_seq.shape[0], seq_sz, train_x_seq.shape[2])
    val_x_seq = val_x_seq.reshape(val_x_seq.shape[0], seq_sz, val_x_seq.shape[2])
    test_x_seq = test_x_seq.reshape(test_x_seq.shape[0], seq_sz, test_x_seq.shape[2])

    # print(train_y_seq.shape)

    # train
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True),
        LSTM(128, activation='tanh'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x_seq, train_y_seq, epochs=ep, batch_size=bs, validation_data=(val_x_seq, val_y_seq), verbose=0)
    # model.save(f'LSTM model (seq{seq_sz}, ep{ep}, bs{bs}).h5')
    
    # pred
    pred = model.predict(test_x_seq)
    # pred = scaler_y.inverse_transform(np.concatenate([pred, np.zeros((pred.shape[0], 2))], axis=1))[:, 0]
    pred = scaler_y.inverse_transform(pred)

    # MSE / RMSE
    # test_y = test_data['crowd number'][seq_sz:]
    test_y = test_y[seq_sz:].reshape(-1, 1)
    mse = mean_squared_error(test_y, pred)
    rmse = np.sqrt(mse)
    mse_base = 14129571424.870806   # from random forest
    rmse_base = 118867.87381319987  # from random forest
    print(f"\n### LSTM result (epoch = {ep}, batch = {bs}, seq_sz = {seq_sz}) ###")
    print("MSE / RMSE")
    print(f"model: MSE = {mse}, RMSE = {rmse}")
    print(f"baseline: MSE = {mse_base}, RMSE = {rmse_base}")
    print(f"improvement(base line rmse - predict rmse): {rmse_base - rmse}")

    # R square
    tot_mean = np.mean(test_y)
    ss_tot = np.sum((test_y - tot_mean) ** 2)
    ss_res = np.sum((test_y - pred) ** 2)
    r_square = 1 - (ss_res / ss_tot)
    print("\nR square")
    print(f"model: R square = {r_square}")

    # MAPE
    mape = mean_absolute_percentage_error(test_y, pred)
    mape_base = 0.21791360470522392
    print("\nMAPE")
    print(f"model: MAPE = {mape}")
    print(f"baseline: MAPE = {mape_base}")
    print(f"improvement(base line - predict): {mape_base - mape}")
    # raise NotImplementedError()

    # save res
    test_data['station id'] = test_data['station id'].map(avg_to_name)
    data = list(zip(test_data['date'], test_data['station id'], pred.flatten(), test_data['crowd number']))
    df = pd.DataFrame(data, columns=['date', 'station', 'predict', 'actual'])
    df.to_csv(f'LSTM result (seq{seq_sz}, ep{ep}, bs{bs}).csv', index=False)

    # visualize
    plt.rc('font', family='Microsoft JhengHei')     # show chinese in plt
    # 折線圖 all station
    plt.figure(1)
    plt.plot(np.array(test_y), label='actual', color='blue', alpha=0.8)
    # plt.scatter(range(len(pred)), pred, label='predict', color='red')
    plt.plot(np.array(pred), label='predict', color='red', alpha=0.8)
    # tic_range = range(0, len(test_y), 7)
    # tic_label = test_data['station id'][:len(test_y)].iloc[tic_range]
    # plt.xticks(tic_range, tic_label)
    t = np.array(df['date'])
    t = np.round(t/100, 2)
    tic_range = range(6, len(t), 12)
    tic_label = t[tic_range]
    plt.xticks(tic_range, tic_label)
    plt.title(f"LSTM (seq{seq_sz}, ep{ep}, bs{bs}) - all station 2024-07 ~ 2025-01")
    plt.xlabel('station')
    plt.ylabel('crowd number')
    plt.legend()
    plt.savefig('LSTM result.png')
    plt.show()

    # scatter, hsinchu
    plt.figure(2)
    draw_x = df[df['station'] == '新竹']
    draw_x = np.array(draw_x)
    plt.plot(draw_x[:, 3], label='actual', color='blue', alpha=0.6)
    # plt.plot(draw_x[:, 2], label='predict', color='red', alpha=0.6)
    plt.scatter(range(len(draw_x[:, 2])), draw_x[:, 2], s=10, color='red', label='predict')
    plt.xticks(range(len(draw_x[:, 0])), draw_x[:, 0])   # date as x label
    plt.ylim(300000, 1000000)
    plt.title("LSTM - Hsinchu station 2024-07 ~ 2025-01")
    plt.xlabel("date")
    plt.ylabel("crowd number")
    plt.legend()
    plt.savefig('LSTM result (新竹).png')
    plt.show()

    # 柱狀圖
    '''
    pivot = df.iloc.pivot_table(index='station', columns='date', values=['predict', 'actual'])
    pivot.plot(kind='bar', width=1.0, figsize=(12, 8))
    # plt.bar(range(len(pivot['predict'].values.flatten())), pivot['predict'].values.flatten(), width=0.5, alpha=0.5, label='predict')
    # plt.bar(range(len(pivot['actual'].values.flatten())), pivot['actual'].values.flatten(), width=0.5, alpha=1.0, label='actual')
    plt.title(f"LSTM (seq{seq_sz}, ep{ep}, bs{bs}) - all station 2024-07 ~ 2025-01")
    plt.xlabel('station')
    plt.ylabel('crowd number')
    # plt.xticks(range(len(test_data)), test_data['station id'])
    plt.legend()
    plt.savefig('LSTM result.png')
    plt.show()
    '''

    # 新竹站柱狀圖
    # plt.plot(np.array(test_data['crowd number']), label='actual', color='blue', alpha=0.8)
    # # plt.scatter(range(len(pred)), pred, label='predict', color='red')
    # plt.plot(np.array(pred), label='predict', color='red', alpha=0.8)
    # tic_range = range(0, len(test_y), 5)
    # tic_label = test_data['station id'][:len(test_y)].iloc[tic_range]
    # plt.xticks(tic_range, tic_label, rotation=90)
    # plt.title(f"LSTM (seq{seq_sz}, ep{ep}, bs{bs}) - all station 2024-07 ~ 2025-01")
    # plt.xlabel('station')
    # plt.ylabel('crowd number')
    # plt.legend()
    # plt.savefig('LSTM result.png')
    # plt.show()