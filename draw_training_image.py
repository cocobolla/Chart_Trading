import keras
from keras import layers, models
from keras import backend
from keras import datasets
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from mpl_finance import candlestick2_ohlc
import matplotlib.ticker as ticker


def draw_chart(csv_path, output_dir, labeling_ma=5, window=60):
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'])
    df = df.sort_index(ascending=True)
    scaler = (df['Price'].max() - df['Price'].min()) / len(df)
    print("Max: {:.2f}, Min:{:.2f}".format(df['Price'].max(), df['Price'].min()))
    print("Date: {}".format(len(df)))
    print("Scaler: {:.2f}".format(scaler))

    for i in range(labeling_ma, len(df) - window - labeling_ma):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        target_df = df.loc[df.index[i]: df.index[i + window], :]
        ax.axis('off')
        candlestick2_ohlc(ax, target_df['Open'], target_df['High'], target_df['Low'], target_df['Price'],
                          width=0.5, colorup='r', colordown='b')

        # close_now = df.loc[df.index[i + window], 'Price']
        # close_future = df.loc[df.index[i + window + labeling_ma], 'Price']
        # ma_gradient = round((close_future - close_now) / labeling_ma, 2)
        future_ma = df.loc[df.index[i + window]:df.index[i + window + labeling_ma], 'Price'].mean()
        ma = df.loc[df.index[i + window - labeling_ma]: df.index[i+window], 'Price'].mean()
        ma_gradient = (future_ma - ma)/labeling_ma
        ma_gradient / scaler
        ma_gradient = round(ma_gradient, 2)
        tag = str(ma_gradient)
        output_name = str(i) + '_' + tag + '.png'
        output_path = os.path.join(output_dir, output_name)

        print("Printing {}th image".format(i - labeling_ma + 1))
        fig.savefig(output_path)
        plt.close(fig)


def psar(barsdata, iaf=0.02, maxaf=0.2):
    length = len(barsdata)
    dates = list(barsdata['Date'])
    high = list(barsdata['High'])
    low = list(barsdata['Low'])
    close = list(barsdata['Close'])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]

    for i in range(2, length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])

        reverse = False

        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf

        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]

    return {"dates": dates, "high": high, "low": low, "close": close, "psar": psar, "psarbear": psarbear,
            "psarbull": psarbull}


if __name__ == '__main__':
    root_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading'
    data_dir = 'data'
    output_dir = 'Future_ma20_Labeling_scaling'
    csv_path = os.path.join(root_path, data_dir, 'TSLA Historical Data.csv')
    output_dir_path = os.path.join(root_path, output_dir)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    draw_chart(csv_path=csv_path, output_dir=output_dir_path, labeling_ma=20, window=60)
