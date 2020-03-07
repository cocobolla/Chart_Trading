import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from mpl_finance import candlestick2_ohlc
import matplotlib.ticker as ticker
import datetime

from chart_drawer import Drawer
from backtest import PsarBacktesting


def draw_train_image(csv_path, output_dir, window, labeling_ma):
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'])
    df = df.sort_index(ascending=True)

    for i in range(0, len(df) - window - labeling_ma):
        print("Printing {}th image".format(i + 1))
        target_df = df.loc[df.index[i]: df.index[i + window], :]

        # Calculate Gradient
        future_ma = df.loc[df.index[i + window]:df.index[i + window + labeling_ma], 'Price'].mean()
        ma = df.loc[df.index[i + window - labeling_ma]: df.index[i + window], 'Price'].mean()

        """
        ma_gradient = (future_ma - ma)/labeling_ma
        ma_gradient / scaler
        """
        ma_gradient = (future_ma - ma) / (target_df['Price'].max() - target_df['Price'].min())
        ma_gradient = round(ma_gradient, 2)
        tag = str(ma_gradient)
        output_name = '_'.join([str(i), str(window), str(labeling_ma), tag]) + '.png'
        output_path = os.path.join(output_dir, output_name)

        drawer = Drawer()
        drawer.draw_chart(target_df, coordinate=False, ohlc=True, psar=False, cum_return=False)
        drawer.save_img(output_path)


def draw_val_image(csv_path, output_dir, window, labeling_ma):
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'])
    df = df.sort_index(ascending=True)

    # Get Backtesting Result
    backtesting = PsarBacktesting(window=window, labeling_ma=labeling_ma)
    df = backtesting.ma_signal(df)
    ret_lim = (df['Total_return'].min(), df['Total_return'].max())

    for i in range(0, len(df) - window - labeling_ma):
        print("Printing {}th image".format(i + 1))
        target_df = df.loc[df.index[i]: df.index[i + window], :]

        # Calculate Gradient
        future_ma = df.loc[df.index[i + window]:df.index[i + window + labeling_ma], 'Price'].mean()
        ma = df.loc[df.index[i + window - labeling_ma]: df.index[i + window], 'Price'].mean()

        """
        ma_gradient = (future_ma - ma)/labeling_ma
        ma_gradient / scaler
        """
        ma_gradient = (future_ma - ma) / (target_df['Price'].max() - target_df['Price'].min())
        ma_gradient = round(ma_gradient, 2)
        tag = str(ma_gradient)
        output_name = '_'.join([str(i), str(window), str(labeling_ma), tag]) + '.png'
        output_path = os.path.join(output_dir, output_name)

        drawer = Drawer()
        drawer.draw_chart(target_df, coordinate=True, ohlc=True, psar=True, cum_return=True, ret_lim=ret_lim)
        drawer.save_img(output_path)


if __name__ == '__main__':
    root_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading'
    data_dir = 'data'
    output_dir = 'Qualcomm_Chart_Test'
    csv_path = os.path.join(root_path, data_dir, 'QCOM Historical Data.csv')
    output_dir_path = os.path.join(root_path, output_dir)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # draw_train_image(csv_path=csv_path, output_dir=output_dir_path, window=60, labeling_ma=10)
    draw_val_image(csv_path=csv_path, output_dir=output_dir_path, window=60, labeling_ma=10)


