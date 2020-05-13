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
import tools


def draw_train_image(csv_path, output_dir, window, labeling_ma):
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'], thousands=',')
    df = df.sort_index(ascending=True)

    # Save log message
    log_txt = os.path.join(output_dir, 'log.txt')
    with open(log_txt, 'w') as fp:
        msg = ('Date: {} ~ {}\n'.format(str(df.index[0]).split()[0], str(df.index[-1]).split()[0]) +
               '# of Date: {}\n'.format(len(df)) +
               'Window: {} \n'.format(window) +
               'Labeling Moving Average: {} \n'.format(labeling_ma) +
               'Image Tag Format: {~th img}_{yyyy-mm-dd}_{gradient}.png')
        fp.write(msg)

    total = len(df) - window - (labeling_ma - 1)

    for i in range(0, total):
        print("Printing {}th image".format(i + 1))
        target_df = df.loc[df.index[i]: df.index[i + window - 1], :]

        # Calculate Gradient
        scaler = target_df['Price'].max() - target_df['Price'].min()
        ma_gradient = tools.calculate_gradient(df, i, scaler, window, labeling_ma)
        tag = str(ma_gradient)

        # Tagging the image
        target_date = str(target_df.index[-1]).split(' ')[0]
        output_name = '_'.join([str(i), target_date, tag]) + '.png'
        output_path = os.path.join(output_dir, output_name)

        drawer = Drawer()
        drawer.draw_chart(target_df, coordinate=False, ohlc=True, psar=False, cum_return=False)
        drawer.save_img(output_path)


def draw_val_image(csv_path, output_dir, window, labeling_ma):
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'], thousands=',')
    df = df.sort_index(ascending=True)

    # Get Backtesting Result
    backtesting = PsarBacktesting(window=window, labeling_ma=labeling_ma)
    df = backtesting.ma_signal(df)
    ret_lim = (df['Total_return'].min(), df['Total_return'].max())

    total = len(df) - window - (labeling_ma - 1)

    for i in range(0, total):
        print("Printing {}th image".format(i + 1))
        target_df = df.loc[df.index[i]: df.index[i + window-1], :]

        # Calculate Gradient
        scaler = target_df['Price'].max() - target_df['Price'].min()
        ma_gradient = tools.calculate_gradient(df, i, scaler, window, labeling_ma)

        # Tagging the image
        tag = str(ma_gradient)
        output_name = '_'.join([str(i), str(window), str(labeling_ma), tag]) + '.png'
        output_path = os.path.join(output_dir, output_name)

        drawer = Drawer()
        drawer.draw_chart(target_df, coordinate=True, ohlc=True, psar=True, cum_return=True, ret_lim=ret_lim)
        drawer.save_img(output_path)


def backtest(csv_path, window, labeling_ma, method='only_psar'):
    backtester = PsarBacktesting(window=window, labeling_ma=labeling_ma, initial_af=0.02, max_af=0.2)
    method_dict = {
        'only_psar': backtester.only_psar,
        'ma_signal': backtester.ma_signal,
        'ma_label': backtester.ma_label
    }
    backtesting = method_dict[method]

    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'], thousands=',')
    df = df.sort_index(ascending=True)

    # Get Backtesting Result
    bt_result = backtesting(df)
    daily_ret = bt_result['Return']
    rf = 1.022
    sharpe_ratio = (daily_ret.mean()**252 - rf) / (daily_ret.std() * np.sqrt(252))
    final_cum_ret = bt_result['Total_return'][-1]

    print('Sharpe Ration: {:.2f}'.format(sharpe_ratio))
    print('Final Return: {:.2f}'.format(final_cum_ret), end='\n\n')
    bt_result['Total_return'].plot()


if __name__ == '__main__':
    root_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading'
    data_dir = 'data'
    # output_dir = r'image\QCOM_Train_Ma20_cur'
    # output_dir = r'image\TSLA_Train_MA20_cur'
    # output_dir = r'image\GOOGL_Train_MA20_cur'
    output_dir = r'image\AAPL_Train_MA20_cur'
    # output_dir = r'image\KOSPI'
    # csv_path = os.path.join(root_path, data_dir, 'QCOM Historical Data.csv')
    csv_path = os.path.join(root_path, data_dir, 'TSLA Historical Data.csv')
    # csv_path = os.path.join(root_path, data_dir, 'GOOGL Historical Data.csv')
    # csv_path = os.path.join(root_path, data_dir, 'AAPL Historical Data.csv')
    # csv_path = os.path.join(root_path, data_dir, 'KOSPI Historical Data.csv')
    output_dir_path = os.path.join(root_path, output_dir)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Draw Image
    # draw_train_image(csv_path=csv_path, output_dir=output_dir_path, window=60, labeling_ma=20)
    # draw_val_image(csv_path=csv_path, output_dir=output_dir_path, window=60, labeling_ma=10)

    # Backtest
    method = ['only_psar', 'ma_signal', 'ma_label'][1]
    backtest(csv_path=csv_path, window=60, labeling_ma=20, method='only_psar')
    backtest(csv_path=csv_path, window=60, labeling_ma=20, method=method)
    plt.show()
