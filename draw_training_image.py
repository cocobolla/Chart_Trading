import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from mpl_finance import candlestick2_ohlc
import matplotlib.ticker as ticker
import datetime


class Drawer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

    def draw_chart(self, df, coordinate=True, ohlc=True, psar=False, cum_return=False):
        # Use Coordinate or not
        if coordinate:
            self.set_axis(df.index.values)
        else:
            self.ax.axis('off')

        # Draw OHLC Candle Chart
        if ohlc:
            self.draw_ohlc(df)

        # Draw PSAR Points
        if psar:
            self.draw_psar(df)

        # Draw Total Return
        if cum_return:
            pass

    def set_axis(self, df_index_values):
        self.ax.axis('on')
        # Set x-axis as Date
        _xticks = list()
        _xlabels = list()
        _wd_prev = 0
        _close = list()
        for _x, d in zip(np.arange(len(df_index_values)), df_index_values):
            weekday = datetime.datetime.strptime(str(d).split('T')[0], '%Y-%m-%d').weekday()
            if weekday <= _wd_prev:
                _xticks.append(_x)
                _xlabels.append(datetime.datetime.strptime(str(d).split('T')[0], '%Y-%m-%d').strftime('%y/%m/%d'))
            _wd_prev = weekday
        self.ax.set_xticks(_xticks)
        self.ax.set_xticklabels(_xlabels, rotation=45, minor=False)
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price($)')

    def draw_ohlc(self, df):
        candle_width = 0.5
        uc = 'r'
        dc = 'b'

        o = df['Open']
        h = df['High']
        l = df['Low']
        c = df['Price']
        candlestick2_ohlc(self.ax, o, h, l, c,
                          width=candle_width, colorup=uc, colordown=dc)

    def draw_psar(self, df):
        if 'Psar' not in df.columns:
            raise KeyError
        psar = df['Psar']
        self.ax.plot(np.arange(len(df)), psar, linestyle='', marker='*')

    def draw_returns(self, df, ret_lim=None):
        if ret_lim is None:
            ret_lim = (df['Total_return'].min(), df['Total_return'].max())
        o_ret = df['Total_return'] * df['Is_long']
        c_ret = df['Total_return'] * (~df['Is_long'])

        self.ax2 = self.ax.twinx()
        self.ax2.set_ylim(ret_lim[0] * 99 / 100, ret_lim[1])
        self.ax2.bar(np.arange(len(df.index)), o_ret, color='r', alpha=0.2)
        self.ax2.bar(np.arange(len(df.index)), c_ret, color='b', alpha=0.2)
        self.ax2.set_ylabel('Cumulative Return')

    def save_img(self, output_path):
        self.fig.savefig(output_path)
        plt.close(self.fig)


def draw_chart(csv_path, output_dir, labeling_ma=5, window=60):
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'])
    df = df.sort_index(ascending=True)
    scaler = (df['Price'].max() - df['Price'].min()) / len(df)
    print("Max: {:.2f}, Min:{:.2f}".format(df['Price'].max(), df['Price'].min()))
    print("Date: {}".format(len(df)))
    print("Scaler: {:.2f}".format(scaler))
    backtesting_ret = psar_backtesting2(df, use_ma=True)
    total_ret = backtesting_ret['Total_return']

    # Sharpe
    r = backtesting_ret['Return']
    sharpe = (r.mean() ** 252 - 1.02) / (r.std() * np.sqrt(252))
    print("Sharpe Ratio: {:.2f}".format(sharpe))
    print("Total Return: {:.2f}".format(total_ret[-1]))

    open_ret = backtesting_ret['Total_return'] * backtesting_ret['Is_long']
    close_ret = backtesting_ret['Total_return'] * (~backtesting_ret['Is_long'])

    for i in range(0, len(df) - window - labeling_ma):
        # Draw Candle Stick
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        target_df = df.loc[df.index[i]: df.index[i + window], :]
        # ax.axis('off')
        candlestick2_ohlc(ax, target_df['Open'], target_df['High'], target_df['Low'], target_df['Price'],
                          width=0.5, colorup='r', colordown='b')

        # """
        # Draw PSAR
        psar_dict = get_psar(target_df)
        psar = psar_dict['Psar']
        # psar = target_df.Price
        ax.plot(np.arange(len(target_df.index)), psar, linestyle='', marker='*')

        # Draw Return
        # ret = total_ret.loc[total_ret.index[i]: total_ret.index[i + window]]
        o_ret = open_ret.loc[df.index[i]: df.index[i+window]]
        c_ret = close_ret.loc[df.index[i]: df.index[i+window]]
        ax2 = ax.twinx()
        ax2.set_ylim(total_ret.min()*99/100, total_ret.max())
        ax2.bar(np.arange(len(target_df.index)), o_ret, color='r', alpha=0.3)
        ax2.bar(np.arange(len(target_df.index)), c_ret, color='b', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price($)')
        ax2.set_ylabel('Cumulative Return')

        # Set x-axis as Date
        _xticks = list()
        _xlabels = list()
        _wd_prev = 0
        _close = list()
        for _x, d in zip(np.arange(len(target_df.index)), target_df.index.values):
            weekday = datetime.datetime.strptime(str(d).split('T')[0], '%Y-%m-%d').weekday()
            if weekday <= _wd_prev:
                _xticks.append(_x)
                _xlabels.append(datetime.datetime.strptime(str(d).split('T')[0], '%Y-%m-%d').strftime('%y/%m/%d'))
            _wd_prev = weekday
        ax.set_xticks(_xticks)
        ax.set_xticklabels(_xlabels, rotation=45, minor=False)
        # """

        # Tagging the name of Candle Chart with MA Gradient
        # close_now = df.loc[df.index[i + window], 'Price']
        # close_future = df.loc[df.index[i + window + labeling_ma], 'Price']
        # ma_gradient = round((close_future - close_now) / labeling_ma, 2)
        future_ma = df.loc[df.index[i + window]:df.index[i + window + labeling_ma], 'Price'].mean()
        ma = df.loc[df.index[i + window - labeling_ma]: df.index[i+window], 'Price'].mean()

        # Calculate Gradient
        """
        ma_gradient = (future_ma - ma)/labeling_ma
        ma_gradient / scaler
        """
        ma_gradient = (future_ma - ma) / (target_df['Price'].max() - target_df['Price'].min())
        ma_gradient = round(ma_gradient, 2)
        tag = str(ma_gradient)
        # output_name = str(i) + '_' + tag + '.png'
        output_name = '_'.join([str(i), str(window), str(labeling_ma), tag]) + '.png'
        output_path = os.path.join(output_dir, output_name)

        print("Printing {}th image".format(i + 1))
        # plt.show()
        fig.savefig(output_path)
        plt.close(fig)


def is_psar_bull(h, l, psar):
    if psar >= (h + l)/2:
        return False
    elif psar <= (h + l)/2:
        return True
    else:
        print(h, l, psar)
        raise


def psar_backtesting(barsdata, use_ma=False):
    initial_af = 0.02
    max_af = 0.2
    psar_dict = get_psar(barsdata)
    barsdata['Psar'] = psar_dict['Psar']
    # barsdata['Psar'] = get_psar(barsdata, initial_af=initial_af, max_af=max_af)['Psar']
    barsdata['Is_bull'] = barsdata.apply(lambda x: is_psar_bull(x['High'], x['Low'], x['Psar']), axis=1)
    barsdata['Return'] = 1.0

    barsdata.loc[barsdata.index[0:60], 'Is_bull'] = False
    bull_index = barsdata['Is_bull']
    barsdata.loc[bull_index, 'Return'] = \
        barsdata.shift(periods=-1).loc[bull_index, 'Price'] / barsdata.loc[bull_index, 'Price']
    barsdata['Total_return'] = barsdata['Return'].cumprod()

    # Future MA
    labeling_ma = 20
    window = 60
    scaler = (barsdata['Price'].max() - barsdata['Price'].min()) / len(barsdata)

    for i in range(0, len(barsdata) - labeling_ma - window):
        prev_row_index = barsdata.index[i-1]
        row_index = barsdata.index[i]
        prev_status = barsdata.loc[prev_row_index, 'Is_bull']
        status = barsdata.loc[row_index, 'Is_bull']
        bull = True
        bear = False

        future_ma = barsdata.loc[barsdata.index[i + window]:barsdata.index[i + window + labeling_ma], 'Price'].mean()
        ma = barsdata.loc[barsdata.index[i + window - labeling_ma]: barsdata.index[i+window], 'Price'].mean()
        ma_gradient = (future_ma - ma)/labeling_ma
        ma_gradient / scaler
        ma_gradient = round(ma_gradient, 2)

        if use_ma:
            bull_index = barsdata['Is_bull']
            barsdata.loc[bull_index, 'Return'] = \
                barsdata.shift(periods=-1).loc[bull_index, 'Price'] / barsdata.loc[bull_index, 'Price']

        else:
            ma_condition = True
        # ma_condition = True

        if prev_status == bull:
            barsdata.loc[row_index, 'Return'] = barsdata.loc[prev_row_index, 'Return'] * \
                                                barsdata.loc[row_index, 'Price'] / barsdata.loc[prev_row_index, 'Price']

        if prev_status == bear:
            barsdata.loc[row_index, 'Return'] = barsdata.loc[prev_row_index, 'Return']

    return barsdata['Return']


def psar_backtesting2(barsdata, use_ma=False):
    initial_af = 0.02
    max_af = 0.2
    psar_dict = get_psar(barsdata)
    barsdata['Psar'] = psar_dict['Psar']
    # barsdata['Psar'] = get_psar(barsdata, initial_af=initial_af, max_af=max_af)['Psar']
    barsdata['Is_bull'] = barsdata.apply(lambda x: is_psar_bull(x['High'], x['Low'], x['Psar']), axis=1)

    barsdata.loc[barsdata.index[0:60], 'Is_bull'] = False

    # Future MA
    labeling_ma = 10
    window = 60
    scaler = (barsdata['Price'].max() - barsdata['Price'].min()) / len(barsdata)

    if use_ma:
        barsdata['MA_gradient'] = 0
        t = 0.2

        for i in range(0, len(barsdata) - labeling_ma - window):
            prev_row_index = barsdata.index[i-1]
            row_index = barsdata.index[i]
            prev_status = barsdata.loc[prev_row_index, 'Is_bull']
            status = barsdata.loc[row_index, 'Is_bull']
            bull = True
            bear = False

            future_ma = barsdata.loc[barsdata.index[i + window]:barsdata.index[i + window + labeling_ma], 'Price'].mean()
            ma = barsdata.loc[barsdata.index[i + window - labeling_ma]: barsdata.index[i+window], 'Price'].mean()
            # Calculate Gradient
            ma_gradient = (future_ma - ma)/(barsdata.loc[barsdata.index[i]: barsdata.index[i + window], 'Price'].max()
             - barsdata.loc[barsdata.index[i]: barsdata.index[i+window], 'Price'].min())
            ma_gradient = round(ma_gradient, 2)

            """
            ma_gradient = (future_ma - ma)/labeling_ma
            ma_gradient / scaler
            """
            ma_gradient = round(ma_gradient, 2)
            barsdata.loc[barsdata.index[i+window], 'MA_gradient'] = ma_gradient

        barsdata['MA_con'] = np.NaN
        bull_str_idx = (barsdata['Is_bull'] == True) * (barsdata['Is_bull'].shift(1) == False)
        barsdata.loc[bull_str_idx * (barsdata['MA_gradient'] > t), 'MA_con'] = True
        barsdata.loc[bull_str_idx * (barsdata['MA_gradient'] <= t), 'MA_con'] = False
        barsdata['MA_con'] = barsdata['MA_con'].fillna(method='ffill')
        barsdata['MA_con'] = barsdata['MA_con'].fillna(False)
        barsdata['Is_long'] = barsdata['Is_bull'] * barsdata['MA_con']

        long_pos_idx = barsdata['Is_long']

    else:
        barsdata['Is_long'] = barsdata['Is_bull'].shift(1)
        barsdata['Is_long'] = barsdata['Is_long'].fillna(False)
        long_pos_idx = barsdata['Is_long']

    barsdata['Return'] = 1.0
    barsdata.loc[long_pos_idx, 'Return'] = \
        barsdata.shift(periods=-1).loc[long_pos_idx, 'Price'] / barsdata.loc[long_pos_idx, 'Price']
    barsdata['Total_return'] = barsdata['Return'].cumprod()

    return barsdata


def get_psar(barsdata, initial_af=0.02, max_af=0.2):
    length = len(barsdata)
    dates = list(barsdata.index)
    high = list(barsdata['High'])
    low = list(barsdata['Low'])
    close = list(barsdata['Price'])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = initial_af
    # ep = low[0]
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
                af = initial_af
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = initial_af

        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + initial_af, max_af)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + initial_af, max_af)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]

    return {"Dates": dates, "High": high, "Low": low, "Close": close, "Psar": psar, "Psarbear": psarbear,
            "Psarbull": psarbull}


if __name__ == '__main__':
    root_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading'
    data_dir = 'data'
    output_dir = 'Qualcomm_Chart_Test'
    csv_path = os.path.join(root_path, data_dir, 'QCOM Historical Data.csv')
    output_dir_path = os.path.join(root_path, output_dir)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    from chart_drawer import Drawer
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'])
    df = df.sort_index(ascending=True)

    df = psar_backtesting2(df, use_ma=True)

    window = 60
    labeling_ma = 10
    ret_lim = (df['Total_return'].min(), df['Total_return'].max())

    for i in range(0, len(df) - window - labeling_ma):
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
        # output_name = str(i) + '_' + tag + '.png'
        output_name = '_'.join([str(i), str(window), str(labeling_ma), tag]) + '.png'
        output_path = os.path.join(output_dir, output_name)

        drawer = Drawer()
        drawer.draw_chart(target_df, coordinate=True, ohlc=True, psar=True, cum_return=True, ret_lim=ret_lim)
        drawer.save_img(output_path)
        print("{}th image".format(i))


if __name__ == '__main_':
    root_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading'
    data_dir = 'data'
    # output_dir = 'Future_ma20_Labeling_scaling_psar_ma'
    # output_dir = 'TSLA_Chart_BT'
    output_dir = 'Qualcomm_Chart_BT_10_60_0.2'
    # csv_path = os.path.join(root_path, data_dir, 'TSLA Historical Data.csv')
    csv_path = os.path.join(root_path, data_dir, 'QCOM Historical Data.csv')
    output_dir_path = os.path.join(root_path, output_dir)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    draw_chart(csv_path=csv_path, output_dir=output_dir_path, labeling_ma=20, window=60)
