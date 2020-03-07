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

    def draw_chart(self, df, coordinate=True, ohlc=True, psar=False, cum_return=False, ret_lim=None):
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
            self.draw_returns(df, ret_lim)

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

    @staticmethod
    def show_img():
        plt.show()

    def save_img(self, output_path):
        self.fig.savefig(output_path)
        plt.close(self.fig)
