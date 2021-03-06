import numpy as np
import pandas as pd
import tools


class PsarBacktesting:
    def __init__(self, window=60, labeling_ma=20, initial_af=0.02, max_af=0.2):
        self.window = window
        self.labeling_ma = labeling_ma
        self.initial_af = initial_af
        self.max_af = max_af

    def only_psar(self, barsdata):
        psar_dict = PsarBacktesting.get_psar(barsdata, initial_af=self.initial_af, max_af=self.max_af)
        barsdata['Psar'] = psar_dict['Psar']
        barsdata['Is_bull'] = barsdata.apply(
            lambda x: PsarBacktesting.is_psar_bull(x['High'], x['Low'], x['Psar']), axis=1)

        barsdata.loc[barsdata.index[0:self.window], 'Is_bull'] = False

        barsdata['Is_long'] = barsdata['Is_bull'].shift(1)
        barsdata['Is_long'] = barsdata['Is_long'].fillna(False)
        long_pos_idx = barsdata['Is_long']

        barsdata['Return'] = 1.0
        barsdata.loc[long_pos_idx, 'Return'] = \
            barsdata.shift(periods=-1).loc[long_pos_idx, 'Price'] / barsdata.loc[long_pos_idx, 'Price']
        assert(barsdata['Return'].isna().sum() <= 1)
        barsdata['Return'] = barsdata['Return'].fillna(1)

        barsdata['Total_return'] = barsdata['Return'].cumprod()
        return barsdata

    def ma_signal(self, barsdata):
        psar_dict = PsarBacktesting.get_psar(barsdata, initial_af=self.initial_af, max_af=self.max_af)
        barsdata['Psar'] = psar_dict['Psar']
        barsdata['Is_bull'] = barsdata.apply(
            lambda x: PsarBacktesting.is_psar_bull(x['High'], x['Low'], x['Psar']), axis=1)

        barsdata.loc[barsdata.index[0:self.window-2], 'Is_bull'] = False

        barsdata['MA_gradient'] = 0
        t = 0.2

        for i in range(0, len(barsdata) - self.labeling_ma - self.window):
            future_ma = barsdata.loc[
                        barsdata.index[i + self.window]:barsdata.index[i + self.window + self.labeling_ma], 'Price'
                        ].mean()
            ma = barsdata.loc[
                 barsdata.index[i + self.window - 1 - self.labeling_ma]: barsdata.index[i + self.window], 'Price'
                 ].mean()

            # Calculate Gradient
            """
            ma_gradient = (future_ma - ma) / (
                        barsdata.loc[barsdata.index[i]: barsdata.index[i + self.window - 1], 'Price'].max()
                        - barsdata.loc[barsdata.index[i]: barsdata.index[i + self.window - 1], 'Price'].min()
                        )
            ma_gradient = round(ma_gradient, 2)
            """

            scaler = (barsdata.loc[barsdata.index[i]: barsdata.index[i + self.window - 1], 'Price'].max()
            - barsdata.loc[barsdata.index[i]: barsdata.index[i + self.window - 1], 'Price'].min())
            ma_gradient = tools.calculate_gradient(barsdata, i, scaler, self.window, self.labeling_ma)
            barsdata.loc[barsdata.index[i + self.window - 1], 'MA_gradient'] = ma_gradient

        barsdata['MA_con'] = np.NaN
        bull_str_idx = (barsdata['Is_bull'] == True) & (barsdata['Is_bull'].shift(1) == False)
        barsdata.loc[bull_str_idx & (barsdata['MA_gradient'] > t), 'MA_con'] = True
        barsdata.loc[bull_str_idx & (barsdata['MA_gradient'] <= t), 'MA_con'] = False
        barsdata['MA_con'] = barsdata['MA_con'].fillna(method='ffill')
        barsdata['MA_con'] = barsdata['MA_con'].fillna(False)
        barsdata['Is_long'] = barsdata['Is_bull'] & barsdata['MA_con']
        long_pos_idx = barsdata['Is_long']

        barsdata['Return'] = 1.0
        barsdata.loc[long_pos_idx, 'Return'] = \
            barsdata.shift(periods=-1).loc[long_pos_idx, 'Price'] / barsdata.loc[long_pos_idx, 'Price']

        assert(barsdata['Return'].isna().sum() <= 1)
        barsdata['Return'] = barsdata['Return'].fillna(1)
        barsdata['Total_return'] = barsdata['Return'].cumprod()
        return barsdata

    def ma_label(self, barsdata):
        log_df = pd.read_csv('apple.csv', index_col='Date', parse_dates=['Date'])
        barsdata['Label'] = log_df['Pred']
        psar_dict = PsarBacktesting.get_psar(barsdata, initial_af=self.initial_af, max_af=self.max_af)
        barsdata['Psar'] = psar_dict['Psar']
        barsdata['Is_bull'] = barsdata.apply(
            lambda x: PsarBacktesting.is_psar_bull(x['High'], x['Low'], x['Psar']), axis=1)

        barsdata.loc[barsdata.index[0:self.window-2], 'Is_bull'] = False

        barsdata['MA_gradient'] = 0
        t = 2
        total = len(barsdata) - self.window - (self.labeling_ma - 1)

        for i in range(0, total):
            target_barsdata = barsdata.loc[barsdata.index[i]: barsdata.index[i + self.window-1], :]
            scaler = target_barsdata['Price'].max() - target_barsdata['Price'].min()
            ma_gradient = tools.calculate_gradient(barsdata, i, scaler, self.window, self.labeling_ma)
            barsdata.loc[barsdata.index[i + self.window - 1], 'MA_gradient'] = ma_gradient

        barsdata['MA_con'] = np.NaN
        bull_str_idx = (barsdata['Is_bull'] == True) & (barsdata['Is_bull'].shift(1) == False)
        barsdata.loc[bull_str_idx & (barsdata['Label'] >= t), 'MA_con'] = True
        barsdata.loc[bull_str_idx & (barsdata['Label'] < t), 'MA_con'] = False
        barsdata['MA_con'] = barsdata['MA_con'].fillna(method='ffill')
        barsdata['MA_con'] = barsdata['MA_con'].fillna(False)
        barsdata['Is_long'] = barsdata['Is_bull'] & barsdata['MA_con']
        long_pos_idx = barsdata['Is_long']

        barsdata['Return'] = 1.0
        barsdata.loc[long_pos_idx, 'Return'] = \
            barsdata.shift(periods=-1).loc[long_pos_idx, 'Price'] / barsdata.loc[long_pos_idx, 'Price']

        assert(barsdata['Return'].isna().sum() <= 1)
        barsdata['Return'] = barsdata['Return'].fillna(1)
        barsdata['Total_return'] = barsdata['Return'].cumprod()
        return barsdata

    @staticmethod
    def is_psar_bull(h, l, psar):
        if psar >= (h + l) / 2:
            return False
        elif psar <= (h + l) / 2:
            return True
        else:
            print(h, l, psar)
            raise

    @staticmethod
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
