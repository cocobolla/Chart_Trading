import numpy as np


def calculate_gradient(df, index, scaler, window, labeling_ma):
    future_ma = df.loc[df.index[index + window]:df.index[index + window + (labeling_ma - 1)], 'Price'].mean()
    ma = df.loc[df.index[index + window - 1 - (labeling_ma - 1)]: df.index[index + window - 1], 'Price'].mean()
    cur_price = df.loc[df.index[index + window - 1], 'Price']

    """
    ma_gradient = (future_ma - ma)/labeling_ma
    ma_gradient / scaler
    """
    # scaler = 1
    # ma_gradient = (future_ma - ma) / scaler
    ma_gradient = (future_ma - cur_price) / scaler
    ma_gradient = round(ma_gradient, 2)

    return ma_gradient


LABEL_DICT_MA10 = {
    (np.pi/18, np.pi/2): 3,
    (0, np.pi * 1/18): 2,
    (-np.pi/18, 0): 1,
    (-np.pi/2, -np.pi/18): 0,
}


LABEL_DICT_MA20 = {
    (np.pi/15, np.pi/2): 3,
    (0, np.pi/15): 2,
    (-np.pi/15, 0): 1,
    (-np.pi/2, -np.pi/15): 0,
}

LABEL_DICT_MA30 = {
    (np.pi/12, np.pi/2): 3,
    (0, np.pi * 1/12): 2,
    (-np.pi/12, 0): 1,
    (-np.pi/2, -np.pi/12): 0,
}


def gradient2label(gradient, label_dict):
    """
    label_dict = {
        (np.pi*5/10, np.pi*3/10): 4,
        (np.pi*3/10, np.pi/10): 3,
        (np.pi/10, -np.pi/10): 2,
        (-np.pi/10, -np.pi*3/10): 1,
        (-np.pi*3/10, -np.pi*5/10): 0,
    }
    """
    """
    label_dict = {
        (np.pi/2, np.pi*3/12): 4,
        (np.pi*3/12, np.pi/12): 3,
        (np.pi/12, -np.pi/12): 2,
        (-np.pi/12, -np.pi*3/12): 1,
        (-np.pi*3/12, -np.pi/2): 0,
    }
    label_dict = {
        (np.pi/2, np.pi/12): 4,
        (np.pi*1/12, 0): 2,
        (0, -np.pi/12): 1,
        (-np.pi/12, -np.pi/2): 0,
    }
    """
    label = -1
    for k, v in label_dict.items():
       if np.tan(k[0]) <= gradient < np.tan(k[1]):
           label = v
    assert(label != -1)
    return label

