import os
from keras import datasets
import keras
import numpy as np
import cv2
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools

from keraspp import aicnn

assert keras.backend.image_data_format() == 'channels_last'
np.random.seed(0)


def gradient2label(gradient):
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
    """
    label_dict = {
        (np.pi/2, np.pi/12): 4,
        (np.pi*1/12, 0): 2,
        (0, -np.pi/12): 1,
        (-np.pi/12, -np.pi/2): 0,
    }
    label = -1
    for k, v in label_dict.items():
       if np.tan(k[0]) > gradient >= np.tan(k[1]):
           label = v
    assert(label != -1)
    return label


def load_data():
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\Future_ma20_Labeling_scaling'
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\QCOM_Train_MA20'
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\total'
    dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\tsla_train'
    file_list = [x for x in os.listdir(dir_path) if x.endswith('.png')]
    gradient_list = [float(os.path.splitext(x)[0].split('_')[-1]) for x in file_list]
    label_list = [tools.gradient2label(x, tools.LABEL_DICT_MA20) for x in gradient_list]
    path_list = [os.path.join(dir_path, x) for x in file_list]

    # Resize
    # img_list = [cv2.resize(x, dsize=(32, 32), interpolation=cv2.INTER_LINEAR) for x in path_list]
    # img_list = [cv2.imread(x) for x in path_list]
    img_list = [cv2.resize(cv2.imread(x), dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for x in path_list]

    """
    for i, a in enumerate(img_list):
        cv2.imshow('a', a)
        print(label_list[i])
        cv2.waitKey(0)
    """

    X = np.array(img_list)
    y = np.array(label_list)

    return X, y


class Machine(aicnn.Machine):
    def __init__(self):
        # (X, y), (x_test, y_test) = datasets.cifar10.load_data()
        (X, y) = load_data()

        super().__init__(X, y, nb_classes=4)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # epochs = 1000
    epochs = 200
    batch_size = 1024
    # batch_size = 128
    verbose = 1
    model_name = '4class'

    m = Machine()
    m.run(epochs=epochs, batch_size=batch_size, verbose=verbose, model_name=model_name)


def test():
    # Load test data
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\Future_ma20_Labeling_scaling\test2'
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\Future_ma20_Labeling_scaling'
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\QCOM_Train_Ma20'
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\TSLA_Train_Ma20'
    dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\tsla_test'
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\KOSPI'
    # dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\image\AAPL_Train_Ma20'
    file_list = [x for x in os.listdir(dir_path) if x.endswith('.png')]
    index_list = [int(x.split('_')[0]) for x in file_list]
    date_list = [x.split('_')[1] for x in file_list]
    gradient_list = [float(os.path.splitext(x)[0].split('_')[-1]) for x in file_list]
    label_list = [tools.gradient2label(x, tools.LABEL_DICT_MA20) for x in gradient_list]
    pd.Series(gradient_list).hist()
    plt.show()
    pd.Series(label_list).hist()
    plt.show()
    # pd.Series([np.arctan(x) / np.pi * 180 for x in gradient_list]).hist()
    path_list = [os.path.join(dir_path, x) for x in file_list]
    img_list = [cv2.resize(cv2.imread(x), dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for x in path_list]
    X = np.array(img_list)
    y = np.array(label_list)
    data = aicnn.DataSetModified(X, y, 5)

    # model_dir = r'C:\Users\USER\workspace\KSIF\Chart_Trading\output_fd4c1a13-fb89-4822-b31a-36f98dfce91b'
    # model_dir = r'C:\Users\USER\workspace\KSIF\Chart_Trading\output_3d97d721-19dc-4b0e-99f6-b0f9ad172162'
    model_dir = os.path.join('model', '4class_20y03m22d1623')
    model_name = r'dl_model.h5'
    model_path = os.path.join(model_dir, model_name)
    print(os.path.exists(model_path))
    model = aicnn.CNN(nb_classes=4, in_shape=data.input_shape)
    model.load_weights(model_path)
    # print(model.summary())
    pred = model.predict(data.X)
    pred_list = np.argmax(pred, axis=1)

    log_df = pd.DataFrame({
        'Index': index_list,
        'Date': date_list,
        'Gradient': gradient_list,
        'Label': label_list,
        'Pred': pred_list
    })
    class_num = pred.shape[1]
    for i in range(class_num):
        log_df[('Prob_' + str(i))] = np.round(pred[:, i], 2)
    log_df = log_df.set_index('Index')
    log_df = log_df.sort_index()
    log_df.to_csv('apple.csv')

    # print(pred)
    print("Accuracy: {:.2f}".format(((np.argmax(pred, axis=1) - y) == 0).sum() / len(y)))


if __name__ == '__main__':
    # test()
    main()