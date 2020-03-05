import os
from keras import datasets
import keras
import numpy as np
import cv2
assert keras.backend.image_data_format() == 'channels_last'
from keras.models import load_model

from keraspp import aicnn
np.random.seed(0)


def gradient2label(gradient):
    label_dict = {
        (np.pi*5/10, np.pi*3/10): 4,
        (np.pi*3/10, np.pi/10): 3,
        (np.pi/10, -np.pi/10): 2,
        (-np.pi/10, -np.pi*3/10): 1,
        (-np.pi*3/10, -np.pi*5/10): 0,
    }
    label = -1
    for k, v in label_dict.items():
       if np.tan(k[0]) > gradient >= np.tan(k[1]):
           label = v
    assert(label != -1)
    return label


def load_data():
    dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\Future_ma20_Labeling_scaling'
    file_list = [x for x in os.listdir(dir_path) if x.endswith('.png')]
    gradient_list = [float(os.path.splitext(x)[0].split('_')[1]) for x in file_list]
    label_list = [gradient2label(x) for x in gradient_list]
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
        super().__init__(X, y, nb_classes=5)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # epochs = 1000
    epochs = 250
    batch_size = 1024
    # batch_size = 128
    verbose = 1

    m = Machine()
    m.run(epochs=epochs, batch_size=batch_size, verbose=verbose)


def test():
    # Load test data
    dir_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading\Future_ma20_Labeling_scaling\test2'
    file_list = [x for x in os.listdir(dir_path) if x.endswith('.png')]
    # file_list = sorted(file_list)
    gradient_list = [float(os.path.splitext(x)[0].split('_')[1]) for x in file_list]
    label_list = [gradient2label(x) for x in gradient_list]
    path_list = [os.path.join(dir_path, x) for x in file_list]
    img_list = [cv2.resize(cv2.imread(x), dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for x in path_list]
    X = np.array(img_list)
    y = np.array(label_list)

    data = aicnn.DataSetModified(X, y, 5)

    model_dir = r'C:\Users\USER\workspace\KSIF\Chart_Trading\output_fd4c1a13-fb89-4822-b31a-36f98dfce91b'
    model_name = r'dl_model.h5'
    model_path = os.path.join(model_dir, model_name)
    print(os.path.exists(model_path))
    model = aicnn.CNN(nb_classes=5, in_shape=data.input_shape)
    model.load_weights(model_path)
    print(model.summary())
    pred = model.predict(data.X)
    print(pred)
    pass


if __name__ == '__main__':
    test()
    # main()