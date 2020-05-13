import pandas as pd
import matplotlib.pyplot as plt
import os
import tools
import numpy as np


def get_gradient_csv(csv_path,  window, labeling_ma):
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=['Date'], thousands=',')
    df = df.sort_index(ascending=True)
    gradient_list = []

    total = len(df) - window - (labeling_ma - 1)
    print("Total number: {}".format(total))

    for i in range(0, total):
        target_df = df.loc[df.index[i]: df.index[i + window - 1], :]

        # Calculate Gradient
        scaler = target_df['Price'].max() - target_df['Price'].min()
        ma_gradient = tools.calculate_gradient(df, i, scaler, window, labeling_ma)
        gradient_list.append(ma_gradient)

    return gradient_list


def get_gradient_image():
    pass


def show_gradient_stat(target_list, report_name):
    target = np.array(target_list)
    target_num = len(target)
    target_mean = target.mean()
    target_std = target.std()
    plt.hist(target_list, alpha=0.5, bins=50)
    print("----------------------------------")
    print("{}".format(report_name))
    print("Number: {}".format(target_num))
    print("Mean: {:.2f}".format(target_mean))
    print("Std: {:.2f}".format(target_std))
    print()


class ExploreTarget:
    def __init__(self, name, gradients, labels):
        self.name = name
        self.gradients = np.array(gradients)
        self.labels = np.array(labels)

    def print_gradient_stat(self):
        print("----------------------------------")
        print("{}'s Gradient Report".format(self.name))
        print("Number: {}".format(len(self.gradients)))
        print("Mean: {:.4f}".format(self.gradients.mean()))
        print("Std: {:.4f}".format(self.gradients.std()))

    def print_label_stat(self):
        print("----------------------------------")
        print("{}'s Label Report".format(self.name))
        print("Class: {}".format(set(self.labels)))
        for i in set(self.labels):
            print("{}: {}, {:.2f}".format(
                i, sum(self.labels == i), sum(self.labels == i) / len(self.labels)))

    def draw_gradient_hist(self, label_dict=None):
        plt.hist(self.gradients, alpha=0.5, bins=70, label=self.name)
        plt.legend()
        if label_dict is not None:
            key_list = np.array(list(label_dict.keys())[:-1])
            x_list = key_list[:, 0]
            x_list = x_list.round(2)
            plt.xticks(x_list)
            for x in x_list:
                plt.axvline(x=x, color='red')

    def draw_label_hist(self):
        plt.hist(self.labels, alpha=0.5, bins=70, label=self.name)
        plt.legend()


def test():
    root_path = r'C:\Users\USER\workspace\KSIF\Chart_Trading'
    data_dir = 'data'
    # output_dir = r'image\QCOM_Train'
    output_dir = r'image\KOSPI'
    csv_path_q = os.path.join(root_path, data_dir, 'QCOM Historical Data.csv')
    csv_path_t = os.path.join(root_path, data_dir, 'TSLA Historical Data.csv')
    csv_path_g = os.path.join(root_path, data_dir, 'GOOGL Historical Data.csv')
    csv_path_a = os.path.join(root_path, data_dir, 'AAPL Historical Data.csv')

    gradient_list_q = get_gradient_csv(csv_path_q, window=60, labeling_ma=20)
    gradient_list_t = get_gradient_csv(csv_path_t, window=60, labeling_ma=20)
    gradient_list_g = get_gradient_csv(csv_path_g, window=60, labeling_ma=20)
    gradient_list_a = get_gradient_csv(csv_path_a, window=60, labeling_ma=20)

    label_dict = tools.LABEL_DICT_MA20
    label_list_q = [tools.gradient2label(x, label_dict) for x in gradient_list_q]
    label_list_t = [tools.gradient2label(x, label_dict) for x in gradient_list_t]
    label_list_g = [tools.gradient2label(x, label_dict) for x in gradient_list_g]
    label_list_a = [tools.gradient2label(x, label_dict) for x in gradient_list_a]

    qualcomm = ExploreTarget('Qualcomm', gradient_list_q, label_list_q)
    tesla = ExploreTarget('Tesla', gradient_list_t, label_list_t)
    google = ExploreTarget('Google', gradient_list_g, label_list_g)
    apple = ExploreTarget('Apple', gradient_list_g, label_list_a)

    # Gradient
    label_dict = None
    qualcomm.print_gradient_stat()
    tesla.print_gradient_stat()
    # google.print_gradient_stat()
    # apple.print_gradient_stat()
    qualcomm.draw_gradient_hist(label_dict)
    tesla.draw_gradient_hist(label_dict)
    # google.draw_gradient_hist(label_dict)
    # apple.draw_gradient_hist(label_dict)
    plt.show()
    plt.close()

    # Label
    qualcomm.print_label_stat()
    tesla.print_label_stat()
    google.print_label_stat()
    apple.print_label_stat()
    qualcomm.draw_label_hist()
    tesla.draw_label_hist()
    google.draw_label_hist()
    apple.draw_label_hist()
    plt.show()


if __name__ == "__main__":
    test()
