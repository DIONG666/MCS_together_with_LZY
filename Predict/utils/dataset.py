from sklearn.svm import SVR
import h5py
import pickle
import numpy as np
from pandas import to_datetime
import pandas as pd
from sklearn import cluster
from datetime import date


class MinMaxNorm01(object):
    """scale data to range [0, 1]"""

    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        # print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x


def get_date_feature(idx):
    a = idx.weekday()
    b = idx.hour
    c = idx.weekday() // 6
    d = idx.weekday() // 7
    return a, b, c, d


def traffic_loader(f, opt):
    if opt.nb_flow == 1:
        data = f['data'][7 * 24:, :, :]
        data_p = f['data'][6 * 24:-1 * 24, :, :]
        data_t = f['data'][:-7 * 24, :, :]
        result = data.reshape((-1, 1, opt.height, opt.width))

        if opt.crop:
            result_o = data.reshape((-1, 1, opt.height, opt.width))
            result_p = data_p.reshape((-1, 1, opt.height, opt.width))
            result_t = data_t.reshape((-1, 1, opt.height, opt.width))
            result = np.concatenate((result_o, result_p, result_t), axis=1)
            # print(result.shape)
        return result
    else:
        print("Wrong parameter with nb_flow")
        exit(0)


def get_label(data, feature, index, clusters):
    samples, channels, h, w = data.shape
    sum_data = np.sum(data, axis=1).reshape((samples, h * w))
    df_data = pd.DataFrame(sum_data, index=index)

    df_data = df_data.resample('1D').sum().transpose()

    feature = pd.DataFrame(np.reshape(feature, (h * w, -1)))
    df = pd.concat([df_data, feature], axis=1)
    df.fillna(0, inplace=True)
    # clustering the data points and get the cluster id
    clf = cluster.AgglomerativeClustering(n_clusters=clusters)
    clf.fit(df)
    return clf.labels_


def get_label_v2(data, feature, index, clusters):
    # print(np.sum(feature, axis=-1))
    # print(data.sum().sum())
    all_data = data.sum().sum() + np.sum(feature, axis=-1)  # (20,20)
    mask = all_data.astype(bool)
    # print(mask)
    from sklearn.feature_extraction import image
    graph = image.img_to_graph(all_data, mask=mask)
    # print(graph)
    graph.data = np.exp(-graph.data / graph.data.std())
    # print(graph)
    labels = cluster.spectral_clustering(graph, n_clusters=clusters, eigen_solver='arpack')
    # print(labels)
    return labels


def read_data(path, opt):
    f = h5py.File(path, 'r')
    data = traffic_loader(f, opt)

    mmn = MinMaxNorm01()
    data_scaled = mmn.fit_transform(data)
    X, y = [], []
    h, w = data.shape[2], data.shape[3]

    for i in range(opt.close_size, len(data)):
        xc_ = [[data_scaled[i - c][0, :, :]] for c in range(1, opt.close_size + 1)]
        xc_.append([data_scaled[i][1, :, :]])
        xc_.append([data_scaled[i][2, :, :]])

        if opt.close_size > 0:
            X.append(xc_)

        y.append(data_scaled[i][0, :, :])

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, mmn


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('-close_size', type=int, default=3)
    parse.add_argument('-period_size', type=int, default=0)
    parse.add_argument('-trend_size', type=int, default=0)
    parse.add_argument('-crop', type=int, default=1)
    parse.add_argument('-nb_flow', type=int, default=1)
    parse.add_argument('-height', type=int, default=4)
    parse.add_argument('-width', type=int, default=4)
    parse.add_argument('-rows', nargs='+', type=int, default=[0, 4])
    parse.add_argument('-cols', nargs='+', type=int, default=[0, 4])
    parse.add_argument('-traffic', type=str, default='sms')
    parse.add_argument('-cluster', type=int, default=3)
    parse.add_argument('-test_size', type=int, default=120)
    opt = parse.parse_args()

    # path = 'data.h5'
    path = '../data/BJ16_In.h5'
    # feature_path = '../data/crawled_feature.csv'
    X, y, mmn = read_data(path, opt)
    x_train, x_test = X[:-opt.test_size], X[-opt.test_size:]
    y_tr, y_te = y[:-opt.test_size], y[-opt.test_size:]
    # y_date_tr, y_date_te = y_date[:-opt.test_size], y_date[-opt.test_size:]
    print('y_data', y_te.shape)
    # print('date',y_date_te.shape)
    # print('first date:',y_date_te[0],'last date:', y_date_te[-1])
    # for i in range(len(y_date_te)):
    #     if y_date_te[i].day == 26:
    #         print(i)
    print(X.shape)
    print(y.shape)
