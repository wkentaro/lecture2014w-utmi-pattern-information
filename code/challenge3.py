#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function, division
import collections

import numpy as np
from skimage.io import imread
from sklearn import svm
import theano.tensor as T

from autoencoder import AutoEncoder
from challenge1 import FisherLDA


def get_jaffe_data():
    label_data = np.loadtxt('../data/jaffe/label.txt', dtype=str)

    sd_rates = label_data[:, 1:-2].astype(float)
    # HAP: 0, SAD: 1, SUR: 2, ANG:3, DIS:4, FEA:5, PIC:6
    y = np.argmax(sd_rates, axis=1)

    # get img data
    X = None
    skip = []
    for i, (label_id, label_nm) in enumerate(
            zip(label_data[:, 0], label_data[:, -1])):
        # get img path
        img_path = '../data/jaffe/'
        img_path += label_nm.replace('-', '.')
        img_path += '.{}.tiff'.format(label_id)
        # load img
        try:
            x = imread(img_path)
            x = x.reshape(-1)
        except IOError:
            skip.append(i)
            continue
        # store img data
        if X is None:
            X = np.zeros((label_data.shape[0], x.shape[0]))
        X[i] = x

    # get clean index
    index = np.array(
        [i for i in range(label_data.shape[0]) if i not in skip])

    # get person labeling index
    person_data = collections.defaultdict(list)
    for i, label_nm in enumerate(label_data[:, -1][index]):
        person = label_nm.split('-')[0]
        person_data[person].append(i)

    return X[index], y[index], person_data


def fisher_lda(x, y):
    flda = FisherLDA(n_components=100)
    flda.fit(x, y)
    return flda.transform(x)


def identity(x):
    return x


def hidden_of_ae(x):
    def sigmoid(x):
        return 1. / (1. + T.exp(-1 * x))

    ae = AutoEncoder(x,
                     hidden_size=100,
                     activation_function=sigmoid,
                     output_function=sigmoid)
    ae.train()
    return ae.get_hidden(x)


def main():
    X, y, person_data = get_jaffe_data()

    # X = identity(X)
    X = fisher_lda(X, y)
    clf = svm.SVC()

    # get scores
    scores = []
    for person, test_index in person_data.items():
        # get data
        X_test, y_test = X[test_index], y[test_index]
        train_index = np.array(
            [i for i in range(X.shape[0]) if i not in test_index])
        X_train, y_train = X[train_index], y[train_index]
        # train and get score
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    score = np.array(scores).mean()
    print("score: {}".format(score))



if __name__ == '__main__':
    main()

