#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Assignment1

Assignment:
    * http://www.mi.t.u-tokyo.ac.jp/harada/lectures/pattern/patternreport20150116.pdf

Reference:
    * http://www.uni-weimar.de/medien/webis/teaching/lecturenotes/machine-learning/unit-en-ml-introduction.pdf
    * http://nbviewer.ipython.org/github/mgrani/LODA-lecture-notes-on-data-analysis/blob/master/II.ML-and-DM/II.ML-and-DM-Example-LMS.ipynb

"""
from __future__ import division, print_function
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

__author__ = "www.kentaro.wada@gmail.com (Kentaro Wada)"


def load_data(filename):
    with open(filename, 'rb') as f:
        data = []
        for row in f.readlines():
            data.append(row.split())
    return np.array(data, dtype=float)


def get_kadai1_dataset():
    # get train dataset
    X1_train = load_data('../data/Train1.txt')
    y1 = np.empty(X1_train.shape[0]).astype(int)
    y1.fill(0)
    X2_train = load_data('../data/Train2.txt')
    y2 = np.empty(X2_train.shape[0]).astype(int)
    y2.fill(1)
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1, y2))
    # get test dataset
    X1_test = load_data('../data/Test1.txt')
    y1_test = np.empty(X1_test.shape[0]).astype(int)
    y1_test.fill(0)
    X2_test = load_data('../data/Test2.txt')
    y2_test = np.empty(X2_test.shape[0]).astype(int)
    y2_test.fill(1)
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_train, X_test, y_train, y_test


class LMS(object):
    def __init__(self, eta=0.001, iterations=10000):
        self.eta = eta
        self.iterations = iterations

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]

        n_data = X_train.shape[0]
        dim = X_train.shape[1]
        X_train = np.concatenate(
            [X_train, np.ones(n_data).reshape(n_data, 1)],
            axis=1)
        y_train = y_train.reshape((n_data, 1))
        w = np.ones(dim+1)
        for i in range(self.iterations):
            choice = np.random.randint(X_train.shape[0])
            predict = np.dot(X_train[choice], w)
            error = y_train[choice] - predict
            dw = self.eta * error * X_train[choice]
            w += dw
        self.w = w

    def predict(self, X_test):
        n_data = X_test.shape[0]
        X_test = np.concatenate(
            [X_test, np.ones(n_data).reshape(n_data, 1)],
            axis=1)
        y_pred = np.zeros(n_data).astype(int)
        for i, xt in enumerate(X_test):
            yp = np.dot(xt, self.w)
            y_pred[i] = np.argmin([(0-yp)**2, (1-yp)**2])
        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)


def main():
    # get kadai data
    X_train, X_test, y_train, y_test = get_kadai1_dataset()
    # plot train data
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], c='b', alpha=0.5, label='Train omega1')
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], c='r', alpha=0.5, label='Train omega2')
    # plot test data
    plt.scatter(X_test[y_test==0][:,0], X_test[y_test==0][:,1], c='g', alpha=0.5, label='Test omega1')
    plt.scatter(X_test[y_test==1][:,0], X_test[y_test==1][:,1], c='y', alpha=0.5, label='Test omega2')
    # lms computing
    lms = LMS()
    lms.fit(X_train, y_train)
    # y_pred = lms.predict(X_test)
    # plot classification surface
    x = np.arange(-3, 5)
    y = 1 / lms.w[1] * (0.5 - lms.w[0]*x - lms.w[2])
    plt.plot(x, y, 'r', label='Classification surface')
    # setup the figure
    plt.legend(loc=2)
    plt.ylim(None, 9)
    # plt.show()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('../output/kadai1_plot_{}.png'.format(now))
    # save score
    score = lms.score(X_test, y_test)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/kadai1_score_{}.txt'.format(now), 'w') as f:
        f.write('score: {}\n'.format(score))


if __name__ == '__main__':
    main()
