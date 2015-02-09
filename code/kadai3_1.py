#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Assignment3

Assignment:
    * http://www.mi.t.u-tokyo.ac.jp/harada/lectures/pattern/patternreport20150116.pdf

"""
from __future__ import division, print_function
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from kadai1 import get_kadai1_dataset

__author__ = "www.kentaro.wada@gmail.com (Kentaro Wada)"


class KNearestNeighbor(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for i, xt in enumerate(X_test):
            distance_map = np.square(self.X_train - xt).sum(axis=1)
            nearests = np.argsort(distance_map)[:self.k]
            n_nonzero = np.count_nonzero(self.y_train[nearests])
            if n_nonzero > self.k - n_nonzero:
                yp = 1
            elif n_nonzero < self.k - n_nonzero:
                yp = 0
            else:
                yp = np.random.randint(2)  # randomly returns 0 or 1
            y_pred[i] = yp
        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def train_self_score(self, X_train, y_train):
        y_pred = np.zeros(X_train.shape[0])
        index = np.arange(X_train.shape[0])
        for i, xt in enumerate(X_train):
            distance_map = np.square(X_train[index!=i] - xt).sum(axis=1)
            nearests = np.argsort(distance_map)[:self.k]
            n_nonzero = np.count_nonzero(y_train[index!=i][nearests])
            if n_nonzero > self.k - n_nonzero:
                yp = 1
            elif n_nonzero < self.k - n_nonzero:
                yp = 0
            else:
                yp = np.random.randint(2)
            y_pred[i] = yp
        score = accuracy_score(y_train, y_pred)
        return score


def analyze_kvalue(do_plot=True):
    # get kadai data
    X_train, _, y_train, _ = get_kadai1_dataset()
    # variate k from 1 to 10
    scores = []
    for k in range(1, 11):
        knn = KNearestNeighbor(k=k)
        score = knn.train_self_score(X_train, y_train)
        scores.append(score)

    if do_plot is True:
        # plot the result
        x = np.arange(1, 11)
        scores = np.array(scores)
        plt.title('k Nearest Neighbor, Relationship between k value and precision')
        plt.xlabel('k value')
        plt.ylabel('precision')
        plt.plot(x, scores)
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig('../output/kadai3_various_kvalue_{}.png'.format(now))

    return np.argmax(scores) + 1  # k value of best precision


if __name__ == '__main__':
    analyze_kvalue()
