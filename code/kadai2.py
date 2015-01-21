#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division, print_function
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from kadai1 import load_data, LMS


def pinv(A):
    A = np.atleast_2d(A)
    return np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)


class WithPinv(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        n_data = X_train.shape[0]
        dim = X_train.shape[1]
        X_train = np.concatenate(
            [X_train, np.ones(n_data).reshape(n_data, 1)],
            axis=1)
        y_train = y_train.reshape((n_data, 1))
        self.w = np.dot(pinv(X_train), y_train)

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
    # get train dataset
    X1_train = load_data('../data/Train1.txt')
    y1 = np.empty(X1_train.shape[0]).astype(int)
    y1.fill(0)
    X2_train = load_data('../data/Train2.txt')
    y2 = np.empty(X2_train.shape[0]).astype(int)
    y2.fill(1)
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1, y2))
    # plot train data
    plt.scatter(X1_train[:,0], X1_train[:,1], c='b', alpha=0.5, label='Train omega1')
    plt.scatter(X2_train[:,0], X2_train[:,1], c='r', alpha=0.5, label='Train omega2')
    # get test dataset
    X1_test = load_data('../data/Test1.txt')
    y1_test = np.empty(X1_test.shape[0]).astype(int)
    y1_test.fill(0)
    X2_test = load_data('../data/Test2.txt')
    y2_test = np.empty(X2_test.shape[0]).astype(int)
    y2_test.fill(1)
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    # plot test data
    plt.scatter(X1_test[:,0], X1_test[:,1], c='g', alpha=0.5, label='Test omega1')
    plt.scatter(X2_test[:,0], X2_test[:,1], c='y', alpha=0.5, label='Test omega2')
    # compute weight with LMS
    lms = LMS()
    lms.fit(X_train, y_train)
    y_pred = lms.predict(X_test)
    x = np.arange(-1, 5)
    y_lms = 1 / lms.w[1] * (0.5 - lms.w[0]*x - lms.w[2])
    plt.plot(x, y_lms, c='r', label='Classification surface (with LMS)', alpha=0.5)
    # compute weight with pseudoinverse
    with_pinv = WithPinv()
    with_pinv.fit(X_train, y_train)
    y_with_pinv = 1 / with_pinv.w[1] * (0.5 - with_pinv.w[0]*x - with_pinv.w[2])
    plt.plot(x, y_with_pinv, c='b', label='Classification surface (with pseudoinverse)', alpha=0.5)
    # setup the figure
    plt.legend(loc=2)
    plt.ylim(None, 13)
    # plt.show()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('../output/kadai2_plot_{}.png'.format(now))

    # save score
    score_lms = lms.score(X_test, y_test)
    score_with_pinv = with_pinv.score(X_test, y_test)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/kadai2_score_{}.txt'.format(now), 'w') as f:
        f.write('score_lms: {}\n'.format(score_lms))
        f.write('score_with_pinv: {}\n'.format(score_with_pinv))


if __name__ == '__main__':
    main()
