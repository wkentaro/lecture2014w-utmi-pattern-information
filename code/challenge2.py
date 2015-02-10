#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Challenge2

Reference:
    * Pattern Recognisiona and Machine Learning, P202-207

"""
from __future__ import print_function, division
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from kadai1 import get_kadai1_dataset


def sigmoid(x, a=-1):
    return 1. / (1 + np.exp(a * x))


class LogisticRegressionTwoClass(object):
    def __init__(self, iterations):
        self.iterations = iterations

    def fit(self, X, t):
        # initialize
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        # W = np.random.random((X.shape[1], 1))
        W = np.zeros((X.shape[1], 1))
        for i in xrange(self.iterations):
            # compute probability
            prob = self._probability(X, W)
            # Newton-Lapson method
            R = np.diag(prob * (1 - prob))
            H = np.dot(np.dot(X.T, R), X)  # \delta \delta E(w)
            # update weight
            dW = -np.linalg.inv(H).dot(X.T).dot(np.atleast_2d(prob-t).T)
            W += dW
            if np.linalg.norm(dW) < 0.01:
                break
            # z = np.dot(X, W) - np.dot(
            #     np.linalg.inv(R),
            #     np.atleast_2d(prob-t).T)
            # W = np.dot(np.dot(np.dot(np.linalg.inv(H), X.T), R), z)
        self.W = W

    def predict(self, X):
        # initialize
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        # probability
        prob = self._probability(X, self.W)
        prob_tbl = np.vstack((prob, 1-prob)).T
        return np.argmin(prob_tbl, axis=1)

    def score(self, X_test, y_true):
        y_pred = self.predict(X_test)
        return accuracy_score(y_true, y_pred)

    def _probability(self, X, W):
        # [p(\omega_1 | x0), p(\omega_1 | x1), ... ]
        return sigmoid(np.dot(X, W).reshape(-1))


def main():
    X_train, X_test, y_train, y_test = get_kadai1_dataset()

    lr = LogisticRegressionTwoClass(iterations=10000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("score: {}".format(accuracy_score(y_test, y_pred)))

    # plot train data
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], c='b', alpha=0.5, label='Train omega1')
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], c='r', alpha=0.5, label='Train omega2')
    # plot test data
    plt.scatter(X_test[y_test==0][:,0], X_test[y_test==0][:,1], c='g', alpha=0.5, label='Test omega1')
    plt.scatter(X_test[y_test==1][:,0], X_test[y_test==1][:,1], c='y', alpha=0.5, label='Test omega2')
    # plot classification surface
    x = np.arange(-3, 5)
    y = 1 / lr.W[1] * (0.5 - lr.W[0]*x - lr.W[2])
    plt.plot(x, y, 'r', label='Classification surface')
    # setup the figure
    plt.legend(loc=2)
    plt.ylim(None, 9)
    # plt.show()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('../output/challenge2_plot_{}.png'.format(now))
    # save score
    score = lr.score(X_test, y_test)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/challenge2_score_{}.txt'.format(now), 'w') as f:
        f.write('score: {}\n'.format(score))

if __name__ == '__main__':
    main()
