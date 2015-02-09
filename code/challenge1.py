#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function, division
import datetime

import numpy as np
import matplotlib.pyplot as plt


def get_iris_dataset():
    data = np.loadtxt('../data/iris.txt')
    X = data[:, :3]
    y = data[:, 4].astype(int)
    return X, y


class PCA(object):
    """Principal Component Analysis"""
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_components = self.n_components

        V = np.cov(X, rowvar=0)
        w, v = np.linalg.eig(V)

        W = np.zeros((X.shape[1], n_components))
        for i, index in enumerate(np.argsort(w)[::-1][:n_components]):
            W[:, i] = v[:, index]
        self.W = W

    def transform(self, X):
        return np.dot(X, self.W)  # for without iteration


class FisherLDA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        n_components = self.n_components

        Sw = self.within_class_cov(X, y)
        Sb = self.between_class_cov(X, y)

        w, v = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
        W = np.zeros((X.shape[1], n_components))
        for i, index in enumerate(np.argsort(w)[::-1][:n_components]):
            W[:, i] = v[:, index]
        self.W = W

    def transform(self, X):
        return np.dot(X, self.W)

    def between_class_cov(self, X, y):
        mu = np.mean(X, axis=0)
        class_covs = []
        for cls in np.unique(y):
            delta = X[y == cls].mean(axis=0) - mu
            class_covs.append(np.dot(delta, delta.T))
        return np.sum(class_covs, axis=0)

    def within_class_cov(self, X, y):
        return np.sum(
            [np.cov(X[y==cls], rowvar=0) for cls in np.unique(y)],
            axis=0)


def main():
    X, y = get_iris_dataset()

    # PCA
    ## decomposition
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca_trans = pca.transform(X)
    ## plot
    plt.title('PCA decomposition')
    for cls in np.unique(y):
        x = X_pca_trans[y == cls]
        plt.plot(x, np.zeros(len(x)), label='class{}'.format(cls))
    plt.legend(loc=2)
    # plt.show()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('../output/challenge1_pca_plot_{}.png'.format(now))
    plt.cla()

    # Fisher LDA
    ## decomposition
    flda = FisherLDA(n_components=1)
    flda.fit(X, y)
    X_flda_trans = flda.transform(X)
    ## plot
    plt.title('FisherLDA decomposition')
    for cls in np.unique(y):
        x = X_flda_trans[y==cls]
        plt.plot(x, np.zeros(len(x)), label='class{}'.format(cls))
    plt.legend(loc=2)
    # plt.show()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('../output/challenge1_flda_plot_{}.png'.format(now))
    plt.cla()


if __name__ == '__main__':
    main()

