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

    def within_class_cov(self, X, y):
        return np.sum(
            [np.cov(X[y==cls], rowvar=0) for cls in np.unique(y)],
            axis=0)

    def between_class_cov(self, X, y):
        mu = np.mean(X, axis=0)
        class_covs = []
        for cls in np.unique(y):
            delta = X[y == cls].mean(axis=0) - mu
            delta = np.atleast_2d(delta).T
            class_covs.append(sum(y==cls) * np.dot(delta, delta.T))
        return np.sum(class_covs, axis=0)


def main():
    X, y = get_iris_dataset()

    # PCA
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca_trans = pca.transform(X)
    # Fisher LDA
    flda = FisherLDA(n_components=1)
    flda.fit(X, y)
    X_flda_trans = flda.transform(X)
    ## plot
    plt.title('FisherLDA decomposition')
    X_pca_trans -= X_pca_trans.min()
    X_pca_trans /= X_pca_trans.max()
    X_flda_trans -= X_flda_trans.min()
    X_flda_trans /= X_flda_trans.max()
    for i, cls in enumerate(np.unique(y)):
        x_pca = X_pca_trans[y == cls]
        plt.plot(x_pca, np.ones(len(x_pca)) + 0.05*i,
            label='class{} (PCA)'.format(cls))
        x_flda = X_flda_trans[y==cls]
        plt.plot(x_flda, np.zeros(len(x_flda)) + 0.05*i,
            label='class{} (FisherLDA)'.format(cls))
    plt.ylim(-0.5, 3)
    plt.legend(loc=2)
    # plt.show()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('../output/challenge1_plot_{}.png'.format(now))


if __name__ == '__main__':
    main()

