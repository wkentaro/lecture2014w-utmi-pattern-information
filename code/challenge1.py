#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function, division

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
        dim = X.shape[1]

        V = np.cov(X, rowvar=0)
        w, v = np.linalg.eig(V)

        W = np.zeros((dim, self.n_components))
        for i in np.argsort(w)[::-1][:self.n_components]:
            W[:, i] = v[:, i]
        self.W = W

    def transform(self, X):
        X_trans = np.zeros((X.shape[0], self.n_components))
        for i, xi in enumerate(X):
            X_trans[i] = np.dot(self.W.T, xi)
        return X_trans


class FisherLDA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self):
        return self

    def transform(self, X):
        return X


def main():
    X, _ = get_iris_dataset()
    # PCA decomposition
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca_trans = pca.transform(X)
    # Fisher LDA
    flda = FisherLDA()
    flda.fit(X)
    X_flda_trans = flda.transform(X)
    # plot
    plt.plot(np.zeros(X_pca_trans.shape[0]), X_pca_trans)
    plt.plot(np.zeros(X_flda_trans.shape[0]), X_flda_trans)


if __name__ == '__main__':
    main()
