#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Challenge3

Assignment:
    * http://www.mi.t.u-tokyo.ac.jp/harada/lectures/pattern/patternreport20150116.pdf

Reference:
    * http://deeplearning.net/tutorial/dA.html
    * http://scikit-learn.org/stable/modules/svm.html

"""
from __future__ import print_function, division
import collections

import numpy as np
from skimage.io import imread
# from sklearn.svm import SVC
from sklearn import preprocessing
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression

__author__ = "www.kentaro.wada@gmail.com (Kentaro Wada)"


def get_jaffe_data():
    label_data = np.loadtxt('../data/jaffe/label.txt', dtype=str)

    sd_rates = label_data[:, 1:-2].astype(float)
    # HAP: 0, SAD: 1, SUR: 2, ANG:3, DIS:4, FEA:5, PIC:6
    labels = np.argmax(sd_rates, axis=1)

    # get img data
    images = None
    skip = []
    for i, (label_id, label_nm) in enumerate(
            zip(label_data[:, 0], label_data[:, -1])):
        # get img path
        img_path = '../data/jaffe/'
        img_path += label_nm.replace('-', '.')
        img_path += '.{}.tiff'.format(label_id)
        # load img
        try:
            img = imread(img_path)
        except IOError:
            skip.append(i)
            continue
        # store img data
        if images is None:
            images = np.zeros((label_data.shape[0], img.shape[0], img.shape[1]))
        images[i] = img

    # get clean index
    index = np.array(
        [i for i in range(label_data.shape[0]) if i not in skip])

    # get person labeling index
    person_data = collections.defaultdict(list)
    for i, label_nm in enumerate(label_data[:, -1][index]):
        person = label_nm.split('-')[0]
        person_data[person].append(i)

    return images[index], labels[index], person_data


def get_challenge3_dataset():
    images, labels, person_data = get_jaffe_data()

    from autoencoder import AutoEncoder
    def sigmoid(x):
        import theano.tensor as T
        return 1. / (1. + T.exp(-1 * x))
    method = 'origin'
    if method == 'autoencoder':
        X = np.zeros((images.shape[0], images[0].size))
        for i, img in enumerate(images):
            X[i] = img.reshape(-1)
        ae = AutoEncoder(X,
            hidden_size=1000,
            activation_function=sigmoid,
            output_function=sigmoid)
        ae.train(n_epochs=10)
        X = ae.get_hidden(X)[0]
    elif method == 'origin':
        X = np.zeros((images.shape[0], images[0].size))
        for i, img in enumerate(images):
            X[i] = img.reshape(-1)
    elif method == 'hog':
        X = np.zeros((images.shape[0], 72900))
        for i, img in enumerate(images):
            hog_img = hog(img)
            X[i] = hog_img
    elif method == 'hog+autoencoder':
        X = np.zeros((images.shape[0], 72900))
        for i, img in enumerate(images):
            hog_img = hog(img)
            X[i] = hog_img
        ae = AutoEncoder(X,
            hidden_size=1000,
            activation_function=sigmoid,
            output_function=sigmoid)
        ae.train(n_epochs=100)
        X = ae.get_hidden(X)[0]
    return X, labels, person_data


def main():
    print("... getting dataset")
    X, y, person_data = get_challenge3_dataset()

    # params = {'C': 1e0, 'kernel': 'rbf'} # hog: 0.475244682853 # origin: 0.467028044419 
    # params = {'C': 1e1, 'kernel': 'rbf'} # 0.523185582533 # origin: 0.506211180124
    # params = {'C': 1e2, 'kernel': 'rbf'} # 0.523185582533
    # params = {'C': 1e3, 'kernel': 'rbf'} # 0.523185582533
    # params = {'C': 1e4, 'kernel': 'rbf'} # 0.523185582533
    # params = {'C': 1e0, 'kernel': 'linear'} # hog: 0.588788819876 # origin: 0.581283643892
    # params = {'C': 1e1, 'kernel': 'linear'} # 0.588788819876 # 0.581283643892
    # params = {'C': 1e2, 'kernel': 'linear'} # 0.588788819876
    # params = {'C': 1e3, 'kernel': 'linear'} # 0.588788819876
    # params = {'C': 1e4, 'kernel': 'linear'} # 0.588788819876
    # clf = SVC(**params)

    # params = {'C': 1e0}  # hog: 0.612842085451 # origin: 0.660273856578
    # params = {'C': 1e1}  # 0.617842085451 # 0.655926030491
    # params = {'C': 1e2}  # 0.608080180689 # 0.645966497271
    params = {'C': 1e3}  # 0.612625635234 # 0.669559570864
    # params = {'C': 1e4}  # 0.612625635234 # 0.66086391869
    # params = {'C': 1e5}  # 0.612625635234 # 0.655449840015
    clf = LogisticRegression(**params)

    print('... params: {}'.format(params))

    # get scores
    print('... getting scores')
    scores = []
    for person, test_index in person_data.items():
        print('==> person: {}'.format(person), end='')
        # get data
        X_test, y_test = X[test_index], y[test_index]
        X_test = preprocessing.scale(X_test)
        train_index = np.array(
            [i for i in range(X.shape[0]) if i not in test_index])
        X_train, y_train = X[train_index], y[train_index]
        X_train = preprocessing.scale(X_train)
        # train and get score
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(', score: {}'.format(score))
        scores.append(score)
    score = np.array(scores).mean()
    print("score: {}".format(score))


if __name__ == '__main__':
    main()

