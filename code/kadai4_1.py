#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Assignment4

Assignment:
    * http://www.mi.t.u-tokyo.ac.jp/harada/lectures/pattern/patternreport20150116.pdf

Reference:
    * http://www.developerstation.org/2012/03/mahalanobis-distance-in-opencv.html

"""
from __future__ import division, print_function
import datetime

import numpy as np
from scipy.spatial.distance import mahalanobis

from kadai1 import load_data

__author__ = "www.kentaro.wada@gmail.com (Kentaro Wada)"


def get_kadai4_data():
    omega1 = load_data('../data/omega1.txt')
    omega2 = load_data('../data/omega2.txt')
    omega3 = load_data('../data/omega3.txt')
    return omega1, omega2, omega3


def get_test_points():
    test_points = np.array(
        [[1, 2, 1],
         [5, 3, 2],
         [0, 0, 0],
         [1, 0, 0]])
    return test_points

def compute_mahalanobis():
    omega1, omega2, omega3 = get_kadai4_data()
    mu1 = omega1.mean(axis=0)
    mu2 = omega2.mean(axis=0)
    mu3 = omega3.mean(axis=0)

    test_points = get_test_points()

    # compute mahalanobis distance between
    # test_points & mu of the set
    mds = np.zeros((len(test_points), 3))
    for i, tp in enumerate(test_points):
        md1 = mahalanobis(tp, mu1, np.cov(omega1, rowvar=0))
        md2 = mahalanobis(tp, mu2, np.cov(omega2, rowvar=0))
        md3 = mahalanobis(tp, mu3, np.cov(omega3, rowvar=0))
        mds[i] = [md1, md2, md3]

    return mds


if __name__ == '__main__':
    mds = compute_mahalanobis()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/kadai4_1_mahalanobis_{}.txt'.format(now), 'w') as f:
        for md in mds:
            md = map(str, md)
            f.write(','.join(md)+'\n')
