#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Assignment3

Assignment:
    * http://www.mi.t.u-tokyo.ac.jp/harada/lectures/pattern/patternreport20150116.pdf

"""
from __future__ import division, print_function
import datetime

from kadai1 import get_kadai1_dataset
from kadai3_1 import KNearestNeighbor, analyze_kvalue

__author__ = "www.kentaro.wada@gmail.com (Kentaro Wada)"


def main():
    # get kadai data
    X_train, X_test, y_train, y_test = get_kadai1_dataset()
    # variate k from 1 to 10
    best_k = analyze_kvalue(do_plot=False)
    knn = KNearestNeighbor(k=best_k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    # save the result
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/kadai3_score.txt'.format(now), 'w') as f:
        f.write('score: {}\n'.format(score))


if __name__ == '__main__':
    main()

