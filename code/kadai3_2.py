#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division, print_function
import datetime

from kadai1 import get_kadai1_dataset
import kadai3_1
from kadai3_1 import KNearestNeighbor


def main():
    # get kadai data
    X_train, X_test, y_train, y_test = get_kadai1_dataset()
    # variate k from 1 to 10
    scores = []
    best_k = kadai3_1.main()
    knn = KNearestNeighbor(k=best_k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    # save the result
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/kadai3_score.txt'.format(now), 'w') as f:
        f.write('score: {}\n'.format(score))


if __name__ == '__main__':
    main()
