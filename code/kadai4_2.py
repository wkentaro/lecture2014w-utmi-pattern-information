#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import numpy as np

from kadai4_1 import (
    compute_mahalanobis,
    get_test_points,
    get_kadai4_data,
    )


def generate_with_probability(probability):
    omega = get_kadai4_data()

    y_pred = []
    mds = compute_mahalanobis()
    for i, md in enumerate(mds):
        g = []
        for j, omg in enumerate(omega):
            g.append(
                - 1 / 2 * md[j]**2
                - 1 / 2 * np.log(np.linalg.norm(np.cov(omg, rowvar=0)))
                - omg.shape[1]/2 * np.log(2 * np.pi)
                + np.log(probability[j]))
        g = np.array(g)
        # index = 0: omega1, 1: omega2, 2: omega3
        index = np.argmax(g)
        y_pred.append(index)
    return y_pred


if __name__ == '__main__':
    y_pred = generate_with_probability(probability=[1/3,1/3,1/3])
    print(y_pred)
