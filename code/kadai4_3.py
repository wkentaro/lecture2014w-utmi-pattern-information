#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Assignment4

Assignment:
    * http://www.mi.t.u-tokyo.ac.jp/harada/lectures/pattern/patternreport20150116.pdf

"""
from __future__ import division, print_function
import datetime

from kadai4_2 import generate_with_probability

__author__ = "www.kentaro.wada@gmail.com (Kentaro Wada)"


def main():
    y_pred = generate_with_probability(probability=[.8, .1, .1])
    print("==> y_pred = {}".format(y_pred))

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/kadai4_3_pred_{}.txt'.format(now), 'w') as f:
        y_pred = map(str, y_pred)
        f.write(','.join(y_pred))


if __name__ == '__main__':
    main()

