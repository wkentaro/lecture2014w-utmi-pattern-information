#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division, print_function
import datetime

import numpy as np

from kadai4_2 import generate_with_probability


def main():
    y_pred = generate_with_probability(probability=[.8, .1, .1])
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open('../output/kadai4_3_pred_{}.txt'.format(now), 'w') as f:
        y_pred = map(str, y_pred)
        f.write(','.join(y_pred))


if __name__ == '__main__':
    main()
