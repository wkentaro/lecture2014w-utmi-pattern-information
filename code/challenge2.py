#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function, division

# import numpy as np
# import matplotlib.pyplot as plt

from kadai1 import get_kadai1_dataset


class LogisticRegression(object):
    def __init__(self):
        pass


def main():
    X_train, X_test, y_train, y_test = get_kadai1_dataset()


if __name__ == '__main__':
    main()