#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import numpy as np

from kadai4_2 import generate_with_probability


def main():
    y_pred = generate_with_probability(probability=[.8, .1, .1])
    print(y_pred)


if __name__ == '__main__':
    main()
