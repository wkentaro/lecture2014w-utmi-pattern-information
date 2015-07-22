========================
utmi-pattern-information
========================
| Assignments of class in UTokyo "Pattern Information".
| http://www.mi.t.u-tokyo.ac.jp/harada/lectures/pattern/patternreport20150116.pdf

Report
======
* `View Report <https://github.com/wkentaro/utmi-pattern-information/blob/master/report/pattern_report2014.pdf>`_
* `Download Report <https://github.com/wkentaro/utmi-pattern-information/raw/master/report/pattern_report2014.pdf>`_


Summary
=======

* Assignment1: Windrow-Hoff Algorithm to get decisive surface.
* Assignment2: Use pseudo-inverse to map data to label.
* Assignment3: K-Nearest Neighbors.
* Assignment4: Probability generation model.
* Challenge1: PCA and FisherLDA.
* Challenge2: LogisticRegression for 2 classes.
* Challenge3: Emotion discrimination.


Codes
=====

Execute below to get output in :code:`output/` dir.


Python code

.. code-block :: bash

  $ cd code
  $ python kadai1.py       # Assignment1
  $ python kadai2.py       # Assignment2
  $ python kadai3_1.py     # Assignment3
  $ python kadai3_2.py     # Assignment3
  $ python kadai4_1.py     # Assignment4
  $ python kadai4_2.py     # Assignment4
  $ python kadai4_3.py     # Assignment4
  $ python challenge1.py   # Challenge1
  $ python challenge2.py   # Challenge2
  $ python challenge3.py   # Challenge3


C++ code

.. code-block :: bash

  $ cd code/cpp
  $ make
  $ ./kadai1


Euslisp code

.. code-block :: bash

  $ cd code/euslisp
  $ eus kadai1.lisp
