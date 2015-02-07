#!/usr/bin/env sh
# author: www.kentaro.wad@gmail.com (Kentaro Wada)
#
g++ kadai1.cpp -o kadai1
output=`./kadai1`
score=`echo $output | cut -f1 -d ','`
line=`echo $output | cut -f2 -d ','`

gnuplot <<- EOF
  set xlabel "x"
  set ylabel "y
  plot "../../data/Train1.txt" using 1:2
  replot "../../data/Train2.txt" using 1:2
  replot "../../data/Test1.txt" using 1:2
  replot "../../data/Test2.txt" using 1:2
  replot $line
  set term png
  set output "kadai1.png"
  replot
EOF

rm -f kadai1