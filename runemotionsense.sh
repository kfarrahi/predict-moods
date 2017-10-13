#!/bin/sh

python testallinputs.py data/binaryPredNextWeekMedian.txt 2 gru 3 100
python testallinputs.py data/binaryPredNextWeekMedian.txt 2 lstm 3 100

python testallinputs.py data/binaryPredNextWeekMedian.txt 2 gru 3 25
python testallinputs.py data/binaryPredNextWeekMedian.txt 2 lstm 3 25

python testallinputs.py data/binaryPredNextWeekMedian.txt 2 gru 3 50
python testallinputs.py data/binaryPredNextWeekMedian.txt 2 lstm 3 50

python testallinputs.py data/binaryPredNextWeekAvg.txt 2 gru 3 25
python testallinputs.py data/binaryPredNextWeekAvg.txt 2 lstm 3 25

python testallinputs.py data/binaryPredNextWeekIncDrop.txt 2 gru 3 25
python testallinputs.py data/binaryPredNextWeekIncDrop.txt 2 lstm 3 25

python testallinputs.py data/binaryPredNextWeekMedian_equal.txt 3 gru 3 25
python testallinputs.py data/binaryPredNextWeekMedian_equal.txt 3 lstm 3 25

python testallinputs.py data/binaryPredNextWeekAvg_equal.txt 3 gru 3 25
python testallinputs.py data/binaryPredNextWeekAvg_equal.txt 3 lstm 3 25

python testallinputs.py data/binaryPredNextWeekIncDropEqual.txt 3 gru 3 25
python testallinputs.py data/binaryPredNextWeekIncDropEqual.txt 3 lstm 3 25

#python testallinputs.py data/binaryPredNextWeekIncDropEqual_normalised.txt 3 gru 3 25