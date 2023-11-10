#!/bin/bash

FKEY=$1
SLIND=$2
SUMM=$3

SLVR=`sed -n ${SLIND}p ../in/salvars.txt`
FP=`sed -n ${FKEY}p ../in/manual_input_configs.txt`



echo $FP
echo $SLVR

python run_DL.py --loadconfig ../out/results/${FP} --groupICA $SUMM --salvars $SLVR

wait
