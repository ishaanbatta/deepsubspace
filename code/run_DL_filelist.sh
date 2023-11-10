#!/bin/bash

FKEY=$1
MODE=$2

FP=`sed -n ${FKEY}p ../in/manual_input_configs.txt`

echo $FP
python run_DL.py --loadconfig ../out/results/${FP} $MODE

wait
