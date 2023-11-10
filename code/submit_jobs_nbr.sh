#!/bin/bash

ARR=$1

for LRT in 0.01 0.001 0.0001 0.00001 
do
    for BSZ in 8 16 32 64
    do
    echo "------------ lr ${LRT}, bs ${BSZ} ------------"
    sbatch --array=$ARR ./JSA_DL.sh run_DL_clf ${BSZ} ${LRT}
    sleep 1s
    done
done
