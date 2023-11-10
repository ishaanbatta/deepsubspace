#!/bin/bash

ARR=$1

for LRT in 0.01 0.001
do
    for SCL in "age" "MMSE" 
    do
    echo "------------ lr ${LRT}, bs/sf/score ${SCL} ------------"
    sbatch --array=$ARR ./JSA_DL.sh run_DL_regASL ${SCL} ${LRT} 
    sleep 1s
    done
done
