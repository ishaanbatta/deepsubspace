#!/bin/bash

FKEY=$1
# Config list in the config files must be sorted according to iterations to avoid manual figuring out of slurm array task IDs appropriately.
FP=`sed -n ${FKEY}p ../in/BSNIP/ray_optimized_configs_clf.txt`

echo $FP
python run_DL.py --loadconfig ../out/results/${FP} --thr 0.03


wait