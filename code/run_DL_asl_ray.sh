#!/bin/bash

FKEY=$1

# Regression
TUNE_DISABLE_STRICT_METRIC_CHECKING=1
export TUNE_DISABLE_STRICT_METRIC_CHECKING=1
python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/raytune_ADNIt1/' --mt 'BrASLnet5' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/ADNI/3way_SMRIrelP_10fold/' --sm '/data/users2/ibatta/projects/deepsubspace/in/analysis_SCORE_SMRI_relP_imputed.csv' --fkey x  --scorename labels3way --nw 4 --cr 'clx' --raytune


wait