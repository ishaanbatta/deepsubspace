#!/bin/bash

FKEY=$1

# Regression

# python run_DL.py --nReps 10 --nc 1 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename age --nw 4 --cr 'reg' 
for SCORE in `cat ../in/scores_ADNI.txt`
do
    python run_DL.py --nReps 10 --nc 1 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename "$SCORE" --nw 4 --cr 'reg' 
    sleep 1s 
done

wait