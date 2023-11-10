#!/bin/bash

# Classification (2-way) 
python run_DL.py --nReps 10 --nc 2 --bs 8 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'BrASLnet' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 
