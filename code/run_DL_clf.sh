#!/bin/bash

# Classification (2-way) 
python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/latest/' --mt 'AN3DdrlrMx' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 
# python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/latest/' --mt 'AN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 

# Classification (3-way)
python run_DL.py --nReps 10 --nc 3 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/latest/' --mt 'AN3DdrlrMx' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 
# python run_DL.py --nReps 10 --nc 3 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/latest/' --mt 'AN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 


wait