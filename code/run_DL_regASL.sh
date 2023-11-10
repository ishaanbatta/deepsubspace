#!/bin/bash

python run_DL.py --thr 0.03 --nReps 10 --nc 1 --sf 16 --bs 16 --lr 0.01 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/ADNIt1/' --mt 'BrASLnet5' --ssd '../in/SampleSplits/ADNI/3way_SMRIrelP/' --sm '../in/analysis_SCORE_SMRI_relP_imputed.csv'  --fkey x  --scorename age --nw 4 --cr 'reg' 

sleep 5s 
python run_DL.py --thr 0.03 --nReps 10 --nc 1 --sf 4 --bs 32 --lr 0.01 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/ADNIt1/' --mt 'BrASLnet5' --ssd '../in/SampleSplits/ADNI/3way_SMRIrelP/' --sm '../in/analysis_SCORE_SMRI_relP_imputed.csv'  --fkey x  --scorename MMSE --nw 4 --cr 'reg' 

# python run_DL.py --nReps 10 --nc 1 --sf 8 --bs 8 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'meanASLnet' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename $1 --nw 4 --cr 'reg' 


# # # # Regression
# python run_DL.py --nReps 10 --nc 1 --sf 4 --bs 32 --lr $2 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'BrASLnet2' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename $1 --nw 4 --cr 'reg' 
# sleep 2s
# python run_DL.py --nReps 10 --nc 1 --sf 4 --bs 16 --lr $2 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'BrASLnet2' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename $1 --nw 4 --cr 'reg' 



# Classification (3-way)
# python run_DL.py --nReps 10 --nc 3 --bs 16 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'ASLnetAN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 


wait