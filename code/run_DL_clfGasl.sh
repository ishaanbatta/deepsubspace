#!/bin/bash

python run_DL.py --thr 0.03 --ds BSNIP --nReps 10 --nc 2 --sf 8 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/BSNIP/' --mt 'BrASLnet5' --ssd '../in/SampleSplits/BSNIP/2way_10fold/' --sm '../in/BSNIP/analysis_SCORE_BSNIP.csv' --fkey x  --scorename 'labelsPR' --nw 4 --cr 'clx' 
# python run_DL.py --thr 0.03 --ds BSNIP --nReps 10 --nc 2 --sf 8 --bs 8 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/BSNIP/' --mt 'BrASLnet5' --ssd '../in/SampleSplits/BSNIP/2way_DX_10fold/' --sm '../in/BSNIP/analysis_SCORE_BSNIP.csv' --fkey x  --scorename 'labelsDX' --nw 4 --cr 'clx' 

# python run_DL.py --thr 0.03 --nReps 10 --nc 2 --sf 8 --bs 8 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/threshmask/' --mt 'BrASLnet5' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 

# python run_DL.py --nReps 10 --nc 2 --sf 8 --bs 8 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'BrASLnet2' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 


# # Classification (2-way) 
# python run_DL.py --nReps 10 --nc 2 --sf $1 --bs 8 --lr $2 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'BrASLnet' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 
# sleep 2s
# python run_DL.py --nReps 10 --nc 2 --sf $1 --bs 16 --lr $2 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'BrASLnet' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 



# Classification (3-way)
# python run_DL.py --nReps 10 --nc 3 --bs 16 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'ASLnetAN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 


wait