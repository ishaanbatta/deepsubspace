#!/bin/bash

# FKEY=$1
SCN=$1
MDL=$2

# Regression


python run_DL.py --ds BSNIP --nReps 10 --nc 1 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/BSNIP/' --mt ${MDL} --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/BSNIP/4way_PR_PANSS_te10_va6/' --sm '/data/users2/ibatta/projects/deepsubspace/in/BSNIP/analysis_SCORE_BSNIP_PANSS.csv' --fkey x  --scorename $SCN --nw 4 --cr 'reg'


# python run_DL.py --thr 0.03  --nReps 10 --nc 1 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '../out/results/ADNIt1/' --mt 'AN3DdrhrMx' --ssd '../in/SampleSplits/ADNI/3way_SMRIrelP_10fold/' --sm '../in/analysis_SCORE_SMRI_relP_imputed.csv' --fkey x  --scorename age --nw 4 --cr 'reg' 

# python run_DL.py --thr 0.03  --nReps 10 --nc 1 --sf 4 --bs 32 --lr 0.01 --es 500 --pp 0 --es_va 1 --es_pat 60 --ml '../out/results/ADNIt1/' --mt 'AN3DdrhrMx' --ssd '../in/SampleSplits/ADNI/3way_SMRIrelP_10fold/' --sm '../in/analysis_SCORE_SMRI_relP_imputed.csv' --fkey x  --scorename MMSE --nw 4 --cr 'reg' 

# for SCORE in `cat ../in/scores_ADNI.txt`
# do
#     python run_DL.py --nReps 10 --nc 1 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename "$SCORE" --nw 4 --cr 'reg' 
#     sleep 1s 
# done

wait



