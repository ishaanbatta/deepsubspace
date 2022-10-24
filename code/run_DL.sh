#!/bin/bash

FKEY=$1

# Classification (3-way) 
# python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.00001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3Ddr_lowresMax' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 


# Classification (2-way)
# python run_DL.py --nReps 10 --nc 3 --bs 32 --lr 0.00001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3Ddr_lrMx' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 
# python run_DL.py --nReps 10 --nc 3 --bs 32 --lr 0.00001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3Ddr_lrMxLtG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 

# # Classification (10 way) - Run two iterations of one task on same GPU concurrently (pp=0,pp=1)
# python run_DL.py --nReps 100 --nc 3 --bs 32 --lr 0.00001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AlexNet3D_Dropout' --ssd '../in/SampleSplits/ADNI/3way/' --sm '../in/analysis_SCORE_SMRI_fpL.csv'  --scorename 'labels_3way' --nw 4 --cr 'clx' &

# python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 10 --bs 32 --lr 0.00001 --es 500 --pp 1 --es_va 1 --es_pat 20 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits/'  --scorename 'label' --nw 4 --cr 'clx' &

# Classification (2 way) - Run two iterations of one task on same GPU concurrently (pp=0,pp=1)
# python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 10 --bs 32 --lr 0.00001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_Sex/'  --scorename 'label' --nw 4 --cr 'clx' &

# python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 10 --bs 32 --lr 0.00001 --es 500 --pp 1 --es_va 1 --es_pat 20 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_Sex/'  --scorename 'label' --nw 4 --cr 'clx' &


# Regression

# python run_DL.py --nReps 10 --nc 1 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3Ddr_lrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename age --nw 4 --cr 'reg' 
for SCORE in `cat ../in/scores_ADNI.txt`
do
    python run_DL.py --nReps 10 --nc 1 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3Ddr_lrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename "$SCORE" --nw 4 --cr 'reg' 
    sleep 1s 
done



#python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 1 --bs 32 --lr 0.001 --es 500 --pp 1 --es_va 1 --es_pat 40 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_Age/'  --scorename 'age' --nw 4 --cr 'reg' &

wait