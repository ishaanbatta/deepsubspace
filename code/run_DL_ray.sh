#!/bin/bash



# Regression
TUNE_DISABLE_STRICT_METRIC_CHECKING=1
export TUNE_DISABLE_STRICT_METRIC_CHECKING=1

python run_DL.py $1 $2 --ds BSNIP --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/BSNIP/raytune/' --mt 'AN3DdrlrMx' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/BSNIP/2way_10fold/' --sm '/data/users2/ibatta/projects/deepsubspace/in/BSNIP/analysis_SCORE_BSNIP.csv' --fkey x  --scorename labelsPR --nw 4 --cr 'clx' --raytune

# python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/raytune_ADNIt1/' --mt 'AN3DdrlrMx' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/ADNI/3way_SMRIrelP_10fold/' --sm '/data/users2/ibatta/projects/deepsubspace/in/analysis_SCORE_SMRI_relP_imputed.csv' --fkey x  --scorename labels3way --nw 4 --cr 'clx' --raytune

# python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/raytune/' --mt 'AN3DdrlrMx' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/ADNI/3way_MMR180d/' --sm '/data/users2/ibatta/projects/deepsubspace/in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename age --nw 4 --cr 'clx' --raytune

# for SCORE in `cat ../in/scores_ADNI.txt`
# do
#     python run_DL.py --nReps 10 --nc 1 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '../out/results/' --mt 'AN3DdrlrMx' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename "$SCORE" --nw 4 --cr 'reg' 
#     sleep 1s 
# done


wait