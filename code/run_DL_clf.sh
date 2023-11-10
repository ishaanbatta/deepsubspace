#!/bin/bash

# Classification (2-way) 

# python run_DL.py $1 $2 --ds BSNIP --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/BSNIP/' --mt 'AN3DdrhrMx' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/BSNIP/2way_10fold/' --sm '/data/users2/ibatta/projects/deepsubspace/in/BSNIP/analysis_SCORE_BSNIP.csv' --fkey x  --scorename labelsPR --nw 4 --cr 'clx'

python run_DL.py $1 $2 --ds BSNIP --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/BSNIP/' --mt 'AN3DdrlrMx' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/BSNIP/2way_PR_10fold_SAD/' --sm '/data/users2/ibatta/projects/deepsubspace/in/BSNIP/analysis_SCORE_BSNIP.csv' --fkey x  --scorename labelsSAD --nw 4 --cr 'clx'
python run_DL.py $1 $2 --ds BSNIP --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/BSNIP/' --mt 'AN3DdrlrMx' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/BSNIP/2way_PR_10fold_BP/' --sm '/data/users2/ibatta/projects/deepsubspace/in/BSNIP/analysis_SCORE_BSNIP.csv' --fkey x  --scorename labelsBP --nw 4 --cr 'clx'

# python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '/data/users2/ibatta/projects/deepsubspace/out/results/raytune_ADNIt1/' --mt 'AN3DdrlrMx' --ssd '/data/users2/ibatta/projects/deepsubspace/in/SampleSplits/ADNI/3way_SMRIrelP_10fold/' --sm '/data/users2/ibatta/projects/deepsubspace/in/analysis_SCORE_SMRI_relP_imputed.csv' --fkey x  --scorename labels3way --nw 4 --cr 'clx'


# python run_DL.py --nReps 10 --nc 2 --bs $1 --lr $2 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3DdrlrMx' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 

# python run_DL.py --nReps 10 --nc 2 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/latest/' --mt 'AN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/2way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 

# # Classification (3-way)
# python run_DL.py --nReps 10 --nc 3 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/' --mt 'AN3DdrlrMx' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 
# # python run_DL.py --nReps 10 --nc 3 --bs 32 --lr 0.0001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../out/results/latest/' --mt 'AN3DdrlrMxG' --ssd '../in/SampleSplits/ADNI/3way_MMR180d/' --sm '../in/analysis_SCORE_MMR180d.csv' --fkey x  --scorename 'labels_3way' --nw 4 --cr 'clx' 


wait