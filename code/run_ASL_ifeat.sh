#!/bin/bash

SCR=$1
REPT=$2

echo "SCR: ${SCR}, REPT: ${REPT}"

if [[ "$SCR" == "age" ]]
then 
    FKEY="4"
fi 

if [[ "$SCR" == "MMSE" ]]
then 
    FKEY="3"
fi


FP=`sed -n ${FKEY}p ../in/ASL_ifeat_input_dirs.txt`

# INDIR="../out/results/mt_BrASLnet2_fkey_lT1_scorename_${SCR}_iter_10${REPT}_nc_1_REPT_${REPT}_bs_16_lr_0.01_espat_20_sf_16/"
INDIR="../out/results/${FP}/"


python ../../subspace/code/learning.py ADNI all hT1 $SCR PSVR $REPT AN3DdrhrMx ../out/predictions/ADNIt1/all_hT1_${SCR}_PSVR_AN3DdrhrMx.csv "$INDIR"  --dry-run

