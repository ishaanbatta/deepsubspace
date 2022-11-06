#!/bin/bash

PYOPENGL_PLATFORM="osmesa"
defaultserverargs="$defaultserverargs +iglx"
FL=$1
DRMIN=$2
DRMAX=$3

STR="../../../data/masks/ADNI/rMNI152_T1_1mm_brain.nii"
BASE="../out/saliency_stats/"

for FN in `cat $FL`
do 
    echo $FN
    FILENAME="$BASE$FN"
    OUTFILE=${FILENAME//".nii"/""}
    OUTFILE=${OUTFILE//".gz"/""}
    OUTFILE="$OUTFILE.png"
    fsleyes render -z 20 --scene lightbox -nc 9 -nr 6 -of $OUTFILE  $STR $FILENAME -cm red-yellow -un -nc blue-lightblue -dr $DRMIN $DRMAX
done
