#!/bin/bash

# N=`cat ../out/nc1LatestFiles_originalName.txt  |  wc -l `
N="80"
D="/data/users2/ibatta/projects/deepsubspace/out/results/latest/"

cd $D

for IND in `seq 1 $N`
do 
    echo $IND
    A=`sed -n ${IND}p < ../../temp_org.txt`
    B=`sed -n ${IND}p < ../../temp_new.txt`
    echo "mv $D$A $D$B"
    mv $D$A $D$B
done

