#!/bin/bash
sbatch --array=0-9 ./JSA_DL.sh run_DL_optimized 1
sbatch --array=10-19 ./JSA_DL.sh run_DL_optimized 2
sbatch --array=20-29 ./JSA_DL.sh run_DL_optimized 3
sbatch --array=30-39 ./JSA_DL.sh run_DL_optimized 4 
sbatch --array=40-49 ./JSA_DL.sh run_DL_optimized 5
sbatch --array=50-59 ./JSA_DL.sh run_DL_optimized 6
sbatch --array=60-69 ./JSA_DL.sh run_DL_optimized 7
sbatch --array=70-79 ./JSA_DL.sh run_DL_optimized 8
sbatch --array=80-89 ./JSA_DL.sh run_DL_optimized 9
sbatch --array=90-99 ./JSA_DL.sh run_DL_optimized 10
