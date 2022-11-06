#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRD
#SBATCH -c 2
#SBATCH --mem-per-cpu=8g
#SBATCH -t 1440
#SBATCH -J Summ
#SBATCH -e ../out/slogs/%x-%A-%a.err
#SBATCH -o ../out/slogs/%x-%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe 
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ibatta@gsu.edu

CL=$1

sleep 3s

echo "job id: $SLURM_JOB_ID, task id:$SLURM_ARRAY_TASK_ID"


export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /data/users2/ibatta/venv/conda/anaconda/bin/activate
conda activate AA_DL2
conda info --envs
which python 

sleep 1s 

# ./monitor_GPU.sh & 

python summarize.py $CL random

sleep 3s

# sacct --format=user,jobID,MaxRSS,MaxVMSize,CPUTime | grep $SLURM_JOB_ID
 