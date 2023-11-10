#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRDHM
#SBATCH -c 4
#SBATCH --mem-per-cpu=32g
#SBATCH -t 1440
#SBATCH -J ASLDL
#SBATCH -e ../out/slogs/%x-%A-%a.err
#SBATCH -o ../out/slogs/%x-%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe 
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ibatta@gsu.edu


sleep 3s

SCR=$1
TID=$SLURM_ARRAY_TASK_ID

if [[ -v $SLURM_ARRAY_TASK_ID ]]; then
    $REP="0"
else
    $REP="$TID"
fi

echo "job id: $SLURM_JOB_ID, task id:$SLURM_ARRAY_TASK_ID"


export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /data/users2/ibatta/venv/conda/anaconda/bin/activate
conda activate AA_DL2
conda info --envs
which python 

sleep 1s 

# ./monitor_GPU.sh & 

./$SCR.sh $2 $SLURM_ARRAY_TASK_ID


# python ../../subspace/code/learning.py ADNI all lT1 $1 PSVR 0 BrASLnet2 ../out/predictions/ADNI/all_lT1_${1}_PSVR_BrASLnet2.csv $2 --dry-run

sleep 3s

# sacct --format=user,jobID,MaxRSS,MaxVMSize,CPUTime | grep $SLURM_JOB_ID
