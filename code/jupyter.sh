#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=64g
#SBATCH -p qTRDHM
#SBATCH -t 7200
#SBATCH -J jt
#SBATCH --output=jupyter-%j.out
#SBATCH -A trends53c17
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ibatta@gsu.edu
#SBATCH --oversubscribe

export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

# eval "$(/data/users2/ibatta/venv/conda/anaconda/bin/conda shell.bash hook)"

source /data/users2/ibatta/venv/conda/anaconda/bin/activate

conda activate AA_DL2
cat /etc/hosts
jupyter-lab --ip=0.0.0.0 --port=${1:-420}
