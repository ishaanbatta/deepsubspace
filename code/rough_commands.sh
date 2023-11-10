cat `find results/ -type f | grep "test.csv" | grep -v tss` | grep -v acc | cut -f2 -d',' > temp.txt 
/trdapps/linux-x86_64/bin/nvtop
 srun -N 1 -n 1 -p qTRDGPUH --gres=gpu:1 -c 4 --mem-per-cpu=4000 -t 7200 -J DLtest -e ../out/slogs/%x-%A-%a.err -o ../out/slogs/%x-%A-%a.out -A trends53c17 --oversubscribe --mail-type=FAIL --mail-user=ibatta@gsu.edu ./JSA_DL.sh 
 cat  | grep -v acc_te | cut -f2 -d','
 cat `cat ../../in/salstats_lists/groupwise.txt` | grep groupwise | grep -E "sorted_names_m|sorted_means" > ../salient_regions/groupwise_mean_v0.csv
 find . -type f | grep csv | grep groupwise | grep ,lT1 | grep -vE "tsmax|tsmin|tsmedian|DCb|fALFF"  | sort >> ../../in/salstats_lists/groupwise.txt

for i in `seq 1 6`; do sbatch --array=110 ./JSA_DL_noGPU.sh run_DL_gica 11  $i ; done