from sklearn.model_selection import train_test_split
import os, sys
import numpy as np
import pdb
import argparse

n = int(sys.argv[1]) # total number of samples
k_te = int(sys.argv[2]) # number of splits for train-test split 
k_va = int(sys.argv[3]) # number of splits for train-val split (within train part of train-test split)
nreps = int(sys.argv[4]) # number of repetitions
labelfile = sys.argv[5] # label file for stratification
outdir = sys.argv[6] # output file path (will be appended with appropriate suffixes creating multiple files)

all_labels = np.loadtxt(labelfile, dtype=int)

# selelct specific labels if needed for binary classification
select_labels = [1,2,3]
labels = np.array([li for li in all_labels if li in select_labels], dtype=int)
sli = np.array([i for i in range(len(all_labels)) if all_labels[i] in select_labels], dtype=int)


for ri in np.arange(nreps):
	tri, test_indices = train_test_split(np.arange(len(labels)), test_size=1.0/k_te, shuffle=True, stratify=labels, random_state=108+ri)
	tr_idx, val_idx = train_test_split(np.arange(len(tri)), test_size=1/k_va, shuffle=True, stratify=labels[tri], random_state=108+ri)
	train_indices, val_indices = tri[tr_idx], tri[val_idx]

	np.savetxt(outdir+'/tr_r'+str(ri)+'.csv', sli[train_indices], fmt='%d')
	np.savetxt(outdir+'/va_r'+str(ri)+'.csv', sli[val_indices], fmt='%d')
	np.savetxt(outdir+'/te_r'+str(ri)+'.csv', sli[test_indices], fmt='%d')


