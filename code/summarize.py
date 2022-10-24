import numpy as np 
import os, sys 
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import ttest_ind
import pandas as pd 

mpl.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'white'


def summarize_config(config_string, regvar, metric):
    """
    Computes the set of test accuracy for the string corresponding to the config_string. 
    The config strings should represent one of the directory. 
    """
    cs = config_string.split('_')
    si = cs.index('scorename')
    config_string =  '_'.join(cs[:si+1] + [regvar] + cs[(si+2):])
    print('Summarizing Updated Config: '+config_string)
    
    basedir = '../out/results/latest/'
    
    cmds = 'cat %s%s/test.csv | head -1'%(basedir,config_string) 
    metrics = os.popen(cmds).read().strip('\n').split(',')
    print(metrics)
    ind = metrics.index(metric)

    cmd = 'cat %s%s/test.csv | grep -v acc | grep -v mae_te | cut -f%d -d\",\"'%(basedir,config_string,ind+1)
    output_stream = os.popen(cmd)
    test_accs = np.array(output_stream.read().split('\n')[:-1], dtype=float)
    return test_accs


def summarize_test_results(configlist_file, output_file, regvar, metric):
    """_summary_

    Args:
        configlist_file (str): File with list of keywords to be used as config_string in the summarize_config function. summarize_config is called for each line of the file as input.
    
    Returns:

    """
    keys = ['mt','fkey']
    with open(configlist_file,'r') as f:
        config_strings = f.read().split('\n')[:-1]
    
    colnames = []
    test_accs = []
    for cs in config_strings: 
        print(cs)
        elements = cs.split('_')
        colnames.append('-'.join([elements[elements.index(key)+1] for key in keys]))
        test_accs.append(summarize_config(cs, regvar, metric))
    test_accs = np.array(test_accs).T
    
    f, ax = plt.subplots()
    ax.boxplot(test_accs, meanline=True, labels=colnames)
    
    # ax = sns.boxplot(data=test_accs)
    ax.set_xticklabels(colnames, rotation=90)
    print('Saving: '+output_file)
    plt.savefig(output_file)
    plt.close(f)



    # Plot t-vals and p-vals    
    
    T, P = [], []
    for i in range(len(colnames)):
        t, p = ttest_ind(test_accs[:,i], test_accs, axis=0)
        T.append(t); P.append(p)
    T, P = np.array(T), np.array(P)
    
    f, ax = plt.subplots()
    ax1 = sns.heatmap(P, yticklabels=colnames, xticklabels=colnames)
    pval_file = '.'.join(output_file.split('.')[:-1]) + '_pvals.' + output_file.split('.')[-1]
    plt.savefig(pval_file)
    plt.close(f)



if __name__ == "__main__":
    outdir = '../out/performances/baseline/'
    cfp = sys.argv[1]
    outfile = sys.argv[2]
    
    with open('../in/scores_ADNI.txt') as f:
        regvars = f.read().split('\n')[:-1]
    clfvars = ['labels_3way']
    metrics = {
        'reg': ['mae_te','r_te','r2_te','nrmse_te'],
        'clf': ['acc','bal_acc','precision','recall','js','rocauc']
        }
    
    if '_reg_' in cfp.split('/')[-1]:
        for regvar in regvars: 
            print(regvar)
            outfile_regvar = '.'.join(outfile.split('.')[:-1]) + '_' + regvar + '.' + outfile.split('.')[-1]
            for metric in metrics['reg']:
                print(metric)
                outfile_metric = '.'.join(outfile_regvar.split('.')[:-1]) + '_' + metric.replace('_te','') + '.' + outfile_regvar.split('.')[-1]
                summarize_test_results(cfp, outfile_metric, regvar, metric)
    elif '_clf_' in cfp.split('/')[-1]:
        for regvar in clfvars: 
            print(regvar)
            outfile_regvar = '.'.join(outfile.split('.')[:-1]) + '_' + regvar + '.' + outfile.split('.')[-1]
            for metric in metrics['clf']:
                print(metric)
                outfile_metric = '.'.join(outfile_regvar.split('.')[:-1]) + '_' + metric.replace('_te','') + '.' + outfile_regvar.split('.')[-1]
                summarize_test_results(cfp, outfile_metric, regvar, metric)



