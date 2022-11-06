import numpy as np 
import os, sys 
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import ttest_ind, ttest_1samp, zscore
import pandas as pd 
import glob
import nibabel as nib
import pickle
# from statsmodels.stats.multitest import fdrcorrection
from mne.stats import fdr_correction
from pathlib import Path
import utils as ut

mpl.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'white'






def load_masked_saliency_maps(config_string, masks, saltype, normalize_method=None, rep_wise=False):
    """
    Loads flattened saliency maps with only voxles in the masks, from all the files in the mentioned config string into a single list of length nch

    Args:
        config_string (str): keywords to be used as configuration string to search from models outputs directory
        
        
    """

    basedir = '../out/results/latest/'
    
    nch, nx, ny, nz = masks.shape
    mr = masks.reshape([nch,nx*ny*nz])
    
    out_data = [[] for ch in range(nch)]
    out_labels = []
    saldirs = glob.glob(basedir+config_string+'/')
    
    for saldir in saldirs:
        salfile = saldir + saltype+'.pkl'
        labelfile = saldir + 'test_labels.txt'
        labels = np.array(np.loadtxt(labelfile), dtype=int)
        out_labels.append(labels.squeeze())
        
        with open (salfile, 'rb') as f:
            sal_data = pickle.load(f)
        assert np.all(sal_data.shape[1:] == masks.shape)
        nsub, _, _, _, _ = sal_data.shape
        sal_data = sal_data.reshape([nsub,nch,nx*ny*nz])
        for ch in range(nch):
            out_data[ch].append(sal_data[:,ch,mr[ch]==1])
    out_labels = np.hstack(out_labels)
    
    for ch in range(nch):
        if rep_wise:
            out_data[ch] = np.array(out_data[ch])
        else:
            out_data[ch] = np.vstack(out_data[ch])
        if normalize_method!=None:
            out_data[ch] = ut.normalize_image(out_data[ch], method=normalize_method, axis=1)
        
    return out_data, out_labels



def save_masked_maps(data, masks, outfilepath, maskObj=None):
    # mask is a 4-D np array with shape (nch, nx, ny, nz)
    # data is a list (and not np.array!) of vectors, each with different length depending on non-zero voxels of the corresponding mask
    # outfilepath should be a path with .nii or .nii.gz as extension.
        
    nch, nx, ny, nz = masks.shape
    assert len(data) == nch
    mr = masks.reshape(nch, nx*ny*nz)
    
    cur_out = np.zeros([nch, nx*ny*nz])
    for ch in range(nch): 
        cur_out[ch,mr[ch]==1] = data[ch]
    
    cur_out =  np.moveaxis(cur_out.reshape([nch,nx,ny,nz]), 0, -1)
    
    if maskObj != None:
        nib.save(nib.Nifti1Image(cur_out, affine=maskObj.affine, header=maskObj.header ), outfilepath)
    else:
        nib.save(nib.Nifti1Image(cur_out, np.eye(4) ), outfilepath)
    
    
    


def run_tstats(data, data2=None, alternateH='two-sided', standardize=True):
    # Assumes (nSamp, nFeatures) as the dim (2-dimensional)
    out_stats = ['tvals','minusTlogP_uncorrected','pvals_uncorrected','pvals_fdrCorrected','fdr_rejectedHypothesis','minusTlogP_fdrCorrected', 'minusSignTlogP_fdrCorrected']
    out_data = {key:None for key in out_stats}
    
    if standardize:
        data = zscore(data, axis=1)
    if np.all(data2 != None):
        if standardize:
            data2 = zscore(data2, axis=1)
        assert np.all(data.shape[1:] == data2.shape[1:])
        tvals, pvals_uncorrected = ttest_ind(data, data2, axis=0, equal_var=False)
    else:
        tvals, pvals_uncorrected = ttest_1samp(data, 0, axis=0)#, alternative=alternateH)
    minusTlogP_uncorrected = (-tvals) * np.log10(pvals_uncorrected)    
    fdr_rejectedHypothesis, pvals_fdrCorrected = fdr_correction(pvals_uncorrected)
    minusSignTlogP_fdrCorrected = -(np.sign(tvals))*np.log10(pvals_fdrCorrected)
    minusTlogP_fdrCorrected = (-tvals) * np.log10(pvals_fdrCorrected)
    for key in out_stats:
        out_data[key] = eval(key)
    
    return out_data
    




def summarize_saliency(config_string, regvar, outfile, saltype='sal_raw', mode='single', rep=-1):
    """_summary_
    Computes significance of saliency differently for clf (one-sided on abs) and reg (two-sided without abs)

    Args:
        config_string (str): keywords to be used as configuration string to search from models outputs directory
    
        
    """
    
    basedir = '../out/results/latest/'
    ## Ensure abs for clf, and not for reg
    config_string = impute_scorename(config_string, regvar)
    
    cs = config_string.split('_')
    nc = int(cs[cs.index('nc')+1])
    
    if rep >= 0:
        config_string = config_string.replace('rep_*','rep_%d'%rep)
        assert len(glob.glob(basedir + config_string)) == 1
    
    # Load masks for all channels
    cfgpath = glob.glob(basedir + config_string)[0] + '/config.pkl' 
    cfg = ut.loadCfg(cfgpath)
    masks = ut.loadMasks(cfg)
    nch, nx, ny, nz = masks.shape
    mask_objects = ut.loadMasks(cfg, nibObjectsOnly=True)
    
    # Load flattened data with only mask voxels
    data, labels = load_masked_saliency_maps(config_string, masks, saltype)
    
    
    
    # Compute stats
    tstats = []
    for ch in range(nch):
        if mode == 'single':
            tstats.append(run_tstats(data[ch], alternateH='two-sided'))
        elif mode == 'groupwise':
            tstats.append(run_tstats(data[ch][labels==0,:], data2=data[ch][labels==1,:], alternateH='two-sided'))
            
    
    # Save all results
    list_stats = tstats[0].keys()
    
    for curstat in list_stats:
    
        cur_out_data = [tstats[ch][curstat] for ch in range(nch)]
        
        head, tail = os.path.split(outfile)
        Path(head).mkdir(parents=True, exist_ok=True)
        assert ('.nii' in tail)
        cur_outfile = head + '/' + tail.replace('.nii','_%s_%s_%s.nii'%(saltype, mode, curstat))
        
        save_masked_maps(cur_out_data, masks, cur_outfile, maskObj=mask_objects[0])
        
    
    
    

        


# def compute_stats(data, masks, outfile, data2=None, alternateH='two-sided'):
#     # Accepts only .nii or .nii.gz as extensions in outfile
#     nsub, nch, nx, ny, nz = data.shape
#     mr = masks.reshape([nch,nx*ny*nz])
#     flat_data = data.reshape([nsub, nch, nx*ny*nz])
#     out_stats = ['tvals','minusTlogP_uncorrected','pvals_uncorrected','pvals_fdrCorrected','fdr_rejectedHypothesis','minusTlogP_fdrCorrected']
#     out_data = {key:[] for key in out_stats}
    
#     print('Performing t-stats..')    
#     if data2!=None:
#         mode = 'group'
#         assert np.all(data.shape[1:] == data2.shape[1:])
#         nsub2 = data2.shape[0]
#         flat_data2 = data2.reshape([nsub2, nch, nx*ny*nz])
#         for ch in range(nch):
#             ttd = flat_data[:,ch,mr[ch]==1] # Consider only brain voxels for statistical testing
#             ttd2 = flat_data2[:,ch,mr[ch]==1] # Consider only brain voxels for statistical testing
#             tvals, pvals_uncorrected = ttest_ind(ttd, ttd2, axis=0, equal_var=False)
#             fdr_rejectedHypothesis, pvals_fdrCorrected = fdr_correction(pvals_uncorrected)
#             minusTlogP_fdrCorrected = (-tvals) * np.log10(pvals_fdrCorrected)
#             for key in out_stats:
#                 out_data[key].append(eval(key))
#     else:
#         mode = ''
#         # 1 sample t-test
#         for ch in range(nch):
#             ttd = flat_data[:,ch,mr==1] # Consider only brain voxels for statistical testing
#             tvals, pvals_uncorrected = ttest_1samp(ttd, 0, axis=0, alternative=alternateH)
#             minusTlogP_uncorrected = (-tvals) * np.log10(pvals_uncorrected)
#             fdr_rejectedHypothesis, pvals_fdrCorrected = fdr_correction(pvals_uncorrected)
#             minusTlogP_fdrCorrected = (-tvals) * np.log10(pvals_fdrCorrected)
#             for key in out_stats:
#                 out_data[key].append(eval(key))
    
#     print('Saving t-stats..')
#     for key in out_stats:
#         out_data[key] = np.array(out_data[key])
#         cur_out_data = np.zeros([nch,nx*ny*nz])
#         for ch in nch:
#             cur_out_data[ch,mr[ch]==1] = out_data[key][ch,:]
#         cur_out_data = np.moveaxis(cur_out_data.reshape([nch,nx,ny,nz]), 0, -1)
        
#         head, tail = os.path.split(outfile)
#         Path(head).mkdir(parents=True, exist_ok=True)
#         assert '.nii' in tail
#         cur_outfile = head + '/' + tail.replace('.nii','_%s.nii'%key)
#         ref_nii = nib.load('/data/users2/ibatta/data/features/lowresSMRI/ADNI/random_subject.nii.gz')
#         nib.save(nib.Nifti1Image(cur_out_data, affine=ref_nii.affine, header=ref_nii.header), cur_outfile)        
            
            
               
            
            
            

    
    
    
def impute_scorename(config_string, regvar):
    cs = config_string.split('_')
    si = cs.index('scorename')
    imputed_config_string =  '_'.join(cs[:si+1] + [regvar] + cs[(si+2):])
    return imputed_config_string

def summarize_config(config_string, regvar, metric):
    """
    Computes the set of test accuracy for the string corresponding to the config_string. 
    The config strings should represent one of the directory. 
    """
    config_string = impute_scorename(config_string, regvar)
    print('Summarizing Updated Config: '+config_string)
    
    basedir = '../out/results/'
    
    cmds = 'cat %s%s/test.csv | head -1'%(basedir,config_string) 
    metrics = os.popen(cmds).read().strip('\n').split(',')
    print(metrics)
    ind = metrics.index(metric)

    cmd = 'cat %s%s/test.csv | grep -v acc | grep -v mae_te | cut -f%d -d\",\"'%(basedir,config_string,ind+1)
    output_stream = os.popen(cmd)
    test_accs = np.array(output_stream.read().split('\n')[:-1], dtype=float)
    return test_accs

def summarize_all_saliencies(configlist_file, regvar, cfg_ind, rep=-1):
    # sals = ['sal','fsal','imsal','fimsal']
    # saltypes = [si+'_raw' for si in sals] + [si+'_zscore' for si in sals]
    saltypes = ['fsal_raw']
    with open(configlist_file,'r') as f:
        config_strings = f.read().split('\n')[:-1]
    
    cs = config_strings[cfg_ind]
    
    # for cs in config_strings:
    print(cs)
    if rep == -1:
        outfile = '../out/saliency_stats/' + cs.replace('*','X') + '/tstats.nii.gz'
    else: 
        outfile = '../out/saliency_stats/' + cs.replace('*','X') + '/repwise/tstats_%d.nii.gz'%rep
    for saltype in saltypes: 
        print(saltype)
        summarize_saliency(cs, regvar, outfile, saltype=saltype, mode='single', rep=rep)
        summarize_saliency(cs, regvar, outfile, saltype=saltype, mode='groupwise', rep=rep)
        
        
    

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
    ax1 = sns.heatmap(-np.sign(T)*np.log10(P), yticklabels=colnames, xticklabels=colnames)
    pval_file = '.'.join(output_file.split('.')[:-1]) + '_pvals.' + output_file.split('.')[-1]
    plt.savefig(pval_file)
    plt.close(f)



if __name__ == "__main__":
    outdir = '../out/performances/baseline/'
    cfp = sys.argv[1]
    outfile = sys.argv[2]
    
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        cfg_ind = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        cfg_ind = 10
        
    with open('../in/scores_ADNI.txt') as f:
        regvars = f.read().split('\n')[:-1]
    clfvars = ['labels_3way']
    metrics = {
        'reg': ['mae_te','r_te','r2_te','nrmse_te'],
        'clf': ['acc','bal_acc','precision','recall','js','rocauc']
        }
    
    
    
    if '_reg_' in cfp.split('/')[-1]:
        # for regvar in regvars: 
        for regvar in ['age']:
            print(regvar)
            outfile_regvar = '.'.join(outfile.split('.')[:-1]) + '_' + regvar + '.' + outfile.split('.')[-1]
            for metric in metrics['reg']:
                print(metric)
                outfile_metric = '.'.join(outfile_regvar.split('.')[:-1]) + '_' + metric.replace('_te','') + '.' + outfile_regvar.split('.')[-1]
                summarize_test_results(cfp, outfile_metric, regvar, metric)
    elif '_clf_' in cfp.split('/')[-1]:
        for regvar in clfvars: 
            print(regvar)
            
            for rep in range(-1,10):
                print('rep=%d'%rep)
                summarize_all_saliencies(cfp, regvar, cfg_ind, rep=rep)
            sys.exit(0)
            outfile_regvar = '.'.join(outfile.split('.')[:-1]) + '_' + regvar + '.' + outfile.split('.')[-1]
            for metric in metrics['clf']:
                print(metric)
                outfile_metric = '.'.join(outfile_regvar.split('.')[:-1]) + '_' + metric.replace('_te','') + '.' + outfile_regvar.split('.')[-1]
                summarize_test_results(cfp, outfile_metric, regvar, metric)



