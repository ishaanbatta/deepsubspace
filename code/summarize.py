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
plt.rcParams.update({'font.size': 16})





def load_masked_saliency_maps(config_string, masks, saltype, normalize_method=None, rep_wise=False):
    """
    Loads flattened saliency maps with only voxles in the masks, from all the files in the mentioned config string into a single list of length nch

    Args:
        config_string (str): keywords to be used as configuration string to search from models outputs directory
        
        
    """

    basedir = '../out/results/'
    
    nch, nx, ny, nz = masks.shape
    mr = masks.reshape([nch,nx*ny*nz])
    
    out_data = [[] for ch in range(nch)]
    out_labels = []
    saldirs = glob.glob(basedir+config_string+'/')
    
    for saldir in saldirs:
        salfile = saldir + saltype+'.pkl'
        labelfile = saldir + 'labels_te.txt'
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
    
    
    


def run_tstats(data, data2=None, alternateH='two-sided', standardize=True, fdr_alpha=0.05):
    # Assumes (nSamp, nFeatures) as the dim (2-dimensional)
    # out_stats = ['tvals','minusTlogP_uncorrected','pvals_uncorrected','pvals_fdrCorrected','fdr_rejectedHypothesis','minusTlogP_fdrCorrected', 'minusSignTlogP_fdrCorrected']
    out_stats = ['fdr_rejectedHypothesis', 'minusSignTlogP_fdrCorrected']
    out_data = {key:None for key in out_stats}
    
    if standardize:
        data = zscore(data, axis=1)
    if np.all(data2 != None):
        if standardize:
            data2 = zscore(data2, axis=1)
        assert np.all(data.shape[1:] == data2.shape[1:])
        tvals, pvals_uncorrected = ttest_ind(data, data2, axis=0, equal_var=False)
    else:
        tvals, pvals_uncorrected = ttest_1samp(data, 0, axis=0, alternative=alternateH)
    minusTlogP_uncorrected = (-tvals) * np.log10(pvals_uncorrected)    
    fdr_rejectedHypothesis, pvals_fdrCorrected = fdr_correction(pvals_uncorrected, alpha=fdr_alpha)
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
    
    basedir = '../out/results/'
    ## Ensure abs for clf, and not for reg
    
    
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
            
        elif mode in ['univsmul']:
            # Assert bimodal arch and compare for both cases
            if nch >= 2:
                #corresponding unimodal data to ch
                curmod = cfg.fkey.split(',')[ch]
                config2 = config_string.replace(cfg.fkey,curmod)
                curmasks = np.expand_dims(masks[ch],0)
                data2, _ = load_masked_saliency_maps(config2, curmasks,saltype)
                # Perform t-test between uni vs bi modal paradigm for modality ch
                tstats.append(run_tstats(data[ch], data2=data2[0], alternateH='two-sided'))
            else:
                print('Unimodal config strings are not supported in %s mode. They will already be compared under multimodal cases. Exiting..'%mode)
                return
                    
        elif mode == 'intrabim':
            # Assert bimodal arch and save as single channel result
            if nch == 2:
                if ch == 0:
                    tstats.append(run_tstats(data[0], data2=data[1], alternateH='two-sided'))                    
                else:
                    print('Skipping redundant iterations for %s mode.'%mode)
                    break
            else: 
                print('Only bi-modal config strings are supported in %s mode. Exiting..'%mode)
                return
        elif mode == 'meansal':
            tstats.append({'meansal':data[ch].mean(axis=0)})
        elif mode == 'stdsal':
            tstats.append({'stdsal':data[ch].std(axis=0)})
    
    # Save all results
    list_stats = tstats[0].keys()
    
    for curstat in list_stats:
    
        nstats = nch 
        if mode == 'intrabim':
            nstats = 1 
        cur_out_data = [tstats[ch][curstat] for ch in range(nstats)]
        
        head, tail = os.path.split(outfile)
        Path(head).mkdir(parents=True, exist_ok=True)
        assert ('.nii' in tail)
        cur_outfile = head + '/' + tail.replace('.nii','_%s_%s_%s.nii'%(saltype, mode, curstat))
        
        save_masked_maps(cur_out_data, masks[:nstats], cur_outfile, maskObj=mask_objects[0])
        
    
    
    

        


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
    cs = config_string.split('/')[-1].split('_')
    si = cs.index('scorename')
    imputed_config_string =  '_'.join(cs[:si+1] + [regvar] + cs[(si+2):])
    return  '/'.join(config_string.split('/')[:-1] +  [imputed_config_string] )

def summarize_config(config_string, regvar, metric):
    """
    Computes the set of test accuracy for the string corresponding to the config_string. 
    The config strings should represent one of the directory. 
    """
    config_string = impute_scorename(config_string, regvar)
    print('Summarizing Updated Config: '+config_string)
    
    basedir = '../out/results/'
    
    cmds = 'cat %s%s/test.csv | head -1'%(basedir,config_string) 
    print(cmds)
    metrics = os.popen(cmds).read().strip('\n').split(',')
    print(metrics)
    ind = metrics.index(metric)

    cmd = 'cat %s%s/test.csv | grep -v acc | grep -v mae_te | cut -f%d -d\",\"'%(basedir,config_string,ind+1)
    print(cmd)
    output_stream = os.popen(cmd)
    test_accs = np.array(output_stream.read().split('\n')[:-1], dtype=float)
    return test_accs

def summarize_all_saliencies(configlist_file, regvar, cfg_ind, rep=-1):
    # sals = ['sal','fsal','imsal','fimsal']
    # saltypes = [si+'_raw' for si in sals] + [si+'_zscore' for si in sals]
    saltypes = ['fsal_te']
    # modes = ['single','groupwise','univsmul','intrabim']
    modes = ['meansal','stdsal','single','groupwise']
    # modes = ['single']
    with open(configlist_file,'r') as f:
        config_strings = f.read().split('\n')[:-1]
    
    cs = config_strings[cfg_ind]
    cs = impute_scorename(cs, regvar)
    
    # for cs in config_strings:
    print(cs)
    if rep == -1:
        outfile = '../out/saliency_stats/' + cs.replace('*','X') + '/tstats.nii.gz'
    else: 
        outfile = '../out/saliency_stats/' + cs.replace('*','X') + '/repwise/tstats_%d.nii.gz'%rep
    for saltype in saltypes: 
        print(saltype)
        for curmode in modes:
            print(curmode)
            summarize_saliency(cs, regvar, outfile, saltype=saltype, mode=curmode, rep=rep)


def summarize_test_results(configlist_file, output_file, regvar, metric):
    """_summary_

    Args:
        configlist_file (str): File with list of keywords to be used as config_string in the summarize_config function. summarize_config is called for each line of the file as input.
    
    Returns:

    """
    # keys = ['mt','fkey','bs','lr','sf']
    keys = ['mt','fkey','bs','lr']
    with open(configlist_file,'r') as f:
        config_strings = f.read().split('\n')[:-1]
    
    colnames = []
    test_accs = []
    for cs in config_strings: 
        print(cs)
        elements = cs.split('/')[-1].split('_')
        print(elements)
        prefix = '.'
        if len(cs.split('/')) > 1:
            prefix = cs.split('/')[-2]
        colstring = '-'.join([prefix] + [elements[elements.index(key)+1] for key in keys])
        colnames.append(colstring)
        test_accs.append(summarize_config(cs, regvar, metric))
    test_accs = np.array(test_accs).T
    
    f, ax = plt.subplots()
    ax.boxplot(test_accs, labels=colnames, showmeans=False, widths=0.3)
    xcoords = 1+np.array(range(test_accs.shape[1]))
    ycoords = test_accs.mean(axis=0)
    ax.scatter(xcoords, ycoords, c='g')
    for i in range(test_accs.shape[1]):
        # ax.annotate('.%d'%int(ycoords[i]*100), (xcoords[i]-0.2,1-0.03), fontsize=16, color='gray')
        ax.annotate('%.2f'%ycoords[i], (xcoords[i]-0.5,ycoords[i]), fontsize=16, color='gray', rotation=90)
    
    # ax = sns.boxplot(data=test_accs)
    
    # if metric == 'acc':
    #     ax.set_ylim([0.6,1])
    #     colnames = [ci.split('-')[-1].replace('KccReHo','ReHo') for ci in colnames if len(ci.split(','))<=2 ] + ['all' for ci in colnames if len(ci.split(',')) > 2 ]
    #     print(colnames)
    #     print(test_accs.std(axis=0))
    #     print(test_accs.mean(axis=0))
        
    ax.set_ylabel(metric)        
    ax.set_xticklabels(colnames, rotation=90)
    print('Saving: '+output_file)
    f.set_size_inches(len(colnames)/4.0, 4.8)
    plt.savefig(output_file)
    plt.close(f)



    # Plot t-vals and p-vals    
    
    T, P = [], []
    for i in range(len(colnames)):
        t, p = ttest_ind(test_accs[:,i], test_accs, axis=0)
        T.append(t); P.append(p)
    T, P = np.array(T), np.array(P)
    
    f, ax = plt.subplots()
    ax1 = sns.heatmap(-np.sign(T)*np.log10(P), yticklabels=colnames, xticklabels=colnames, annot=True)
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
        cfg_ind = int(sys.argv[3])
        
    with open('../in/scores_BSNIP.txt') as f:
        regvars = f.read().split('\n')[:-1]
    clfvars = ['labelsBP']
    metrics = {
        'reg': ['mae_te','r2_te','nrmse_te','r_te'],
        'clf': ['acc','bal_acc','precision','recall','js','rocauc']
        # 'clf': ['acc']
        }
    
    
    
    if '_reg_' in cfp.split('/')[-1]:
        # for regvar in regvars: 
        for regvar in regvars:
            print(regvar)
            
            # for rep in range(-1,10):
            #     print('rep=%d'%rep)
            #     summarize_all_saliencies(cfp, regvar, cfg_ind, rep=rep)
            # sys.exit(0)
            
            outfile_regvar = '.'.join(outfile.split('.')[:-1]) + '_' + regvar + '.' + outfile.split('.')[-1]
            for metric in metrics['reg']:
                print(metric)
                outfile_metric = '.'.join(outfile_regvar.split('.')[:-1]) + '_' + metric.replace('_te','') + '.' + outfile_regvar.split('.')[-1]
                summarize_test_results(cfp, outfile_metric, regvar, metric)
    elif '_clf_' in cfp.split('/')[-1]:
        for regvar in clfvars: 
            print(regvar)
            
            
            # for rep in range(-1,10):
            #     print('rep=%d'%rep)
            #     summarize_all_saliencies(cfp, regvar, cfg_ind, rep=rep)
            # sys.exit(0)
            
            outfile_regvar = '.'.join(outfile.split('.')[:-1]) + '_' + regvar + '.' + outfile.split('.')[-1]
            for metric in metrics['clf']:
                print(metric)
                outfile_metric = '.'.join(outfile_regvar.split('.')[:-1]) + '_' + metric.replace('_te','') + '.' + outfile_regvar.split('.')[-1]
                summarize_test_results(cfp, outfile_metric, regvar, metric)



