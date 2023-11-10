import utils as ut
import pickle 
import numpy as np 
import nibabel as nib 
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import zscore 

import sys
sys.path.insert(-1,'../../subspace/code/')
from learning import compute_predictions_tuned, scale_data

from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, auc, precision_score, recall_score
from scipy.stats import pearsonr

from nilearn.plotting import plot_stat_map
from nilearn import image

import os






def summarize_gica(cfg, md, varname, nc_pca=100, nc_ica=10):
    
    basedir = cfg.ml
    targetvar = cfg.scorename
    nreps = cfg.nReps
    
    
    outdir =  basedir + '/groupICA/pca_%d_ica_%d/'%(nc_pca, nc_ica)
    

    ## Load target variables and class labels
    print('Loading class variables..')

    labels = {'age':[],'MMSE':[],'dx':[]}
    cls = []

    for i in range(nreps):
        curbasedir = basedir.replace('iter_110','iter_11%d'%i).replace('rep_0','rep_%d'%i)
        fname = curbasedir + 'labels_%s.txt'%md
        labels[targetvar].append(np.loadtxt(fname, delimiter=','))
        cfg = ut.loadCfg(curbasedir + 'config_test.pkl')
        df = ut.readFramesSeparately(cfg.sm, cfg.ssd, md, i)
        cls.append(df['labels3way'])
        
    labels[targetvar] = np.hstack(labels[targetvar])
    cls = np.hstack(cls).squeeze()



    # Load computed gica results
    print('Loading computed gICA results..')

    gpca_comp_fname = outdir  + '%s_%s_groupPCAEigenVectors.pkl'%(varname,md)
    with open(gpca_comp_fname, 'rb') as f:
        gpca_components = pickle.load(f)
    gpca_loadings_fname = outdir + '%s_%s_groupPCAloadings.pkl'%(varname,md)
    with open(gpca_loadings_fname, 'rb') as f:
        gpca_loadings = pickle.load(f)
    gpca_evr_fname = outdir + '%s_%s_groupPCAExplainedVarianceRatio.pkl'%(varname,md)
    with open(gpca_evr_fname, 'rb') as f:
        gpca_evr = pickle.load(f)



    # gica_comp_fname = basedir + '%s_%s_groupICAcomponents.pkl'%(varname,md)
    # with open(gica_comp_fname, 'rb') as f:
    #     gica_components = pickle.load(f)
    gica_loadings_fname = outdir + '%s_%s_groupICAloadings.pkl'%(varname,md)
    with open(gica_loadings_fname, 'rb') as f:
        gica_loadings = pickle.load(f)
        
    with open(outdir + '%s_%s_groupICAreconComponents.pkl'%(varname,md),'rb') as f:
        recon_components = pickle.load(f)

    # Save Explained variance for PCA 
    print('Saving explained variance plots for PCA..')
    vthresh = 0.9
    xdata = np.arange(1,len(gpca_evr)+1)
    ydata = np.cumsum(gpca_evr) / gpca_evr.sum()
    f, ax = plt.subplots()
    plt.plot(xdata, ydata, c='blue', label='EVR-CDF')
    plt.plot(xdata, [vthresh for i in xdata], c='r', linestyle='--', label='vthresh')
    plt.title('Total EVR=%.3f'%gpca_evr.sum(), c='gray')
    plt.legend(loc='lower right')
    plt.savefig(outdir + '%s_%s_groupPCAExplainedVarianceRatioCDF.png'%(varname,md))
    plt.close(f)

    # Z-score of results 
    print('Standardizing results..')
    recon_components = zscore(recon_components, axis=0)
    gpca_components = zscore(gpca_components, axis=0)
    gica_loadings = zscore(gica_loadings, axis=1)
    gpca_loadings = zscore(gpca_loadings, axis=1)
    
    
    # Prediction using loadings
    print('Doing Prediction with loadings..')
    clf_scores = {i:i for i in ['accuracy_score', 'auc', 'precision_score', 'recall_score']}
    reg_scores = {i:i for i in ['neg_mean_absolute_error','r2']}
    if (int(cfg.nc) == 1):
        params =ParameterGrid({'alpha':[[0.0001,0.001,0.01,0.1,1,10,20,50,75,100,150,200,500,1000]]})#, 'gamma':[[0.1,1,10]]})#, 'kernel': [[ 'rbf']] })
        estimator = Ridge()
        scoring = reg_scores
    else:
        params =ParameterGrid({'C':[[0.0001,0.001,0.01,0.1,1,10,20,50,75,100,150,200,500,1000]]})#, 'gamma':[[0.1,1,10]], 'kernel': [[ 'rbf']] })
        estimator = LogisticRegression()
        scoring = clf_scores
    
    gsresult_ica = GridSearchCV(estimator=estimator, n_jobs=16, param_grid=params, cv=10, scoring=scoring, refit='neg_mean_absolute_error').fit(gica_loadings.T, labels[targetvar])
    gsresult_pca = GridSearchCV(estimator=estimator, n_jobs=16, param_grid=params, cv=10, scoring=scoring, refit='neg_mean_absolute_error').fit(gpca_loadings.T, labels[targetvar])
    
    print('Saving results for prediction with loadings..')
    with open(outdir + '%s_%s_loadingPredictions.pkl'%(varname,md), 'wb') as fpred:
        pickle.dump({'predictions_ica':gsresult_ica, 'predictions_pca':gsresult_pca}, fpred)
    fpred.close()
    coeff_pca = gsresult_pca.best_estimator_.coef_
    coeff_ica = gsresult_ica.best_estimator_.coef_
    f, ax = plt.subplots(nrows=2, ncols=1)
    bl0 = [str(i) for i in range(len(coeff_pca))]
    ax[0].bar(bl0, coeff_pca)
    ax[0].set_ylabel('PCA')
    bl1 = [str(i) for i in range(len(coeff_ica))]
    ax[1].bar(bl1, coeff_ica)
    ax[1].set_ylabel('ICA')
    ax[1].set_xlabel('component index')
    plt.savefig(outdir + '%s_%s_loadingPredictionCoeffs.png'%(varname,md))
    plt.close(f)
    

    # Correlation of target vars
    print('Performing correlation analysis with target variables..')
    comp_corrs = np.corrcoef(labels[targetvar].squeeze(), gica_loadings)
    label_corrs = comp_corrs[0][1:] # Skipping the first entry because that's self-correlation of labels (r=1)
    np.savetxt(outdir + '%s_%s_loadingsCorr.csv'%(varname,md), label_corrs)
    



    besti = np.abs(label_corrs).argsort()[-1]

    ### Group ICA Results Summary on Saliency Maps
    
    f, ax = plt.subplots()
    barlabels = ['%d'%(i+1) for i in range(len(label_corrs))]
    bar = ax.bar(barlabels, label_corrs, label=label_corrs)
    ax.bar_label(bar, labels=['%.2f'%li for li in  label_corrs], fontsize=9)
    ax.set_xticklabels(barlabels, rotation=90)
    plt.title('Correlation between gICA component loadings and target label (%s)'%targetvar + '\n(pca_%d_ica_%d; %s_%s)'%(nc_pca, nc_ica, varname, md))
    plt.xlabel('gICA components')
    plt.ylabel('Correlation with %s'%targetvar)
    # plt.show()
    plt.savefig(outdir + '%s_%s_loadingsCorr.png'%(varname,md))
    plt.close()


    ## PLot best correlation

    # plt.scatter(labels[targetvar].squeeze(), gica_loadings[besti,:])
    f, ax = plt.subplots()
    sns.regplot(gica_loadings[besti,:], labels[targetvar].squeeze(), scatter_kws={'color':'gray'},line_kws={'color':'blue'}, robust=False)
    plt.xlabel('Loadings from component %d'%(besti+1))
    plt.ylabel(targetvar)
    plt.title('Plot for component loadings with highest correlation with target variable')
    plt.savefig(outdir + '%s_%s_bestComponentCorr.png'%(varname,md))
    plt.close()



    # ### Group Comparison of loadings
    # print('Perfroming group comparison of gICA loadings..')

    # ## Group comparison of gPCA loadings
    # f, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,4), sharex=True)
    # f.suptitle('Group comparison of gPCA loadings')
    # for ci in range(gica_loadings.shape[0]):
        
    #     ax[ci // 5, ci % 5].boxplot([gpca_loadings[ci,cls==1], gpca_loadings[ci,cls==2], gpca_loadings[ci,cls==3]], labels=['CN','AD','MCI'])
    #     ax[ci // 5, ci % 5].set_title('Component %d'%(ci+1))
        

    # plt.savefig(outdir + '%s_%s_gPCAgroupComparison.png'%(varname,md))
    # plt.close()


    # ## Group comparison of gICA loadings
    # f, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,4), sharex=True)
    # f.suptitle('Group comparison of gICA loadings')
    # for ci in range(gica_loadings.shape[0]):
        
    #     ax[ci // 5, ci % 5].boxplot([gica_loadings[ci,cls==1], gica_loadings[ci,cls==2], gica_loadings[ci,cls==3]], labels=['CN','AD','MCI'])
    #     ax[ci // 5, ci % 5].set_title('Component %d'%(ci+1))
        

    # plt.savefig(outdir + '%s_%s_gICAgroupComparison.png'%(varname,md))
    # plt.close()




    ### Save Group ICA maps 
    print('Saving gICA component maps as nii..')

    nsub, nc = recon_components.shape

    cfg = ut.loadCfg(basedir + 'config.pkl')
    masks = ut.loadMasks(cfg)
    maskObj = ut.loadMasks(cfg, nibObjectsOnly=True)
    nch, nx, ny, nz = masks.shape

    mr = masks.reshape(nch, nx*ny*nz)
    curout = np.zeros([nc, nx*ny*nz])
    for ci in range(nc):
        curout[ci,mr[0]==1] = recon_components[:,ci]

    curout =  np.moveaxis(curout.reshape([nc,nx,ny,nz]), 0, -1)

    gica_outfile = outdir + '%s_%s_groupICAreconComponents_zscored.nii'%(varname,md)
    gica_nimg = nib.Nifti1Image(curout, affine=maskObj[0].affine, header=maskObj[0].header )
    nib.save(gica_nimg, gica_outfile)


    

    ### Plot Group ICA maps 
    print('Plotting gICA maps..')




    gica_file = outdir + '%s_%s_groupICAreconComponents_zscored.nii'%(varname,md)
    gica_img = image.load_img(nib.load(gica_file))
    
    nx, ny, nz, nvol = gica_img.shape
    rowcoldict = {10:(4,3), 30:(6,5), 100:(10,10), 5:(3,2)}
    nrows, ncols = rowcoldict[nvol]
    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*4,ncols*4), facecolor='black')
    
    axcover = np.zeros([nrows,ncols], dtype=bool)

    for vol in range(nvol):
        rw, col = vol // ncols, vol % ncols
        # if vol == 9:
        #     col += 1

        plot_stat_map(image.index_img(gica_img, vol), threshold=1, black_bg=True, title='C%d'%(vol+1), figure=f, axes=ax[rw,col])
        axcover[rw,col] = True
        
    # ax[-1,0].axis('off')
    # ax[-1,-1].axis('off')
    for rw,col in zip(np.where(axcover==False)[0], np.where(axcover==False)[1]):
        ax[rw,col].axis('off')
        ax[rw,col].axis('off')

    f.suptitle('Components from gICA on saliency maps (pca_%d_ica_%d; %s_%s)'%(nc_pca, nc_ica, varname, md), color='white')
    plt.savefig(os.path.splitext(gica_file)[0] + '_ortho.png')
    plt.close(f)
    
    
    





    ### Save Group PCA maps 
    print('Saving gPCA component maps as nii..')

    nsub, nc = recon_components.shape

    cfg = ut.loadCfg(basedir + 'config.pkl')
    masks = ut.loadMasks(cfg)
    maskObj = ut.loadMasks(cfg, nibObjectsOnly=True)
    nch, nx, ny, nz = masks.shape

    mr = masks.reshape(nch, nx*ny*nz)
    curout = np.zeros([nc, nx*ny*nz])
    for ci in range(nc):
        curout[ci,mr[0]==1] = gpca_components[:,ci]

    curout =  np.moveaxis(curout.reshape([nc,nx,ny,nz]), 0, -1)

    gpca_outfile = outdir + '%s_%s_groupPCAcomponentsTop%d_zscored.nii'%(varname,md,nc)
    gpca_nimg = nib.Nifti1Image(curout, affine=maskObj[0].affine, header=maskObj[0].header )
    nib.save(gpca_nimg, gpca_outfile)


    

    ### Plot Group PCA maps 
    print('Plotting gPCA maps..')

    gpca_file = outdir + '%s_%s_groupPCAcomponentsTop%d_zscored.nii'%(varname,md,nc)
    gpca_img = image.load_img(nib.load(gpca_file))
    
    nx, ny, nz, nvol = gpca_img.shape
    rowcoldict = {10:(4,3), 30:(6,5), 100:(10,10), 5:(3,2)}
    nrows, ncols = rowcoldict[nvol]
    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*4,ncols*4), facecolor='black')
    
    axcover = np.zeros([nrows,ncols], dtype=bool)

    for vol in range(nvol):
        rw, col = vol // ncols, vol % ncols
        # if vol == 9:
        #     col += 1

        plot_stat_map(image.index_img(gpca_img, vol), threshold=1, black_bg=True, title='C%d'%(vol+1), figure=f, axes=ax[rw,col])
        axcover[rw,col] = True
        
    # ax[-1,0].axis('off')
    # ax[-1,-1].axis('off')
    for rw,col in zip(np.where(axcover==False)[0], np.where(axcover==False)[1]):
        ax[rw,col].axis('off')
        ax[rw,col].axis('off')

    f.suptitle('Components from gPCA on saliency maps (pca_%d_ica_%d; %s_%s)'%(nc_pca, nc_ica, varname, md), color='white')
    plt.savefig(os.path.splitext(gpca_file)[0] + '_ortho.png')
    plt.close(f)
    
    