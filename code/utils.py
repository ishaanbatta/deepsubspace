import os, json, sys
import pickle
import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import nibabel as nib
from models import AlexNet3D_Dropout, AN3Ddr_highresMax, AN3Ddr_lowresAvg, AN3Ddr_lowresMax, AN3Ddr_lowresMaxLight
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, jaccard_score, roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix
from dataclasses import dataclass
from scipy.stats import pearsonr, zscore
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field, asdict

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air.checkpoint import Checkpoint



@dataclass
class Config:
    iter: int = 0  # slurmTaskIDMapper maps this variable using tr_smp_sizes and nReps to tss and rep
    tr_smp_sizes: tuple = (100, 200, 500, 1000, 2000, 5000, 10000)
    nReps: int = 20
    nc: int = 10
    bs: int = 16
    lr: float = 0.001
    es: int = 1
    pp: int = 1
    es_va: int = 1
    es_pat: int = 40
    # fl: str = '/data/users2/ibatta/projects/deepsubspace/in/filelist.csv' # master file with list of paths to input scans for all subjects (should have header ideally for supporting multimodal)
    ds: str = 'ADNI' #Name of the dataset being used.
    ml: str = '/data/users2/ibatta/projects/deepsubspace/out/modelout/'
    mt: str = 'AlexNet3D_Dropout_'
    ssd: str = '../../SampleSplits/'
    sm: str = '/data/users2/ibatta/projects/deepsubspace/in/analysis_score.csv' # master file for scores, labels and relative filepaths for all subjects
    fkey: str = 'smriPath' # Name of filepath key(s) to be used for the set of features being used. Must be a comma separated string listing the features, if a multimodal architecture is used.
    nch: int = 1 # Number of input channles (modalities)
    ngr: int = 1 # Number of input groups (modalities/subspaces)
    fmpfile: str = '/data/users2/ibatta/projects/deepsubspace/in/filemapper.json' # filemapper path: json file with mapping of fkey variables to relevant filepaths and filenames for data loading.
    fm: dict = field(default_factory=lambda: json.load(open('/data/users2/ibatta/projects/deepsubspace/in/filemapper.json','r')))
    scorename: str = 'labels'
    cuda_avl: bool = True
    nw: int = 8
    cr: str = 'clx'
    tss: int = 100  # modification automated via slurmTaskIDMapper
    rep: int = 0  # modification automated via slurmTaskIDMapper
    raytune: bool = False


class MRIDataset(Dataset):

    def __init__(self, cfg, mode):
        # self.df = readFrames(cfg.ssd, mode, cfg.tss, cfg.rep)
        self.df = readFramesSeparately(cfg.sm, cfg.ssd, mode, cfg.rep)
        self.fkey = cfg.fkey
        self.scorename = cfg.scorename
        self.cr = cfg.cr
        self.fm = cfg.fm
        self.ds = cfg.ds
        self.nch = len(cfg.fkey.split(','))
        self.ngr = cfg.ngr
        self.raytune = cfg.raytune

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        X, y = read_X_y_5D_idx(self.df, self.fkey, idx, self.scorename, self.cr, self.fm, self.ds, replace_nan=True)
        if self.cr == 'reg':
            # print('Replacing nan..')
            y[np.isnan(y)] = np.mean(y[~np.isnan(y)])
        return [X, y]


def load_split_idx(ssd, mode, rep):
    """
    Loads indices for a particular split. 

    Args:
        ssd: directory path for splits files 
        mode: train (tr), validation (va) or test (te)					  
        rep: repetition number

    Returns:
        idx: 1-D array of indices corresponding to the arguments

    """
    return np.loadtxt( ssd + mode + '_r' + str(rep) + '.csv', dtype=int)


def readFramesSeparately(sm, ssd, mode, rep):
    """
    Reads frames into a df, but assumes files with tr,va,te splits to contain only indices. Uses master files to load corresponding subject features.

    Args:
        sm: master file for scores, labels, filepaths for all subjects
        (deprecated) fl: master file with list of paths to input scans for all subjects (should have header ideally for supporting multimodal)
        ssd: directory path for splits files 
        mode: train (tr), validation (va) or test (te)					  
        rep: repetition number

    Returns:
        df: pandas dataframe with with fields from sm, fl files and indices from corresponding file in ssd based on mode, rep

    """
    idx = load_split_idx(ssd, mode, rep)
    # df_fl = pd.read_csv(fl)
    df_sm = pd.read_csv(sm).iloc[idx]
    # assert df_fl.shape[0] == df_sm.shape[0]
    # df_sm['smriPath'] = df_fl['smriPath']
    # df_sm = df_sm.iloc[idx]
    
    return df_sm


def readFrames(ssd, mode, tss, rep):
    
    # Read Data Frame
    df = pd.read_csv(ssd + mode + '_' + str(tss) +
                     '_rep_' + str(rep) + '.csv')

    print('Mode ' + mode + ' :' + 'Size : ' +
          str(df.shape) + ' : DataFrames Read ...')

    return df


def read_X_y_5D_idx(df, fkey, idx, scorename, cr, fm, ds, replace_nan=True):
    # print('Read 5D XY: '+fkey+'\n')
    X_all = []
    fkeys = fkey.split(',')
    for curkey in fkeys:
        X, y = [], []

        basekey = fm['basepathmapper'][ds][curkey]
        relpath = df[basekey].iloc[idx]
        # relpath = df['relative_dir_path'].iloc[idx]
        filename = fm['filename'][ds][curkey]
        basedir = fm['basedir'][ds][basekey]
        fN = basedir+relpath+filename
        # print(fN)

        # Read image
        X = np.float32(nib.load(fN).get_fdata())
        X = (X - X.min()) / (X.max() - X.min())
        # X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        
        # if np.any(np.isnan(X)):
        #     # print('Nan FOUND: '+fN)
        #     X[np.isnan(X)] = np.mean(X[~np.isnan(X)])
            
        X_all.append(X)
    X_all = np.array(X_all)
    
    # Read label
    # ydf = df[scorename]
    # ydf = ydf.fillna(ydf.values.mean())
    y = df[scorename].iloc[idx]
    # y = df[scorename].values[idx]
    
    if 'label' in scorename:
        y -= 1
        
    if cr == 'reg':
        y = np.array(np.float32(y))
        # import pdb; pdb.set_trace()
        if replace_nan:
            # print('REPLACING nan..')
            y[np.isnan(y)] = np.mean(y[~np.isnan(y)])
            y[np.isinf(y)] = np.mean(y[~np.isinf(y)])
        # y = np.array(np.float64(y))
    elif cr == 'clx':
        y = np.array(y)

    return X_all, y


def train(dataloader, net, optimizer, criterion, cuda_avl):

    net.train()

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        # Fetch the inputs
        inputs, labels = data
        # labels[np.isnan(labels)] = labels[~np.isnan(labels)].mean()

        # Wrap in variable and load batch to gpu
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs[0].squeeze(), labels)
        loss.backward()
        optimizer.step()

    return loss


def test(dataloader, net, cuda_avl, cr):

    net.eval()
    y_pred = np.array([])
    y_true = np.array([])

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        inputs, labels = data
        # import pdb; pdb.set_trace()
        # labels[np.isnan(labels)] = labels[~np.isnan(labels)].mean()

        # Wrap in variable and load batch to gpu
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass
        outputs = net(inputs)

        if cr == 'clx':
            _, predicted = torch.max(outputs[0].data, 1)
            y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))
        elif cr == 'reg':
            y_pred = np.concatenate((y_pred, outputs[0].data.cpu().numpy().squeeze()))

        y_true = np.concatenate((y_true, labels.data.cpu().numpy()))

    return y_true, y_pred


def evalMetrics(dataloader, net, cfg):

    # Batch Dataloader
    y_true, y_pred = test(dataloader, net, cfg.cuda_avl, cfg.cr)
    # import pdb; pdb.set_trace()

    if cfg.cr == 'clx':

        # Evaluate classification performance
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0, average='micro')
        # ap = average_precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='micro')
        js = jaccard_score(y_true, y_pred, average='micro')
        rocauc = roc_auc_score(hot_encode(y_true), hot_encode(y_pred, vals=np.unique(y_true)), average='micro', multi_class='ovr')
        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # specificity = tn / (tn+fp)
        # sensitivity = 


        return acc, bal_acc, precision, recall, js, rocauc

    elif cfg.cr == 'reg':
        # Evaluate regression performance
        mae = mean_absolute_error(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        nrmse = mean_squared_error(y_true, y_pred, squared=False) / np.ptp(y_true)
        r2 = r2_score(y_true, y_pred)
        r, p = pearsonr(y_true, y_pred)

        return mae, ev, mse, nrmse, r2, r, p

    else:
        print('Check cr flag')


def generate_validation_model(cfg):
    print('Running genval..')
    # Initialize net based on model type (mt, nc)
    net = initializeNet(cfg)

    # Training parameters
    epochs_no_improve = 0
    valid_acc = 0

    if cfg.cr == 'clx':
        criterion = nn.CrossEntropyLoss()
        reduce_on = 'max'
        m_val_acc = 0
        history = pd.DataFrame(columns=['scorename', 'iter', 'epoch',
                                        'tr_acc', 'bal_tr_acc', 'val_acc', 'bal_val_acc', 'loss'])
    elif cfg.cr == 'reg':
        criterion = nn.MSELoss()
        reduce_on = 'min'
        m_val_acc = 100
        history = pd.DataFrame(columns=['scorename', 'iter', 'epoch', 'tr_mae', 'tr_ev', 'tr_mse', 'tr_nrmse',
                                        'tr_r2', 'tr_r', 'tr_p', 'val_mae', 'val_ev', 'val_mse', 'val_nrmse', 'val_r2', 'val_r', 'val_p', 'loss'])
    else:
        print('Check config flag cr')

    # Load model to gpu
    if cfg.cuda_avl:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Declare optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)


    # To restore a checkpoint, use `session.get_checkpoint()`.
    if cfg.raytune:
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)


    # Declare learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode=reduce_on, factor=0.5, patience=7, verbose=True)

    # Batch Dataloader
    trainloader = loadData(cfg, 'tr')
    validloader = loadData(cfg, 'va')
    
    

    for epoch in range(cfg.es):

        # Train
        print('Training: ')
        loss = train(trainloader, net, optimizer, criterion, cfg.cuda_avl)
        loss = loss.data.cpu().numpy()

        if cfg.cr == 'clx':

            print('Validating: ')

            # Evaluate classification perfromance on training and validation data
            train_acc, bal_train_acc = evalMetrics(trainloader, net, cfg)[:2]
            valid_acc, bal_valid_acc = evalMetrics(validloader, net, cfg)[:2]

            # Log Performance
            history.loc[epoch] = [cfg.scorename, cfg.iter, epoch, train_acc,
                                  bal_train_acc, valid_acc, bal_valid_acc, loss]

            # Check for maxima (e.g. accuracy for classification)
            isBest = valid_acc > m_val_acc

        elif cfg.cr == 'reg':

            print('Validating: ')
            # Evaluate regression perfromance on training and validation data
            train_mae, train_ev, train_mse, train_nrmse, train_r2, train_r, train_p = evalMetrics(
                trainloader, net, cfg)
            valid_acc, valid_ev, valid_mse, valid_nrmse, valid_r2, valid_r, valid_p = evalMetrics(
                validloader, net, cfg)

            print('LOG..')
            # Log Performance
            history.loc[epoch] = [cfg.scorename, cfg.iter, epoch, train_mae, train_ev, train_mse, train_nrmse, train_r2,
                                  train_r, train_p, valid_acc, valid_ev, valid_mse, valid_nrmse, valid_r2, valid_r, valid_p, loss]

            # Check for minima (e.g. mae for regression)
            isBest = valid_acc < m_val_acc

        else:
            print('Check cr flag')

        # Write Log
        history.to_csv(cfg.ml + 'history.csv', index=False)

        # Early Stopping
        if cfg.es_va:

            # If minima/maxima
            if isBest:

                # Save the model
                torch.save(net.state_dict(), open(
                    cfg.ml + 'model_state_dict.pt', 'wb'))

                # Reset counter for patience
                epochs_no_improve = 0
                m_val_acc = valid_acc

            else:

                # Update counter for patience
                epochs_no_improve += 1

                # Check early stopping condition
                if epochs_no_improve == cfg.es_pat:

                    print('Early stopping!')

                    # Stop training: Return to main
                    return history, m_val_acc

        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_acc)
        
        if cfg.raytune:
            torch.save(
                (net.state_dict(), optimizer.state_dict()), cfg.ml+"checkpoint.pt")
            checkpoint = Checkpoint.from_directory(cfg.ml)
            session.report({"loss": valid_acc}, checkpoint=checkpoint)
            # session.report({"loss": valid_acc})
    print("Finished Training")
        
        
        


def hot_encode(a, vals=None):
    if np.any(vals) == None: 
        vals = np.unique(a)
    ## Assumes labels as ints starting from 0
    b = np.zeros([len(a),len(vals)])
    for i in range(len(a)):
        b[i,int(a[i])] = 1
    return b

def evaluate_test_accuracy(cfg):

    # Load validated net
    net = loadNet(cfg)
    net.eval()

    # Dataloader
    testloader = loadData(cfg, 'te')

    if cfg.cr == 'clx':

        # Initialize Log File
        outs = pd.DataFrame(columns=['iter', 'acc', 'bal_acc', 'precision', 'recall', 'js', 'rocauc'])

        print('Testing: ')

        # Evaluate classification performance
        acc, bal_acc, precision, recall, js, rocauc = evalMetrics(testloader, net, cfg)

        # Log Performance
        outs.loc[0] = [cfg.iter, acc, bal_acc, precision, recall, js, rocauc]

    elif cfg.cr == 'reg':

        # Initialize Log File
        outs = pd.DataFrame(columns=[
                            'iter', 'mae_te', 'ev_te', 'mse_te', 'nrmse_te', 'r2_te', 'r_te', 'p_te'])

        print('Testing: ')

        # Evaluate regression performance
        mae, ev, mse, nrmse, r2, r, p = evalMetrics(testloader, net, cfg)

        # Log Performance
        outs.loc[0] = [cfg.iter, mae, ev, mse, nrmse, r2, r, p]

    else:
        print('Check cr mode')
    # import pdb; pdb.set_trace()
    # Write Log
    outs.to_csv(cfg.ml+'test.csv', index=False)
    saveCfg(cfg,fname='config_test.pkl')
    

def saveCfg(cfg, fname='config.pkl'):
    with open(cfg.ml+fname, 'wb') as f:
        pickle.dump(cfg, f)

def loadCfg(fpath):
    with open(fpath,'rb') as f:
        cfg_dict = dataclasses.asdict(pickle.load(f))
    return Config(** cfg_dict)

def loadData(cfg, mode):

    # Batch Dataloader
    prefetch_factor = 8 # doesn't seem to be working; tried 1, 2, 4, 8, 16, 32 - mem used stays the same! need to verify the MRIdataset custom functionality maybe
    dset = MRIDataset(cfg, mode)

    # import pdb; pdb.set_trace()

    dloader = DataLoader(dset, batch_size=cfg.bs,
                         shuffle=True, num_workers=cfg.nw, drop_last=True, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)

    return dloader


def loadNet(cfg):

    # Load validated model
    net = initializeNet(cfg)
    model = torch.nn.DataParallel(net)
    net = 0
    net = load_net_weights2(model, cfg.ml+'model_state_dict.pt')

    return net


def updateIterML(cfg):
    print('Updating Iter ML.. ')
    # Update Iter (in case of multitask training)
    
    # if cfg.pp!=None and cfg.pp:
    #     cfg.iter += 1

    # Map slurmTaskID to training sample size (tss) and CV rep (rep)
    cfg = slurmTaskIDMapper(cfg)

    # Update Model Location
    ml_suffix = 'mt_'+cfg.mt+'_fkey_'+cfg.fkey+'_scorename_'+cfg.scorename+'_iter_' + \
        str(cfg.iter)+'_nc_'+str(cfg.nc)+'_rep_'+str(cfg.rep)+'_bs_'+str(cfg.bs)+'_lr_' + \
        str(cfg.lr)+'_espat_'+str(cfg.es_pat)+'/'
    if np.all(['mt_' in cfg.ml, 'fkey_' in cfg.ml, '_scorename_' in cfg.ml, '_espat_' in cfg.ml]):
        # Only change suffix
        cfg.ml = '/'.join(cfg.ml.split('/')[:-2]) + '/' + ml_suffix
    else:
        # Only append the suffix
        cfg.ml = cfg.ml + ml_suffix
        
    # if len(cfg.fkey) > 1000:
    #     print('filename long!!!!! Replacing with xyz')
    #     cfg.ml = cfg.ml.replace(cfg.fkey,'xyz')
        
    # Make Model Directory
    try:
        os.stat(cfg.ml)
    except:
        os.mkdir(cfg.ml)

    return cfg


def slurmTaskIDMapper(cfg):
    # Only valid when slurm task IDs directly correspond to reps.
    cfg.rep = cfg.iter
    fns = list(cfg.fm['filename'][cfg.ds].keys()) # list of feature names
    # fns = ['tsavg','tsmedian','tsmax','tsmin']
    new_mods = ['tsavg','tsmedian','tsmax','tsmin'] 
    fns_remove = ['hT1','PerAF'] #+ ['tsavg','tsmedian','tsmax','tsmin'] 
    for fi in fns_remove:
        fns.remove(fi)
    fns.append(','.join([fi for fi in fns if fi not in new_mods])) #multimodal with all features into DL model
    fns += [fi+',lT1' for fi in fns if 'lT1' not in fi] # multimodal with 2 modalities: with one fmri measure with low-res smri 
    # fns.append(','.join([','.join(fns) for i in range(8)] )) # Testing 56 groups for future  
    # print(fns)

    # # Map iter value (slurm taskID) to training sample size (tss) and crossvalidation repetition (rep)
    fv, rv = np.meshgrid(np.arange(len(fns)), np.arange(cfg.nReps))
    fv = fv.reshape((1, np.prod(fv.shape)))
    rv = rv.reshape((1, np.prod(rv.shape)))
    fkey = fns[fv[0][cfg.iter]]
    rep = rv[0][cfg.iter]
    print(fkey, rep)
    cfg.fkey = fkey
    cfg.nch = len(fkey.split(','))
    cfg.rep = rep
    print(cfg.iter, cfg.tss, cfg.rep)

    return cfg


def initializeNet(cfg):
    print("Model:%s, Channels:%d, Groups:%d"%(cfg.mt, cfg.nch,cfg.nch))
    # Initialize net based on model type (mt, nc)
    if cfg.mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=cfg.nc, num_channels=cfg.nch)
    elif cfg.mt == 'AN3DdrlowresAvg':
        net = AN3Ddr_lowresAvg(num_classes=cfg.nc, num_channels=cfg.nch)
    elif cfg.mt == 'AN3DdrlrMx':
        net = AN3Ddr_lowresMax(num_classes=cfg.nc, num_channels=cfg.nch)
    elif cfg.mt == 'AN3DdrlrMxLt':
        net = AN3Ddr_lowresMaxLight(num_classes=cfg.nc, num_channels=cfg.nch)
    elif cfg.mt == 'AN3DdrlrMxG':
        ## Have kept ngr = nch for now, since each modality is being considered as a group.
        ## Can change it later when using sub-region instead of modalities.
        net = AN3Ddr_lowresMax(num_classes=cfg.nc, num_channels=cfg.nch, num_groups=cfg.nch)
    elif cfg.mt == 'AN3DdrlrMxLtG':
        ## Have kept ngr = nch for now, since each modality is being considered as a group.
        ## Can change it later when using sub-region instead of modalities.
        net = AN3Ddr_lowresMaxLight(num_classes=cfg.nc, num_channels=cfg.nch, num_groups=cfg.nch)
    elif cfg.mt == 'AN3DdrhighresMax':
        net = AN3Ddr_highresMax(num_classes=cfg.nc, num_channels=cfg.nch)
    else:
        print('Check model type')

    return net


def load_net_weights2(net, weights_filename):

    # Load trained model
    state_dict = torch.load(
        weights_filename,  map_location=lambda storage, loc: storage)
    state = net.state_dict()
    state.update(state_dict)
    net.load_state_dict(state)

    return net




def temp_valid_func(hyper_params,cfg):
    
    cfg_dict = asdict(cfg)
    if type(hyper_params) == dict:
        for key, value in hyper_params.items():
            cfg_dict[key] = value
    cfg = Config(**cfg_dict)
    cfg = updateIterML(cfg)
    generate_validation_model(cfg)
    return


def tune_hyperparams(cfg, num_samples=10):
    print('Model will run with hyper-parameter tuning..')

    search_space = {
        'lr': tune.loguniform(1e-5,1e-1),
        'bs': tune.choice([8,16,32,64,128]),
    }
    max_num_epochs = cfg.es
    if cfg.cr == 'clx':
        mode = 'max'
    elif cfg.cr == 'reg':
        mode = 'min'
    metric = 'loss'
    
    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(lambda x: temp_valid_func(x, cfg)),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_space,
        run_config=air.RunConfig(local_dir='../out/raytune/',name=cfg.ml.split('/')[-2])
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(metric,mode)
    
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    
    return best_result


def loadMasks(cfg, nibObjectsOnly=False):
    masks = []
    fkeys = cfg.fkey.split(',')
    for curkey in fkeys:
        basekey = cfg.fm['basepathmapper'][cfg.ds][curkey]
        maskpath = cfg.fm['maskbasepath'] + cfg.ds +'/' + cfg.fm['maskmapper'][cfg.ds][basekey]
        if not nibObjectsOnly:
            curmask = np.float32(nib.load(maskpath).get_fdata())
            curmask = np.reshape(curmask, (curmask.shape[0],curmask.shape[1],curmask.shape[2]))
        else:
            curmask = nib.load(maskpath)
        masks.append(curmask)
    if not nibObjectsOnly:
        masks = np.array(masks)
    return masks
        



def normalize_image(X, method='zscore', axis=None):
    if method == 'zscore':
        return zscore(X, axis=axis)
    elif method == 'minmax':
        return minmax(X)
    elif method == None:
        return X

def minmax(X):
    return ((X-X.min())/(X.max()-X.min()))


def normalize_4D(im4D, method='zscore'):
    # 4D normalization (assumes final dimension is subject)
    mim = []
    for i in np.arange(im4D.shape[3]):
        im = im4D[..., i]
        im = normalize_image(im, method=method)
        mim.append(im)
    mim = np.array(mim)
    return mim

def normalize_5D(im5D, method='zscore'):
    
    mim = np.zeros(im5D.shape)
    for i in range(im5D.shape[0]):
        for j in range(im5D.shape[1]):
            mim[i,j,:,:,:] = normalize_image(im5D[i,j,:,:,:], method=method)
    return mim


def fil_im(smap, normalize_method='zscore'):
    # smap : 5D: nSubs x nCh(1) x X x Y x Z
    s = 2  # sigma gaussian filter
    w = 9  # kernal size gaussian filter
    # truncate gaussian filter at "t#" standard deviations
    t = (((w - 1)/2)-0.5)/s
    fsmap = []
    for i in np.arange(smap.shape[0]):
        im = smap[i]
        im = normalize_image(im, method=normalize_method)
        im = gaussian_filter(im, sigma=s, truncate=t)
        im = normalize_image(im, method=normalize_method)
        fsmap.append(im)
    fsmap = np.array(fsmap)
    return fsmap


def fil_im_5d(smap, normalize_method='zscore'):
    # smap : 5D: nSubs x nCh(1) x X x Y x Z
    s = 2  # sigma gaussian filter
    w = 9  # kernal size gaussian filter
    # truncate gaussian filter at "t#" standard deviations
    t = (((w - 1)/2)-0.5)/s
    fsmap = np.zeros(smap.shape)
    for i in np.arange(smap.shape[0]):
        for j in np.arange(smap.shape[1]):
            im = smap[i,j,:,:,:]
            im = normalize_image(im, method=normalize_method)
            im = gaussian_filter(im, sigma=s, truncate=t)
            im = normalize_image(im, method=normalize_method)
            fsmap[i,j,:,:,:] = im
    return fsmap


def area_occlusion(model, image_tensor, area_masks, target_class=None, occlusion_value=0, apply_softmax=True, cuda=False, verbose=False, taskmode='clx'):
    
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor    
    
    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False))[0]
    
    if apply_softmax:
        output = F.softmax(output)
    
    if taskmode == 'reg':
        unoccluded_prob = output.data
    elif taskmode == 'clx':
        output_class = output.max(1)[1].data.cpu().numpy()[0]    

        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
        
        if target_class is None:
            target_class = output_class
        unoccluded_prob = output.data[0, target_class]
    
    relevance_map = torch.zeros(image_tensor.shape[1:])
    if cuda:
        relevance_map = relevance_map.cuda()
    
    for area_mask in area_masks:

        area_mask = torch.FloatTensor(area_mask)

        if cuda:
            area_mask = area_mask.cuda()
        image_tensor_occluded = image_tensor * (1 - area_mask).view(image_tensor.shape)
        
        output = model(Variable(image_tensor_occluded[None], requires_grad=False))[0]
        if apply_softmax:
            output = F.softmax(output)
            
        if taskmode == 'reg':
            occluded_prob = output.data
        elif taskmode == 'clx':
            occluded_prob = output.data[0, target_class]
        
        ins = area_mask.view(image_tensor.shape) == 1
        ins = ins.squeeze()
        relevance_map[ins] = (unoccluded_prob - occluded_prob)

    relevance_map = relevance_map.cpu().numpy()
    relevance_map = np.maximum(relevance_map, 0)
    return relevance_map



def test_saliency(cfg):
    print('Testing saliency..')
    cuda_avl = cfg.cuda_avl
    
    net = loadNet(cfg)
    
    
    
    if cfg.cuda_avl:
        net.load_state_dict(torch.load(cfg.ml+'model_state_dict.pt'))
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    else:
        net.load_state_dict(torch.load(cfg.ml+'model_state_dict.pt', map_location=torch.device('cpu')))

    net.eval()
    
    sal , fsal, imsal, fimsal = [],[],[],[]
    dataloader = loadData(cfg, 'te')
    # Iterate over dataloader batches
    test_labels = []
    for _, data in enumerate(dataloader, 0):
        print('Running new batch..')
        inputs, labels = data
        test_labels.append(np.squeeze(labels))
        
        # # Wrap in variable and load batch to gpu
        # if cuda_avl:
        #     inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # else:
        #     inputs, labels = Variable(inputs), Variable(labels)

        # tempmask = np.ones(inputs.shape[2:])
        masks = loadMasks(cfg)
        sal_out = run_saliency(cfg.ml, 'BPraw', inputs, net, masks, cfg.scorename, cfg.cr, cuda_avl=cuda_avl)
        sal.append(sal_out[0])
        fsal.append(sal_out[1])
        imsal.append(sal_out[2])
        fimsal.append(sal_out[3])
        
    test_labels = np.hstack(test_labels).squeeze()
    np.savetxt(cfg.ml+'test_labels.txt', test_labels, fmt='%d')
    
    print('Saving saliency results to: \n'+cfg.ml)
    sal = np.vstack(sal)
    with open(cfg.ml+'sal_zscore.pkl','wb') as f:
        pickle.dump(sal,f)

    fsal = np.vstack(fsal)
    with open(cfg.ml+'fsal_zscore.pkl','wb') as f:
        pickle.dump(fsal,f)

    imsal = np.vstack(imsal)
    with open(cfg.ml+'imsal_zscore.pkl','wb') as f:
        pickle.dump(imsal,f)

    fimsal = np.vstack(fimsal)
    with open(cfg.ml+'fimsal_zscore.pkl','wb') as f:
        pickle.dump(fimsal,f)

    
    
    



def run_saliency(odir, itrpm, images, net, area_masks, scorename, taskM, cuda_avl=True):
    nsubs, nch, nx, ny, nz = images.shape
    sal_measures = ['saliency', 'filtered_saliency', 'imtimes_saliency', 'filtered_imtimes_saliency']
    sal_out = np.array([np.zeros(images.shape) for m in sal_measures])
    for nSub in range(nsubs):
        print(nSub)
        for chi in range(nch): 
            fname = odir + itrpm + '_' + scorename  + '_nSub_' + str(nSub) + '.nii' 
            if itrpm == 'AO':
                interpretation_method = area_occlusion
                sal_map = interpretation_method(net, images[nSub], area_masks, occlusion_value=0, apply_softmax=False, cuda=True, verbose=False,taskmode=taskM)
            elif itrpm == 'BPraw':
                interpretation_method = sensitivity_raw
                # im = images[nSub].reshape(1,1,121,145,121)
                # im = images[nSub,chi,:,:,:].reshape(1,1,images.shape[2],images.shape[3],images.shape[4])
                
                for mi in range(len(sal_measures)):
                    sal_out[mi] = interpretation_method(net, images, area_masks, sal_measures[mi], None,cuda=cuda_avl, verbose=False, taskmode=taskM)
                
            elif itrpm == 'BP':
                interpretation_method = sensitivity_analysis
                # im = images[nSub].reshape(1,1,121,145,121)
                im = images[nSub,chi,:,:,:].reshape(1,1,images.shape[2],images.shape[3],images.shape[4])
                
                cur_sal_outs = interpretation_method(net, im, area_masks[chi,:,:,:],None,cuda=cuda_avl, verbose=False, taskmode=taskM)
                for mi in range(len(sal_measures)):
                    sal_out[mi,nSub,chi,:,:,:] = cur_sal_outs[mi]
            else:
                print('Verify interpretation method')   
            
        
            # nib.save(nib.Nifti1Image(sal_map.squeeze() , np.eye(4)), fname)
    return sal_out

def sensitivity_raw(model, im, mask,gradmode, target_class=None, cuda=True, verbose=False, taskmode='clx'):
    # model: pytorch model set to eval()
    # im: 5D image - nSubs x numChannels x X x Y x Z
    # mask: group input data mask - numChannles x X x Y x Z
    # gradmode: 'saliency', 'filtered_saliency', 'imtimes_saliency', 'filtered_imtimes_saliency'
 
    # sal_map: gradient [4D image: X X Y X Z X nSubs]
    
    if cuda:
        im = torch.Tensor(im).cuda()
    else:
        im = torch.Tensor(im)
 
    im = Variable(im, requires_grad=True)
 
    # Forward Pass
    output = model(im)[0]
 
    # Predicted labels
    if taskmode == 'clx':
        output = F.softmax(output, dim=1)
        #print(output, output.shape)
 
    # Backward pass.
    model.zero_grad()
 
    output_class = output.cpu().max(1)[1].numpy()
    output_class_prob = output.cpu().max(1)[0].detach().numpy()
 
    if verbose:
        print('Image was classified as', output_class,
              'with probability', output_class_prob)
 
    # one hot encoding
    one_hot_output = torch.zeros(output.size())
 
    for i in np.arange(output.size()[0]):
        if target_class is None:
            one_hot_output[i][output_class[i]] = 1
        else:
            one_hot_output[i][target_class[i]] = 1
 
    if cuda:
        one_hot_output = one_hot_output.cuda()
 
    # Backward pass
    output.backward(gradient=one_hot_output)
 
    # Gradient
    
    # sal_map = np.squeeze(im.grad.cpu().numpy(), axis=1) # Removes the channel axis
    # sal_map = im.grad.cpu().numpy().squeeze(axis=0) # Remove the subject axis
    sal_map = im.grad.cpu().numpy() # Remove the subject axis
    
    if 'imtimes' in gradmode:
        sal_map = sal_map * im.cpu().detach().numpy()
        
    if 'filtered' in gradmode:
        sal_map = fil_im_5d(sal_map, normalize_method='zscore')
    else:
        # only normalize withouxt filtering
        sal_map = normalize_5D(sal_map, method='zscore')
    
    # Mask Gradient
    # mask = np.tile(mask[None], (im.shape[0], 1, 1, 1))
    sal_map *= mask
    
    
    return sal_map

 
def sensitivity_analysis(model, im, mask, target_class=None, cuda=True, verbose=False, taskmode='clx'):
   
    # model: pytorch model set to eval()
    # im: 5D image - nSubs x numChannels x X x Y x Z
    # mask: group input data mask - numChannles x X x Y x Z
 
    # sal_map: gradient [4D image: X X Y X Z X nSubs]
    # fil_sal_map: filtered gradient [4D image: X X Y X Z X nSubs]
    # grad_times_im_sal_map: gradient x input [4D image: X X Y X Z X nSubs]
    # fil_grad_times_im_sal_map: filtered (gradient x input) [4D image: X X Y X Z X nSubs]
     
 
    if cuda:
        im = torch.Tensor(im).cuda()
    else:
        im = torch.Tensor(im)
 
    im = Variable(im, requires_grad=True)
 
    # Forward Pass
    output = model(im)[0]
 
    # Predicted labels
    if taskmode == 'clx':
        output = F.softmax(output, dim=1)
        #print(output, output.shape)
 
    # Backward pass.
    model.zero_grad()
 
    output_class = output.cpu().max(1)[1].numpy()
    output_class_prob = output.cpu().max(1)[0].detach().numpy()
 
    if verbose:
        print('Image was classified as', output_class,
              'with probability', output_class_prob)
 
    # one hot encoding
    one_hot_output = torch.zeros(output.size())
 
    for i in np.arange(output.size()[0]):
        if target_class is None:
            one_hot_output[i][output_class[i]] = 1
        else:
            one_hot_output[i][target_class[i]] = 1
 
    if cuda:
        one_hot_output = one_hot_output.cuda()
 
    # Backward pass
    output.backward(gradient=one_hot_output)
 
    # Gradient
    # sal_map = np.squeeze(im.grad.cpu().numpy(), axis=1) # Removes the channel axis
    sal_map = im.grad.cpu().numpy().squeeze(axis=0) # Remove the subject axis
   
    print(sal_map.shape)
 
    
    
    # Gradient X Image
    grad_times_im_sal_map = sal_map * im.cpu().detach().numpy()
    grad_times_im_sal_map = np.moveaxis(grad_times_im_sal_map, 0, -1)
   
    print(grad_times_im_sal_map.shape)
 
    # Filter : Gradient X Image
    fil_grad_times_im_sal_map = fil_im(grad_times_im_sal_map)
 
    fil_grad_times_im_sal_map = np.moveaxis(
        fil_grad_times_im_sal_map, 0, -1)
 
    # Mask Gradient
    mask = np.tile(mask[None], (im.shape[0], 1, 1, 1))
    sal_map *= mask
    sal_map = np.moveaxis(sal_map, 0, -1)
 
    # Filter : Gradient
    fil_sal_map = fil_im(sal_map)
    fil_sal_map = np.moveaxis(fil_sal_map, 0, -1)
 
    # Scale non-filtered
    sal_map = np.moveaxis(normalize_4D(sal_map), 0, -1)
    grad_times_im_sal_map = np.moveaxis(
        normalize_4D(grad_times_im_sal_map), 0, -1)
 
    return sal_map, fil_sal_map, grad_times_im_sal_map, fil_grad_times_im_sal_map
 
