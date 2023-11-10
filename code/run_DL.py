import os, sys
from time import sleep
import utils as ut
import torch
import argparse
import numpy as np
import itertools


import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from summarize_gica import summarize_gica

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SLvsML')
    parser.add_argument('--tr_smp_sizes', nargs="*", type=int,
                        default=(100, 200, 500, 1000, 2000, 5000, 10000), help='')
    parser.add_argument('--nReps', type=int, default=20, metavar='N',
                        help='random seed (default: 20)')
    parser.add_argument('--thr', type=float, default=0.0, metavar='THR',
                        help='two-sided threshold for masking out voxels (default = 0.0)')
    parser.add_argument('--nc', type=int, default=10, metavar='N',
                        help='number of classes in dataset (default: 10)')
    parser.add_argument('--bs', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iter', type=int, default=0, metavar='N',
                        help='value for iteration variable')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--es', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--pp', type=int, default=0, metavar='N',
                        help='iteration flag to allow multiple  processes')
    parser.add_argument('--es_va', type=int, default=1, metavar='N',
                        help='1: use val accuracy; 0: use val loss : default on i.e. uses val accuracy')
    parser.add_argument('--es_pat', type=int, default=20, metavar='N',
                        help='patience for early stopping (default: 20)')
    parser.add_argument('--ml', default='./temp/', metavar='S',
                        help='model location (default: ./temp/)')
    parser.add_argument('--mt', default='AlexNet3D', metavar='S',
                        help='modeltype (default: AlexNet3D)')
    parser.add_argument('--ds', default='ADNI', metavar='S',
                        help='dataset name')
    parser.add_argument('--ssd', default='/SampleSplits/', metavar='S',
                        help='sample size directory (default: /SampleSplits/)')
    parser.add_argument('--sm', default=None, metavar='S',
                        help='filepath for master file with all scores, filepaths, labels etc.')
    parser.add_argument('--fkey', default=None, metavar='F',
                        help='Name of filepath key to be used from sm file')                        
    parser.add_argument('--scorename', default='age', metavar='S',
                        help='scorename (default: fluid_intelligence_score_f20016_2_0)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='turn off to enable CUDA training')
    parser.add_argument('--nw', type=int, default=8, metavar='N',
                        help='number of workers (default: 8)')
    parser.add_argument('--sf', type=int, default=4, metavar='N',
                        help='scaling factor for the model if applicable (default: 4)')
    parser.add_argument('--cr', default='clx', metavar='S',
                        help='classification (clx) or regression (reg) - (default: clx)')
    parser.add_argument('--tss', type=int, default=100, metavar='N',
                        help='training sample size (default: 100)')
    parser.add_argument('--rep', type=int, default=0, metavar='N',
                        help='crossvalidation rep# (default: 0)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--raytune', default=False,action='store_true',
                        help='include if Ray tune is to be used for hyper-parameter tuning')
    parser.add_argument('--saliency', default=False, action='store_true',
                        help='include if running in saliency analysis mode. Produces saliency maps as output.')
    parser.add_argument('--groupICA', default=False, action='store_true',
                        help='include if running in group ICA analysis mode. Produces group ICA results from saliency maps from directories across all repetitions as output. Results are saved only in the corresponding configuration model directory for the given repetition.')
    parser.add_argument('--gicaSummary', default=False, action='store_true',
                        help='include if summarizing group ICA analysis results. Produces group ICA plots and summaries from gica files generate from gICA analyssis on saliency maps from directories across all repetitions as output. Results are saved only in the corresponding configuration model directory for the given repetition.')
    parser.add_argument('--filtersal', default=False, action='store_true',
                        help='include if performing gaussian filtering on results from saliency maps. Results are saved only in the corresponding configuration model directory for the given repetition.')
    parser.add_argument('--salvars', default='sal', metavar='S',
                        help='comma separated list of filename prefixes for saliency analysis.')
    parser.add_argument('--testmode', default=False, action='store_true',
                        help='include if running only testing from a pre-trained model saved in model directory with appropritate name')
    parser.add_argument('--loadconfig', default=None, metavar='S',
                        help='include if running with a config already saved in a file')

    

    args = parser.parse_args()
    cuda_avl = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda_avl:
        torch.cuda.manual_seed(args.seed)
        
        
    

    # Slurm TaskID is mapped to training sample size and CV rep by
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        iter = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        iter = args.iter 
        
    

    if args.loadconfig != None: 
        cfg = ut.loadCfg(args.loadconfig)
        
        if not args.raytune: 
            cfg.raytune = False
            # cfg.ml = '../out/results/optimized/'
            cfg.ml = cfg.ml.replace('/raytune/','/optimized/')
            cfg.iter = iter
    else:
        cfg = ut.Config(iter=iter, ds=args.ds, tr_smp_sizes=args.tr_smp_sizes, thr=args.thr, nReps=args.nReps, nc=args.nc, bs=args.bs, lr=args.lr, es=args.es, pp=args.pp,
                        es_va=args.es_va, es_pat=args.es_pat, ml=args.ml, mt=args.mt, ssd=args.ssd, sm=args.sm, fkey=args.fkey, scorename=args.scorename, cuda_avl=cuda_avl, nw=args.nw, cr=args.cr, tss=args.tss, rep=args.rep, raytune=args.raytune, sf=args.sf)

    # Update iteration (multitask training is controlled by pp flag) and model location
    cfg = ut.updateIterML(cfg)
    ut.saveCfg(cfg)
    
    
    if args.saliency:
        print('Running in Saliency Mode..')
        ut.test_saliency(cfg)
        sys.exit(0)
    
    if args.filtersal:
        print('Performing filtering..')
        ut.test_filtering(cfg)
        sys.exit(0)
        
        
    if args.groupICA:
        print('Running group ICA on saliency maps across repetitions..')
        for varname in args.salvars.split(','):
            # for nc_pca, nc_ica in itertools.product([100,500,1000],[5,10,30,100]):
            # for nc_pca, nc_ica in itertools.product([10,30],[10,30]):
            for nc_pca, nc_ica in [(1000,10)]:
                
                if not args.gicaSummary:
                    print('Running gICA on '+varname + ', params: nc_pca=%d, nc_ica=%d'%(nc_pca, nc_ica))
                    ut.groupICA(cfg, md='te', varname=varname, nc_pca=nc_pca, nc_ica=nc_ica)
                    sleep(5)
                print('Summarizing gICA on '+varname + ', params: nc_pca=%d, nc_ica=%d'%(nc_pca, nc_ica))
                summarize_gica(cfg, md='te', varname=varname, nc_pca=nc_pca, nc_ica=nc_ica)
                sleep(3)
        sys.exit(0)
    
    
    # if (cfg.mt in ['AN3Ddr_lrMxG','AN3Ddr_lrMxLtG']) and (',' not in cfg.fkey):
    #     print('Skipping Computation due to Single Channel Multigroup Model!! ')
    #     sys.exit(0)

    if not cfg.raytune:

        if not args.testmode:
            #train
            ut.generate_validation_model(cfg)

        #test
        ut.evaluate_test_accuracy(cfg)
        
    else:
        
        # train with hyperparameter tuning 
        best_result = ut.tune_hyperparams(cfg)
        print('Completed hyperparameter tuning..')
                
        # Update the config and iteration (multitask training is controlled by pp flag) and model location
        ## iteration update happens within the update_hyperparams_cfg function
        best_config = ut.update_hyperparams_cfg(best_result.metrics['config'], cfg)
        ut.saveCfg(best_config,fname='best_config.pkl')
        
        # import pdb; pdb.set_trace()
        
        # test on the best model obtained
        ut.evaluate_test_accuracy(best_config)