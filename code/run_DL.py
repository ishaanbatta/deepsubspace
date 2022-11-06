import os, sys
import utils as ut
import torch
import argparse
import numpy as np


import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SLvsML')
    parser.add_argument('--tr_smp_sizes', nargs="*", type=int,
                        default=(100, 200, 500, 1000, 2000, 5000, 10000), help='')
    parser.add_argument('--nReps', type=int, default=20, metavar='N',
                        help='random seed (default: 20)')
    parser.add_argument('--nc', type=int, default=10, metavar='N',
                        help='number of classes in dataset (default: 10)')
    parser.add_argument('--bs', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
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
    parser.add_argument('--cr', default='clx', metavar='S',
                        help='classification (clx) or regression (reg) - (default: clx)')
    parser.add_argument('--tss', type=int, default=100, metavar='N',
                        help='training sample size (default: 100)')
    parser.add_argument('--rep', type=int, default=0, metavar='N',
                        help='crossvalidation rep# (default: 0)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--raytune', action='store_true',
                        help='include if Ray tune is to be used for hyper-parameter tuning')


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
        iter = 10

    cfg = ut.Config(iter=iter, tr_smp_sizes=args.tr_smp_sizes, nReps=args.nReps, nc=args.nc, bs=args.bs, lr=args.lr, es=args.es, pp=args.pp,
                    es_va=args.es_va, es_pat=args.es_pat, ml=args.ml, mt=args.mt, ssd=args.ssd, sm=args.sm, fkey=args.fkey, scorename=args.scorename, cuda_avl=cuda_avl, nw=args.nw, cr=args.cr, tss=args.tss, rep=args.rep)

    # Update iteration (multitask training is controlled by pp flag) and model location
    cfg = ut.updateIterML(cfg)
    ut.saveCfg(cfg)
    
    
        
    # ut.test_saliency(cfg)
    # sys.exit(0)
    
    # if (cfg.mt in ['AN3Ddr_lrMxG','AN3Ddr_lrMxLtG']) and (',' not in cfg.fkey):
    #     print('Skipping Computation due to Single Channel Multigroup Model!! ')
    #     sys.exit(0)

    if not args.raytune:

        #train
        ut.generate_validation_model(cfg)

        #test
        ut.evaluate_test_accuracy(cfg)
        
    else:
        # train with hyperparameter tuning 
        best_result = ut.tune_hyperparams(cfg)
        
        import pdb; pdb.set_trace()
        
        # Update iteration (multitask training is controlled by pp flag) and model location
        best_result = ut.updateIterML(best_result)
        ut.saveCfg(best_result)
            
        # test on the best model obtained
        ut.evaluate_test_accuracy(best_result)