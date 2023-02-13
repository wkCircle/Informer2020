import hyperopt 
import hyperopt.hp as hp
from hyperopt.pyll import scope as hpscope
from hyperopt.early_stop import no_progress_loss
from pathlib import Path 
import functools as ft 
import torch 
import random, sys, os, re
import numpy as np  
import pandas as pd 
import main_informer
import copy 
import pickle 
import shutil
from utils.trials import load_trials 


def get_informer_hyperspace(): 
    hyperspace = {
        "batch_size": hp.choice('batch_size', [1,2,4,8,16]), 
        "d_model": hp.choice('d_model', [32, 64, 128, 256, 512]), 
        "d_ff": hp.choice('d_ff', [16, 32, 64, 128, 256]), 
        "seq_len": hp.choice('seq_len', [36, 24, 12]), 
        "label_len": hp.choice('label_len', [24, 12, 6])
    }
    return hyperspace

def objective(hps: dict, args): 
    if hps['seq_len'] < hps['label_len']: 
        return {'loss': np.nan, 'status': hyperopt.STATUS_FAIL }
    # init 
    args = copy.deepcopy(args)
    for k, v in hps.items(): 
        setattr(args, k, v)

    # run train/val 
    main_informer.main(args)
    
    # get history 
    setting = main_informer.get_setting(args, 0)
    path = Path(args.checkpoints) / setting / 'history.pkl'
    with open(path, 'rb') as fr: 
        history = pickle.load(fr)

    # delete all files 
    shutil.rmtree(f'./results/{setting}')
    shutil.rmtree( Path(args.checkpoints) / f'{setting}')

    return {'loss': min(history['valid_epoch_loss']), 'status': hyperopt.STATUS_OK}
    
def main(): 
    # rng control 
    seed = 42 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # read args 
    args = main_informer.args_parse(verbose=False)
    args.iter = 1 # only support for iter =1 

    # define hyperparameter space for best hyperparams searching 
    hyperspace = get_informer_hyperspace()

    # define trial path to save 
    trialpath = './informer_aviation_hpotrials.pkl'
    # start hpo 
    trials = load_trials(trialpath, force_new=False)
    besthps: dict = hyperopt.fmin(
        fn=ft.partial(
            objective, args=args
        ),
        space=hyperspace,
        algo=hyperopt.tpe.suggest,
        max_evals=60,
        early_stop_fn=no_progress_loss(15),
        trials=trials,
        rstate=np.random.default_rng(42),
        trials_save_file=trialpath 
    )
    besthps = hyperopt.space_eval(hyperspace, besthps)
    print('besthps is:', besthps)

    # re-train the best model with besthps 
    bestargs = copy.deepcopy(args)
    for k, v in besthps.items(): 
        setattr(bestargs, k, v)
    main_informer.main(bestargs)


if __name__ == "__main__": 
    main() 

# command to run the file: 
# python -u Informer-aviation-hpo.py --model informer --data Aviation --root_path "./data/_data/aviation/01/" --features M --freq m --batch_size 4 --d_model 64 --d_ff 128 --train_epochs 200 --patience 70 --learning_rate 0.0001 --seq_len 24 --label_len 12 --pred_len 6 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --do_predict