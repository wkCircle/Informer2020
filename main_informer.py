import argparse
import os
import torch
import random 
from pathlib import Path 
import numpy as np 
import pandas as pd 

from exp.exp_informer import Exp_Informer
from utils.read_aviation import read_aviation 
from data.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, 
    Dataset_Aviation, Dataset_AviationPred
)

def args_parse(verbose=False):
    # arg parser
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data name as key str to match parser')
    parser.add_argument('--root_path', type=str, default='./data/_data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file name with extension. Note: --data takes precedence and might override this setting')    
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task. Note: --data takes precedence and might override this setting')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size. Note: --data takes precedence and might override this setting')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size. Note: --data takes precedence and might override this setting')
    parser.add_argument('--c_out', type=int, default=7, help='output size. Note: --data takes precedence and might override this setting')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience for number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',help='experiment description to add as the suffix of checkpoint title name')
    parser.add_argument('--loss', type=str, default='mse',help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    args = parser.parse_args() # type: argparse.Namespace

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]}, # 7 incdlues target but excludes date column
        'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
        'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
        'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
        'Aviation': {'data': 'train_lower.parquet.gzip', 'T':'SUM_ophrs_act', 'M': [], 'mode': 'single-emb'} # mode: ['single-emb', 'multi-emb']
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data'] # overrides data_path via data setting in data_parser.
        args.target = data_info['T']       # overrides target
        if data_info[args.features] == [] and args.data == "Aviation": 
            args.mode = data_info['mode']
            tmp, _ = read_aviation(args.root_path, args.data_path, args.mode)
            ncols = tmp.columns.size
            data_info['M'] = [ncols, ncols, ncols]
            args.des = args.des + "-" + args.mode 
        args.enc_in, args.dec_in, args.c_out = data_info[args.features] # get either keys: [M, S, MS] depending on --features option

    # s_layer and model type (informer, informerstack) together determine if e_layer should be overridden or not. (see Exp_Informer)
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')] # default=[3,2,1]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    
    if verbose: 
        print('Args in experiment:')
        print(args)
    return args 

def get_setting(args, ii: int): 
    # setting record of experiments
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                args.model, args.data, Path(args.root_path).name, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)
    return setting 

def main(args): 
    # control randomness 
    seed = 42 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    Exp = Exp_Informer
    for ii in range(args.itr):
        # setting record of experiments
        setting = get_setting(args, ii)

        exp = Exp(args) # set experiments: independent of ii at the initialization stage
        best_model_path = Path(args.checkpoints) / setting / "checkpoint.pth"
        if best_model_path.is_file(): 
            print('>>>>>>>Loading model: {}>>>>>>>>>>>>>>>>'.format(setting))
            exp.model.load_state_dict(torch.load(best_model_path))
        else: 
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
        
        # Aviation does not have testing case but directly out-of-sample forecast in each rolling backtesting iteration.
        if args.data != "Aviation": 
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.insample_predict(setting, load=True, flag='train', save_names=['train_prediction.npy', 'train_meta.pkl'], 
                                shuffle_flag=False, drop_last=False, batch_size=args.batch_size, freq=args.freq)
            exp.insample_predict(setting, load=True, flag='val', save_names=['val_prediction.npy', 'val_meta.pkl'], 
                                shuffle_flag=False, drop_last=False, batch_size=args.batch_size, freq=args.freq)
            # exp.insample_predict(setting, load=True, save_name='test_prediction.npy')
            exp.predict(setting, load=True)

        torch.cuda.empty_cache()


if __name__ == "__main__": 
    args = args_parse(verbose=True)
    main(args)