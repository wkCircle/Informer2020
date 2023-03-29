from typing import Union 
import numpy as np
import pickle, os, time 

from data.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, 
    Dataset_Aviation, Dataset_AviationPred
)
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args) # auto call _build_model() to create sefl.model
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            # e_layers = 2 if informer, o.w., =[3,2,1] for default case
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, **kwargs):
        args = self.args

        # "Data" is the pointer that points to Dataset_Custom, Dataset_ETT_xxx etc.
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'Aviation': Dataset_Aviation, # LHT aviation dataset
            'custom':Dataset_Custom,
        }
        Data = data_dict[args.data] 
        # args.embed can be [timeF, fixed, learned]. Only timeF (default) corresponds to timeenc=1
        timeenc = 0 if args.embed!='timeF' else 1

        # set default config
        # flag can be train, val, test, pred
        # when train/val mode (else): shuffle and drop last non-full batch
        # when test mode: no shuffling and still drop last non-full batch
        # when pred mode: no shuffling and no any drop. (batch_szie=1)
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            if self.args.data == 'Aviation': 
                Data = Dataset_AviationPred
            else: 
                Data = Dataset_Pred
        elif flag in ['train', 'val']:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
            if self.args.data == 'Aviation': 
                drop_last = False; shuffle_flag=False;
        else: 
            raise ValueError('flag value is not allowed.')
        
        # override config by kwargs if exists 
        if len(kwargs): 
            if 'Data' in kwargs.keys(): 
                Data = kwargs['Data']
            if 'shuffle_flag' in kwargs.keys(): 
                shuffle_flag = kwargs['shuffle_flag']
            if 'drop_last' in kwargs.keys(): 
                drop_last = kwargs['drop_last']
            if 'batch_size' in kwargs.keys(): 
                batch_size = kwargs['batch_size']
            if 'freq' in kwargs.keys(): 
                freq = kwargs['freq']
    
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols, 
            mode = args.mode 
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self, ):
        criterion =  nn.MSELoss(reduction='none')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, mask) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            loss = (loss * mask.float()).sum() / mask.sum()
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        # test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) # patience for epoch
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        history = {
            'meta': self.args, 
            'train_epoch_loss':[], 'valid_epoch_loss': [], 'test_epoch_loss': [], 'criterion': criterion._get_name(), 
            'epochs': self.args.train_epochs, 'earlystop_patience': self.args.patience, 
        }
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            # _mark: related to data_stamp
            # case ETTH1 (features M): batch_x(_mark) = (32,seq_len,7 (4)); batch_y(_mark)= (32, label_len+pred_len, 7 (4))
            # where 4 is determined by time_features(timeenc (args.embed),  args.freq) that generates 
            # [Hour of day, day of week, day of month, day of year] features for hourly freq.
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                loss = (loss * mask.float()).sum() / mask.sum()
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            history['train_epoch_loss'].append(train_loss)
            history['valid_epoch_loss'].append(vali_loss)
            #history['test_epoch_loss'].append(test_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format( # Test Loss: {4:.7f}
                epoch + 1, train_steps, train_loss, vali_loss, ))#test_loss))
            # override history
            with open(path + '/history.pkl', 'wb') as fw: 
                pickle.dump(history, fw)
            # (override) check point per epoch: 
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch+1, self.args)
        # load the best 
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        best_val_loss = min(history['valid_epoch_loss'])
        idx = history['valid_epoch_loss'].index(best_val_loss)
        print("Best model found at epoch {} with train loss: {:.7f} Vali loss ever: {:.7f}".format(
            idx, history['train_epoch_loss'][idx], best_val_loss
        ))
        
        return history

    def test(self, setting: str, load=False):
        test_data, test_loader = self._get_data(flag='test')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, mask) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def insample_predict(self, setting: str, flag, load=False, save_names=['insample_prediction.npy', 'meta_info.pkl'], **kwargs):
        pred_data, pred_loader = self._get_data(flag=flag, **kwargs)
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []
        # In prediction mode, since we predict out-of-data, 
        # batch_y is only of label_len while batch_y_mark is of label_len+pred_len. 
        # len(pred_loader)==1
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, mask) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.vstack(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+save_names[0], preds)
        pred_data.save(folder_path+save_names[1])
        return


    def predict(self, setting: str, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        # note: following the default pipeline, model is not retrained on test and directly give out-of-sample
        # prediction here.
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        # In prediction mode, since we predict out-of-data, 
        # batch_y is only of label_len while batch_y_mark is of label_len+pred_len. 
        # len(pred_loader)==1
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, mask) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object: Dataset_Custom, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Args: 
            dataset_object (Dataset_Custom, Dataset_ETT_hour/minute)
            batch_x (np.ndarray): shape (batch_size, seq_len, nondate_features_dim)
            batch_y (np.ndarray): shape (batch_size, label_len+pred_len, nondate_features_dim)
            batch_x_mark (np.ndarray): shape (batch_size, seq_len, date_features_dim)
            batch_y_mask (np.ndarray): shape (batch_size, label_len+pred_len, date_features_dim)
        
        Returns: 
            outputs (np.ndarray): the model prediction of shape (batch_size, pred_len, target_dim)
            batch_y (np.ndarray): the ground truth of shape (batch_size, pred_len, target_dim), 
                which cuts off label_len from original input.
        
        Naming rules: 
            encoder length = seq_len 
            decoder length = label_len + pred_len 
            encoder & decoder overlaps on label_len which is the last part of seq_len
            _x: for encoder 
            _y: for decoder 
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0: # default action
            func_pointer = torch.zeros 
        elif self.args.padding==1:
            func_pointer = torch.ones 
        dec_inp = func_pointer([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp: # default false
            with torch.cuda.amp.autocast():
                if self.args.output_attention: # default false 
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        # re-format ground truth
        # when MS mode: take only the last column (assuming it is the target feature position), o.w., 
        # keep all features. 
        # batch_y is always cut to have only pred len at the end.
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
