import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pickle 
# from sklearn.preprocessing import StandardScaler

from utils.read_aviation import read_aviation
from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target: str='OT', scale=True, inverse=False, timeenc:int=0, freq:str='h', cols=None, **kwargs):
        # target: str - currently it doesn't accept a list, i.e., multioutput.
        # size: [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path

        # call read data 
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # num_train = 12 months/yr * 30 days/month * 24 hrs/day
        # num_vali = 4 months ^ 30 days/month * 24 hrs/day
        # total: 20 months data = (12+8)*30*24
        # border1s/2s indices meaning: 0=train, 1=val, 2=test (type_map)
        # border1s: [   0, 864s-seqlen, 11520-seqlen]
        # border2s: [8640,       11520,        14400]
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # df_data: non-dates relevant features including target feature
        # df_data varialbe is definitely of the original scale. 
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:] # remove "date" col
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]] # use only history y to predict itself

        # data: (standardized) non-dates relevant features
        if self.scale: # default true
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # data_stamp: a dataframe containg artifitial features of date
        df_stamp = df_raw[['date']][border1:border2] # fetch train/val/test dates
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # data_x: (scaled) non-dates relevant features [train/val/test]
        # data_y: (inversed) non-dates relevant features [train/val/test] (equiv. to data_x when inverse is False)
        self.data_x = data[border1:border2]
        if self.inverse: 
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.mask = (~df_raw.isna()).values[border1:border2]
    
    def __getitem__(self, index):
        # s_begin:s_end -- the encoder length 
        # r_begin:r_end -- the decoder length (= label_len + pred_len) 
        #                  which overlaps with encoder length on last label_len indices.
        # seq_x: (scaled) features of encoder length 
        # seq_y: the (scaled) features of label length, ie, the last part of encoder length, plus 
        #        the (inversed) features of pred_len.
        # x & y naming: encoder relevant & decoder relevant features indices
        # Note, the last part of seq_y, ie, the pred_len, would be the ground to calculate loss.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            # note: take last part of data_x, the scaled version, to form encoder part 
            # but take data_y, the inversed version of all relevant features but date, 
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # x_mark: dates features with encoder length 
        # y_mark: dates features with decoder length 
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        mask = self.mask[s_end:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask 
    
    def __len__(self):
        # counts the number iterations avaialbe to take seq_x, seq_y
        # no need to deduct label_len because it is always part of seq_len (the last part)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.mask = (~df_raw.isna()).values[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        mask = self.mask[s_end:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask 
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        
        # df_raw.columns final ordering: ['date', ...(other features), target feature]
        # cols can be exogenous non-date features from external or df_raw columns  
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); 
            cols.remove(self.target); 
            cols.remove('date')
        # re-ordering. Target feature must be the last position
        df_raw = df_raw[['date']+cols+[self.target]] 

        # 70:10:20 ratio for train:val:test. 
        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.mask = (~df_raw.isna()).values[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # seq_len
        if self.inverse:
            seq_y = np.concatenate([
                self.data_x[r_begin:r_begin+self.label_len], 
                self.data_y[r_begin+self.label_len:r_end]
            ], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] # seq_len 
        seq_y_mark = self.data_stamp[r_begin:r_end] # label_len + pred_len 
        mask = self.mask[s_end:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask 
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        # df_raw.columns: ['date', ...(other features), target feature]
        # # cols can be exogenous non-date features from external or df_raw columns  
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); 
            cols.remove(self.target); 
            cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2] # take the last seq_len part of the whole dataset
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:]) # seq_len + pred_len (label_len is the last part of seq_len.)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Aviation(Dataset):
    """
    Workflow: read data -> partition into train/val and intentioanlly make test==val -> 
    categorical encoding (make sure train/val same pattern) -> organize format 
    """
    def __init__(self, root_path, flag='train', size=None, 
                 features='M', data_path='train_lower.parquet.gzip', target='SUM_ophrs_act', 
                 scale=True, inverse=False, timeenc=0, freq='m', cols=None, mode='single-emb', **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len, self.label_len, self.pred_len = 24, 12, 6
        else:
            self.seq_len, self.label_len, self.pred_len = size
        # init
        assert flag in ['train', 'val']
        type_map = {'train':0, 'val':1}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.mode = mode 
        self.__read_data__()

        

    def __read_data__(self):
        
        # df_raw.columns final ordering: ['date', ...(other features), target feature]
        # cols can be exogenous non-date features from external or df_raw columns  
        self.scaler = StandardScaler()
        df_raw, hiercols = read_aviation(self.root_path, self.data_path, mode=self.mode)
        df_raw = df_raw.reset_index()
        self.hiercols = hiercols 

        # 90:10 ratio for train:val:test. # TODO: fix val size to be pred_len?
        num_train = int(len(df_raw)*0.8)
        border1s = [0, num_train-self.seq_len]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:] # remove date column
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            raise NotImplementedError()
            # df_data = df_raw[[self.target]]

        # Informer does not accept nan values
        ## iterpolation for missing values in between non-missings, then fillna(0).
        mask_notnan = (~df_data.isna()).values
        df_data[border1s[0]:border2s[0]] = df_data[border1s[0]:border2s[0]].interpolate(axis=0).fillna(0)
        df_data = df_data.fillna(0)

        # scaling 
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.mask = mask_notnan[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # seq_len
        if self.inverse:
            seq_y = np.concatenate([
                self.data_x[r_begin:r_begin+self.label_len], 
                self.data_y[r_begin+self.label_len:r_end]
            ], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] # seq_len 
        seq_y_mark = self.data_stamp[r_begin:r_end] # label_len + pred_len 
        mask = self.mask[s_end:r_end] # not affecting model prediction but only affects backward when comapring pred to gt 

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask 
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def save(self, path): 
        save_dict = {
            "root_path": self.root_path, "data_path": self.data_path, "set_type": self.set_type, 
            "size": [self.seq_len, self.label_len, self.pred_len], 
            "target": self.target, "features": self.features, 
            "scale": self.scale, "inverse": self.inverse, "timeenc": self.timeenc, 
            "freq": self.freq, "cols": self.cols, "mode": self.mode 
        }
        with open(path, 'wb') as fw: 
            pickle.dump(save_dict, fw)

class Dataset_AviationPred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='M', data_path='train_lower.parquet.gzip', target='SUM_ophrs_act', 
                 scale=True, inverse=False, timeenc=0, freq='m', cols=None, mode='single-emb', **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len, self.label_len, self.pred_len = 24, 12, 6
        else:
            self.seq_len, self.label_len, self.pred_len = size
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.mode = mode 
        self.__read_data__()

    def __read_data__(self):

        # df_raw.columns final ordering: ['date', ...(other features), target feature]
        # cols can be exogenous non-date features from external or df_raw columns  
        self.scaler = StandardScaler()
        df_raw, hiercols = read_aviation(self.root_path, self.data_path, mode=self.mode)
        df_raw = df_raw.reset_index()
        self.hiercols = hiercols 
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            raise NotImplementedError()
            # df_data = df_raw[[self.target]]

        # Informer does not accept nan values
        ## iterpolation for missing values in between non-missings + fillna(0)
        mask_notnan = (~df_data.isna()).values
        df_data = df_data.interpolate(axis=0).fillna(0)

        # scaling 
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2] # take the last seq_len part of the whole dataset
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:]) # seq_len + pred_len (label_len is the last part of seq_len.)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.mask = mask_notnan[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        mask = self.mask[s_end:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def save(self, path): 
        save_dict = {
            "root_path": self.root_path, "data_path": self.data_path, "set_type": self.set_type, 
            "size": [self.seq_len, self.label_len, self.pred_len], 
            "target": self.target, "features": self.features, 
            "scale": self.scale, "inverse": self.inverse, "timeenc": self.timeenc, 
            "freq": self.freq, "cols": self.cols, "mode": self.mode 
        }
        with open(path, 'wb') as fw: 
            pickle.dump(save_dict, fw)