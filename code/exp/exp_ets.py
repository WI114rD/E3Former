from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_FaaS_minute
from exp.exp_basic import Exp_Basic
from models.ts2vec.ncca import TSEncoder, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
import pdb
import numpy as np
from einops import rearrange
from collections import OrderedDict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from utils.augmentations import Augmenter
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from math import ceil

import torch
import statsmodels.api
import numpy as np

class ETS:
    def __init__(self, trend=None, damped_trend=False,
                 seasonal=None, seasonal_periods=None, initialization_method='estimated', horizon=60):
        '''
        指数平滑模型，接口来自statsmodels
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html#statsmodels.tsa.holtwinters.ExponentialSmoothing.fit
        https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
        '''
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.model = None
        self.training_res = None
        self.horizon = horizon
        self.predictions = None
        
    def fit_predict(self, x):
        b, t, d = x.shape
        self.predictions = None
        x = x.cpu().numpy()
        for i in range(b):  
            channels = []
            for j in range(d):
                model = statsmodels.tsa.holtwinters.ExponentialSmoothing(x[i,:,j], trend = self.trend,
                                                            damped_trend = self.damped_trend,
                                                            seasonal = self.seasonal,
                                                            seasonal_periods = self.seasonal_periods,
                                                    initialization_method = self.initialization_method)
                training_model = model.fit()
                channels.append(training_model.forecast(steps=self.horizon))
            if self.predictions is None:
                self.predictions = np.array(channels).reshape(1,self.horizon,d)
            else:
                self.predictions = np.concatenate((self.predictions, np.array(channels).reshape(1,self.horizon,d)), axis=0)
        self.predictions = torch.from_numpy(self.predictions).cuda()
        return self.predictions
    

class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.input_channels_dim = args.enc_in
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor', 'encoder', 'inv']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = ETS(horizon = args.pred_len)
        self.augmenter = None
        self.aug = args.aug
        if args.finetune:
            inp_var = 'univar' if args.features == 'S' else 'multivar'
            model_dir = str([path for path in Path(f'/export/home/TS_SSL/ts2vec/training/ts2vec/{args.data}/')
                .rglob(f'forecast_{inp_var}_*')][args.finetune_model_seed])
            state_dict = torch.load(os.path.join(model_dir, 'model.pkl'))
            for name in list(state_dict.keys()):
                if name != 'n_averaged':
                    state_dict[name[len('module.'):]] = state_dict[name]
                del state_dict[name]
            self.model[0].encoder.load_state_dict(state_dict)

    def get_augmenter(self, sample_batched):
    
        seq_len = sample_batched.shape[1]
        num_channel = sample_batched.shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
        #IF THERE IS ONLY ONE CHANNEL, WE NEED TO MAKE SURE THAT CUTOUT AND CROPOUT APPLIED (i.e. their probs are 1)
        #for extremely long sequences (such as SSC with 3000 time steps)
        #apply the cutout in multiple places, in return, reduce history crop
        elif self.input_channels_dim == 1 and seq_len>1000: 
            self.augmenter = Augmenter(cutout_length=cutout_len, cutout_prob=1, crop_min_history=0.25, crop_prob=1, dropout_prob=0.0)
            #we apply cutout 3 times in a row.
            self.augmenter.augmentations = [self.augmenter.history_cutout, self.augmenter.history_cutout, self.augmenter.history_cutout,
                                            self.augmenter.history_crop, self.augmenter.gaussian_noise, self.augmenter.spatial_dropout]
        #if there is only one channel but not long, we just need to make sure that we don't drop this only channel
        else:
            self.augmenter = Augmenter(cutout_length=cutout_len, dropout_prob=0.0)
            
    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'FaaS': Dataset_FaaS_minute,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_FaaS_minute, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        if flag == 'test':
            print(args.transfer_data_path)
            data_set = Data(
                root_path=args.root_path,
                data_path=args.transfer_data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols
            )
        else:
            print(args.data_path)
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
                cols=args.cols
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
        # self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # self.opt = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        return
    #     train_data, train_loader = self._get_data(flag='train')
    #     vali_data, vali_loader = self._get_data(flag='val')
    #     test_data, test_loader = self._get_data(flag='test')

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     time_now = time.time()

    #     train_steps = len(train_loader)
    #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    #     self.opt = self._select_optimizer()
    #     criterion = self._select_criterion()

    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()

    #     for epoch in range(self.args.train_epochs):
    #         iter_count = 0
    #         train_loss, aug_loss = [], []

    #         self.model.train()
    #         epoch_time = time.time()
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    #             iter_count += 1
                
    #             if self.augmenter == None:
    #                self.get_augmenter(batch_x)

    #             self.opt.zero_grad()
    #             pred, true = self._process_one_batch(
    #                 train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                
    #             loss = criterion(pred, true)
                
    #             train_loss.append(loss.item())

    #             if (i + 1) % 100 == 0:
    #                 print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()

    #             if self.args.use_amp:
    #                 scaler.scale(loss).backward()
    #                 scaler.step(self.opt)
    #                 scaler.update()
    #             else:
    #                 loss.backward()
    #                 self.opt.step()
                
    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)
    #         vali_loss = self.vali(vali_data, vali_loader, criterion)
    #         #test_loss = self.vali(test_data, test_loader, criterion)
    #         test_loss = 0.
    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
    #             epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    #         early_stopping(vali_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #         adjust_learning_rate(self.opt, epoch + 1, self.args)

    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))

    #     return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        preds = []
        trues = []
        preds_reverse = []
        trues_reverse = []
        start = time.time()
        maes,mses,rmses,mapes,mspes,wmapes = [],[],[],[],[],[]
        maes_reverse,mses_reverse,rmses_reverse,mapes_reverse,mspes_reverse,wmapes_reverse = [],[],[],[],[],[]

        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader): batch_y is the predicted label
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            pred_reverse = test_data.inverse_transform(pred.view(batch_y.shape)).detach().cpu()
            true_reverse = test_data.inverse_transform(true.view(batch_y.shape)).detach().cpu()
            preds_reverse.append(pred_reverse)
            trues_reverse.append(true_reverse)
            mae, mse, rmse, mape, mspe= metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)
            mae_reverse, mse_reverse, rmse_reverse, mape_reverse, mspe_reverse = metric(pred_reverse.numpy(), true_reverse.numpy())
            maes_reverse.append(mae_reverse)
            mses_reverse.append(mse_reverse)
            rmses_reverse.append(rmse_reverse)
            mapes_reverse.append(mape_reverse)
            mspes_reverse.append(mspe_reverse)
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        preds_reverse = torch.cat(preds_reverse, dim=0).numpy()
        trues_reverse = torch.cat(trues_reverse, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        MAE, MSE, RMSE, MAPE, MSPE= cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe= MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        MAE_reverse, MSE_reverse, RMSE_reverse, MAPE_reverse, MSPE_reverse = cumavg(maes_reverse), cumavg(mses_reverse), cumavg(rmses_reverse), cumavg(mapes_reverse), cumavg(mspes_reverse)
        mae_reverse, mse_reverse, rmse_reverse, mape_reverse, mspe_reverse =  MAE_reverse[-1], MSE_reverse[-1], RMSE_reverse[-1], MAPE_reverse[-1], MSPE_reverse[-1]

        end = time.time()
        exp_time = end - start
        print(preds_reverse.shape, trues_reverse.shape)
        wmape_reverse = np.sum(np.abs(preds_reverse-trues_reverse)) / np.sum(trues_reverse)
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('mape_rever:{}, mspe_rever:{}, wmape_rever:{}'.format(mape_reverse, mspe_reverse, wmape_reverse))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode =='test' and self.online != 'none':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device)
        x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x, x_mark)
        else:
            outputs = self.model(x, x_mark)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return rearrange(outputs, 'b t d -> b (t d)'), rearrange(batch_y, 'b t d -> b (t d)')
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        b, t, d = batch_y.shape
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        x = batch_x.float().to(self.device)
        x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()
        for _ in range(self.n_inner):
            outputs = self.model.fit_predict(x)
            outputs = rearrange(outputs, 'b t d -> b (t d)').float().to(self.device)
            # loss = criterion(outputs[:, :d], true[:, :d])
            # loss = criterion(outputs, true)
            # if self.aug:
            #     xa, xa_mark = self.augmenter(x, torch.ones_like(x))
            #     pred = self.model(xa)
            #     pred = rearrange(pred, 'b t d -> b (t d)').float().to(self.device)
            #     loss += self.args.loss_aug * criterion(pred, true)
            # loss.backward()
            # self.opt.step()       
            
            # self.opt.zero_grad()

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')
