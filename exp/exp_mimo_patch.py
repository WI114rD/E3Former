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

from layers.PatchTST_backbone import PatchTST_backbone, _MultiheadAttention
from layers.PatchTST_layers import series_decomp


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x, train=True):
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = F.sigmoid(x)
        return x
    
class OnlineScaler(nn.Module):
    def __init__(self, pred_len, model_num, d_model, n_heads, act=nn.ReLU()):
        super(OnlineScaler, self).__init__()
        self.model_num = model_num
        self.d_model = d_model
        n_outputs = model_num - 1
        # self.norm = nn.LayerNorm(n_inputs)
        # self.mlp = MLP(n_inputs=n_inputs, n_outputs=d_model, mlp_width=d_model, mlp_depth=1, mlp_dropout=0.1, act=act)
        self.mlp = nn.Linear(pred_len, d_model)
        self.act = act
        self.attn = _MultiheadAttention(d_model, n_heads, online=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.out_layer = nn.Linear(d_model * model_num, n_outputs)
    def forward(self, x): # x: [B, pred_len, channels, n]
        # x = torch.reshape(x, (x.shape[0], self.model_num, -1)) # [B, model_num, channels * pred_len]
        b, t, d, m = x.shape
        x = rearrange(x, 'b t d m -> (b d) m t', m=m, t=t, d=d, b=b)
        x = self.mlp(x) # [B, model_num, d_model]
        x = self.act(x)
        x, _ = self.attn(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.out_layer(x) # [B, n_outputs]
        x = rearrange(x, '(b d) n -> b d n', d=d)
        return x
    def store_grad(self):
        for name, layer in self.attn.named_modules():    
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()
    
class AttentionChoice_test(nn.Module):
    def __init__(self, n_inputs, c_out, n_outputs, model_num, d_model, n_heads, act=nn.ReLU()):
        super(AttentionChoice_test, self).__init__()
        self.model_num = model_num
        self.d_model = d_model
        self.norm = nn.LayerNorm(n_inputs)
        # self.mlp = MLP(n_inputs=n_inputs, n_outputs=d_model, mlp_width=d_model, mlp_depth=1, mlp_dropout=0.1, act=act)
        self.mlp = nn.Linear(n_inputs, d_model)
        self.act = act
        self.attn = _MultiheadAttention(n_inputs, n_heads)
        self.flatten = nn.Flatten(start_dim=1)
        self.out_layer = nn.Linear( n_inputs * n_outputs, n_outputs)
    def forward(self, x): # x: [B, model_num * channels * pred_len]
        x = torch.reshape(x, (x.shape[0], self.model_num, -1)) # [B, model_num, channels * pred_len]
        # x = self.mlp(x) # [B, model_num, d_model]
        # x = self.act(x)
        q_x = x[:,-1:,:]
        k_x = x[:,:-1,:]
        x = self.attn(q_x, k_x, k_x)[0]
        x = self.act(x)
        x = self.flatten(x)
        x = self.out_layer(x) # [B, n_outputs]
        return x

class net(nn.Module):
    def __init__(self, configs, device='cuda', max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        # patch_len = configs.patch_len
        patch_lens = configs.patch_lens
        strides = configs.strides
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        self.channel_cross = configs.channel_cross
        
        

        self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_lens=patch_lens, strides=strides, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose, channel_cross=self.channel_cross, online=True, **kwargs)
    
        self.to(device)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x

    def forward_weight(self, x, weight): # x: [Batch, Input length, Channel], weight: [Batch, Channel, num_models]
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        outputs = self.model(x) 
        z = torch.stack(outputs, -1).detach()
        z = weight*z
        z = z.sum(-1)  # z: [Batch, Input length, Channel]
        return z, outputs

    def store_grad(self):
        for name, layer in self.model.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()

class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        # torch.manual_seed(3407)
        torch.manual_seed(2024)
        self.args = args
        self.input_channels_dim = args.enc_in
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor', 'encoder', 'inv']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device = self.device)
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


        self.num_outputs = len(args.patch_lens)
        self.scalerlayer = OnlineScaler(pred_len=args.pred_len, model_num=self.num_outputs + 1, d_model=16, n_heads=args.n_heads).to(self.device)
        # self.decision = MLP(n_inputs=(args.c_out * args.pred_len) * (self.num_outputs+1), n_outputs=self.num_outputs, mlp_width=128, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh()).to(self.device)
        # self.onlinefixer = MLP(n_inputs=args.pred_len, n_outputs=2, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh()).to(self.device)
        self.weight = torch.randn((args.c_out, self.num_outputs), device = self.device)
        self.scaler = torch.zeros((args.c_out, self.num_outputs), device = self.device)
        print(sum(p.numel() for p in self.scalerlayer.parameters()))
        # self.fixer = torch.zeros(2, device = self.device)
        # self.fixer[0], self.fixer[1]=0., 1.
        self.softmax_w = nn.Softmax(dim=-1)
        self.weight.requires_grad = True

    def get_augmenter(self, sample_batched):
    
        seq_len = sample_batched.shape[1]
        num_channel = sample_batched.shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
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
            'custom': Dataset_Custom,
            'FaaS': Dataset_FaaS_minute,
            'FaaS_medium': Dataset_FaaS_minute,
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
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        self.opt_w = optim.Adam([self.weight], lr=self.args.learning_rate_w)
        self.opt_bias = optim.Adam(self.scalerlayer.parameters(), lr=self.args.learning_rate_bias)
        # self.opt_fixer = optim.Adam(self.onlinefixer.parameters(), lr=self.args.learning_rate_fixer)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss, aug_loss = [], []
            loss_ws, loss_biass = [], []


            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                if self.augmenter == None:
                   self.get_augmenter(batch_x)

                self.opt.zero_grad()
                _, pred, true, loss_w, loss_bias = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = 0.0
                for output in pred:
                    single_loss = criterion(output, true)
                    loss += single_loss

                train_loss.append(loss.item())
                loss_ws.append(loss_w)
                loss_biass.append(loss_bias)

                # if self.aug > 0:
                #     loss_aug = 0
                #     for i in range(self.aug):
                #         batch_xa, _ = self.augmenter(batch_x.float().to(self.device), torch.ones_like(batch_x).to(self.device))
                #         pred, true = self._process_one_batch(train_data, batch_xa, batch_y, batch_x_mark, batch_y_mark)
                #         loss_aug += criterion(pred, true)
                #     loss_aug /= self.aug
                #     loss += self.args.loss_aug * loss_aug
                # else:
                #     loss_aug = torch.tensor(0)
                
                # aug_loss.append(loss_aug.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                self.model.store_grad()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            loss_ws = np.average(loss_ws)
            loss_biass = np.average(loss_biass)
            # aug_loss = np.average(aug_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} loss_ws:  {4:.4f} loss_bias:  {5:.4f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, loss_ws, loss_biass))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            output, pred, true, _, _ = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            loss = criterion(output.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):

        self.weight = torch.randn((self.args.c_out,self.num_outputs), device = self.device)
        self.scaler = torch.zeros((self.args.c_out,self.num_outputs), device = self.device)
        self.weight.requires_grad = True
        self.opt_w = optim.Adam([self.weight], lr=self.args.learning_rate_w)

        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        if self.online == 'regressor':
            if self.model.decomposition:
                for p in self.model.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.backbone.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.model.backbone.parameters():
                    p.requires_grad = False 
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
        elif self.online == 'inv':
            if self.model.decomposition:
                for p in self.model.model_res.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.head.parameters():
                    p.requires_grad = False 
                for p in self.model.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_trend.head.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.model.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model.head.parameters():
                    p.requires_grad = False 
        elif self.online == 'encoder':
            if self.model.decomposition:
                for p in self.model.model_trend.head.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.head.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.model.head.parameters():
                    p.requires_grad = False 
        
        preds = []
        trues = []
        preds_reverse = []
        trues_reverse = []
        start = time.time()
        maes,mses,rmses,mapes,mspes,wmapes = [],[],[],[],[],[]
        maes_reverse,mses_reverse,rmses_reverse,mapes_reverse,mspes_reverse,wmapes_reverse = [],[],[],[],[],[]
        # total_weights = torch.zeros((14*4), 168).to(self.device)
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader): batch_y is the predicted label
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true, weights_iter = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            # flat_weights = weights_iter.contiguous().view(-1)
            # total_weights[:, i] = flat_weights  
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            pred_reverse = test_data.inverse_transform(pred).detach().cpu()
            true_reverse = test_data.inverse_transform(true).detach().cpu()
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
        np.save('/mnt/bn/jiadong2/Online_results/' + 'preds_reverse.npy', preds_reverse)
        np.save('/mnt/bn/jiadong2/Online_results/' + 'tures_reverse.npy', trues_reverse)
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('mape_rever:{}, mspe_rever:{}, wmape_rever:{}'.format(mape_reverse, mspe_reverse, wmape_reverse))

        # np.save('weights_128.npy', np.array(total_weights.detach().cpu()))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode =='test' and self.online != 'none':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        # if self.args.use_amp:
        #     with torch.cuda.amp.autocast():
        #         outputs = self.model(x)
        # else:
        #     outputs = self.model(x)
        b, t, d = batch_y.shape

        loss1 = self.softmax_w(self.weight)  # the weight of cross-time forecaster
        loss1 = loss1.view(1,1,d,-1)
        loss1 = loss1.repeat(b, t, 1, 1)
        output, output_list= self.model.forward_weight(x, loss1)
        # self.fixer = self.onlinefixer(x.permute(0,2,1))
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        b, t, d = batch_y.shape
        criterion = self._select_criterion()
                
        loss_w = criterion(output, batch_y)

        loss_w.backward()
        self.opt_w.step()   
        self.opt_w.zero_grad()   
        

        
        # inputs_decision = torch.cat([loss1*y1_w, (1-loss1)*y2_w, true_w], dim=1)
        loss1 = self.softmax_w(self.weight) # [n]
        loss1 = loss1.view(1, 1, d, -1)
        true_w = batch_y.view(b, t, d).detach() # [B, T, D]
        cat_outputs = [output_list[i].detach() * loss1[:,:,:,i] for i in range(len(output_list))] 
        # weighted_flattened = [torch.flatten(t, start_dim=1) for t in weighted_flattened]
        cat_outputs.append(true_w)
        # cat_outputs = cat_outputs.unsqueeze(dim=-1)
        inputs_decision = torch.stack(cat_outputs, dim=-1) # [B, T, D, n]

        self.scaler = self.scalerlayer(inputs_decision)
        weight = self.softmax_w(self.weight + self.scaler)
        bias_output_list = []
        for _output in output_list:
            bias_output_list.append(_output.detach())
        output_tensors = torch.stack(bias_output_list, -1)
        weight = weight.view(b,1,d,-1)
        weight = weight.repeat(1, t, 1, 1)
        bias_outputs = weight*output_tensors
        bias_output = bias_outputs.sum(-1)
        true_w = batch_y.detach()
        loss_bias = criterion(bias_output, true_w)
        loss_bias.backward()
        self.opt_bias.step()
        self.scalerlayer.store_grad()
        self.opt_bias.zero_grad()   
        

        # return output, output_list, batch_y, loss_w.detach().cpu().item(), loss_bias.detach().cpu().item()
        return output, output_list, batch_y, loss_w.detach().cpu().item(), loss_w.detach().cpu().item()
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        b, t, d = batch_y.shape
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        for _ in range(self.n_inner):
            # if self.args.use_amp:
            #     with torch.cuda.amp.autocast():
            #         outputs = self.model(x)
            # else:
            #     outputs = self.model(x)
            # outputs = rearrange(outputs, 'b t d -> b (t d)').float().to(self.device)
            # loss = criterion(outputs[:, :d], true[:, :d])
            loss1 = self.softmax_w(self.weight + self.scaler)
            weights_epoch = loss1
            loss1 = loss1.view(b,1,d,-1)
            loss1 = loss1.repeat(1, t, 1, 1)
            outputs, output_list= self.model.forward_weight(x, loss1)
            loss = 0.0
            for output in output_list:
                single_loss = criterion(output, batch_y)
                loss += single_loss
            
            loss.backward()
            self.opt.step()       
            self.model.store_grad()
            self.opt.zero_grad()

            loss1 = self.softmax_w(self.weight) # [n]
            loss1 = loss1.view(1, 1, d, -1)
            true_w = batch_y.view(b, t, d).detach() # [B, T, D]
            cat_outputs = [output_list[i].detach() * loss1[:,:,:,i] for i in range(len(output_list))] 
            # weighted_flattened = [torch.flatten(t, start_dim=1) for t in weighted_flattened]
            cat_outputs.append(true_w)
            # cat_outputs = cat_outputs.unsqueeze(dim=-1)
            inputs_decision = torch.stack(cat_outputs, dim=-1) # [B, T, D, n]

            # loss1 = self.softmax_w(self.weight)
            # true_w = batch_y.view(b, t * d).detach()
            # weighted_flattened = [output_list[i].detach() * loss1[i] for i in range(len(output_list))]
            # weighted_flattened = [torch.flatten(t, start_dim=1) for t in weighted_flattened]
            # weighted_flattened.append(true_w)
            # inputs_decision = torch.cat(weighted_flattened, dim=-1)  

            self.scaler = self.scalerlayer(inputs_decision)
            weight = self.softmax_w(self.weight + self.scaler)
            bias_output_list = []
            for _output in output_list:
                bias_output_list.append(_output.detach())
            output_tensors = torch.stack(bias_output_list, -1)

            weight = weight.view(b,1,d,-1)
            weight = weight.repeat(1, t, 1, 1)
            bias_outputs = weight*output_tensors
            bias_output = bias_outputs.sum(-1)
            true_w = batch_y.detach()
            loss_bias = criterion(bias_output, true_w)
            loss_bias.backward()
            self.opt_bias.step()   
            self.scalerlayer.store_grad()
            self.opt_bias.zero_grad()   

            loss1 = self.softmax_w(self.weight)  

            output_w = torch.stack(output_list, -1).detach()
            weight = loss1.view(1,1,d,-1)
            weight = weight.repeat(b, t, 1, 1)
            output_w = weight*output_w
            output_w = output_w.sum(-1)
            loss_w = criterion(output_w, batch_y)
            loss_w.backward()
            self.opt_w.step()   
            self.opt_w.zero_grad()   

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, batch_y, weights_epoch

