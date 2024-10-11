__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from itertools import chain
#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from models.ts2vec.ncca import TSEncoderTime, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.fsnet import TSEncoder, GlobalLocalMultiscaleTSEncoder

from itertools import chain


# Cell
class PatchTST_backbone_TCN_Mix(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, tcn_output_dim=320, tcn_layer=5,tcn_hidden=64, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        
        self.TCN_backbone = TSEncoderTime(input_dims=context_window,
                             output_dims=tcn_output_dim,  # standard ts2vec backbone value
                             hidden_dims=tcn_hidden, # standard ts2vec backbone value
                             depth=tcn_layer)

        # Head
        self.head_nf = d_model * patch_num + tcn_output_dim
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head_TCN(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
  
        z_tcn = self.TCN_backbone(z.permute(0,2,1))
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        # z = torch.zeros_like(z)
        z = self.head(z, z_tcn)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
            
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )

class PatchTST_backbone_TCN_MOE(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'weighted', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, tcn_output_dim=320, tcn_layer=5,tcn_hidden=64, online=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.n_vars = c_in
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        
        self.TCN_backbone = TSEncoderTime(input_dims=context_window,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=tcn_hidden, # standard ts2vec backbone value
                             depth=tcn_layer)
        self.w_gate = nn.Linear(context_window, 2, bias=False)
        # self.w_gate = nn.ModuleList()
        # for i in range(self.n_vars):
        #     self.w_gate.append(nn.Linear(context_window, 2))
            
        self.softmax = nn.Softmax(-1)
        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs

        self.head_patch = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        self.head = Flatten_Head_Gate(self.individual, self.n_vars, 320, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        # gates = []
        # for i in range(self.n_vars):   
        #     gate = self.w_gate[i](z[:,i,:])             
        #     gate = self.softmax(gate)
        #     gates.append(gate)
        # gates = torch.stack(gates, dim=1) 
        gates = self.w_gate(z)
        gates = self.softmax(gates)
        
        z_tcn = self.TCN_backbone(z.permute(0,2,1))
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        # z = torch.zeros_like(z)
        z = self.head_patch(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        
        z = self.head(z, z_tcn, gates)               
        return z
            
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )
# Cell
class PatchTST_backbone_TCN(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, tcn_output_dim=320, tcn_layer=5,tcn_hidden=64, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        
        self.TCN_backbone = TSEncoderTime(input_dims=context_window,
                             output_dims=tcn_output_dim,  # standard ts2vec backbone value
                             hidden_dims=tcn_hidden, # standard ts2vec backbone value
                             depth=tcn_layer)

        # Head
        self.head_nf = d_model * patch_num + tcn_output_dim
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head_TCN(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
  
        z_tcn = self.TCN_backbone(z.permute(0,2,1))
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        # z = torch.zeros_like(z)
        z = self.head(z, z_tcn)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
            
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )

# Cell
# class PatchTST_backbone(nn.Module):
#     def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
#                  n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
#                  d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
#                  padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
#                  pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
#                  pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False, channel_cross=False,
#                  verbose:bool=False, mix_tcn=False,**kwargs):
        
#         super().__init__()
        
#         # RevIn
#         self.revin = revin
#         if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch = padding_patch
#         patch_num = int((context_window - patch_len)/stride + 1)
#         if padding_patch == 'end': # can be modified to general case
#             self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
#             patch_num += 1
        
#         # Backbone 
#         self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
#                                 n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
#                                 attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
#                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
#                                 pe=pe, learn_pe=learn_pe, verbose=verbose, mix_tcn=mix_tcn, **kwargs)

#         # Head
#         self.channel_cross = channel_cross
#         if channel_cross:
#             self.head_nf = d_model * c_in # use the represetations of the last patch for forecasting
#             target_window = c_in * target_window
#         else:
#             self.head_nf = d_model * patch_num
#         self.n_vars = c_in
#         self.pretrain_head = pretrain_head
#         self.head_type = head_type
#         self.individual = individual

#         if self.pretrain_head: 
#             self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
#         elif head_type == 'flatten': 
#             self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
#     def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
#         # norm
#         if self.revin: 
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'norm')
#             z = z.permute(0,2,1)
#         bsz, nvars, seq_len = z.shape
#         # do patching
#         if self.padding_patch == 'end':
#             z = self.padding_patch_layer(z)
#         z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
#         z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
#         # model
#         z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
#         if self.channel_cross:
#             z = z.permute(0, 3, 1, 2)                             # x: [bs x patch_len x nvars x patch_num]     
#             z = z[:,-1,:,:]
#         z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
#         if self.channel_cross:
#             z = z.view(bsz, nvars, -1)
#         # denorm
#         if self.revin: 
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'denorm')
#             z = z.permute(0,2,1)
#         return z
    
#     def create_pretrain_head(self, head_nf, vars, dropout):
#         return nn.Sequential(nn.Dropout(dropout),
#                     nn.Conv1d(head_nf, vars, 1)
#                     )

class Positional_Embedding(nn.Module):
    def __init__(self, pe, learn_pe, q_len, d_model):
        super().__init__()
        self.PositionalEmbedding = positional_encoding(pe, learn_pe, q_len, d_model)
    def forward(self, x):
        return self.PositionalEmbedding

# Cell
class PatchTST_backbone_origin(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False, channel_cross=False,
                 verbose:bool=False, mix_tcn=False,**kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder_origin(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, mix_tcn=mix_tcn, **kwargs)

        # Head
        self.channel_cross = channel_cross
        if channel_cross:
            self.head_nf = d_model * c_in # use the represetations of the last patch for forecasting
            target_window = c_in * target_window
        else:
            self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        bsz, nvars, seq_len = z.shape
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        if self.channel_cross:
            z = z.permute(0, 3, 1, 2)                             # x: [bs x patch_len x nvars x patch_num]     
            z = z[:,-1,:,:]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        if self.channel_cross:
            z = z.view(bsz, nvars, -1)
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_lens:list, strides:list, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False, channel_cross=False,
                 verbose:bool=False, mix_tcn=False, online=False,**kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # self.head_nfs = [
        #     configs.d_model *
        #     int((configs.seq_len - patch_len) / stride + 2)
        #     for patch_len, stride in zip(self.patch_sizes, strides)
        # ]
        
        # Patching
        self.patch_lens = patch_lens
        self.strides = strides
        self.padding_patch = padding_patch
        patch_nums = [int((context_window - patch_len)/stride + 1) for patch_len, stride in zip(patch_lens, strides)]
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = [nn.ReplicationPad1d((0, stride)) for stride in self.strides]
            patch_nums = [patch_num + 1 for patch_num in patch_nums]
        
        # Input encoding
        # q_len = patch_num

        self.W_P = nn.ModuleList([
            nn.Linear(patch_len, d_model)
            for patch_len in self.patch_lens
        ])
        # self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        # self.seq_len = q_len

        # Positional encoding
        # self.W_pos = []
        # for i, patch_len in enumerate(patch_nums):
        #     q_len = patch_len
        #     self.W_pos.append(positional_encoding(pe, learn_pe, q_len, d_model))
        self.W_pos = nn.ModuleList([
            Positional_Embedding(pe, learn_pe, patch_len, d_model)
            for patch_len in patch_nums
        ])
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        # Backbone 
        self.backbone = TSTiEncoder(max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, mix_tcn=mix_tcn, online=online, **kwargs)

        # Head
        self.channel_cross = channel_cross
        if channel_cross:
            self.head_nf = d_model * c_in # use the represetations of the last patch for forecasting
            target_window = c_in * target_window
        else:
            self.head_nfs = [
            d_model * patch_num
            for patch_num in patch_nums
        ]
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        # if self.pretrain_head: 
        #     self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        # elif head_type == 'flatten': 
        #     self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        self.heads = nn.ModuleList([
            Flatten_Head(self.individual, self.n_vars, head_nf, target_window, head_dropout=head_dropout)
            for head_nf in self.head_nfs
        ])
    def forward(self, z): # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        bsz, nvars, seq_len = z.shape
        outputs = []
        # output = None
        z_model = z
        for i, patch_len in enumerate(self.patch_lens):
            # do patching
            if self.padding_patch == 'end':
                z = self.padding_patch_layer[i](z_model)
            z = z.unfold(dimension=-1, size=patch_len, step=self.strides[i])                   # z: [bs x nvars x patch_num x patch_len]
            # z = z.permute(0,1,3,2)     
            
            # z = z.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
            x = self.W_P[i](z)                                                          # x: [bs x nvars x patch_num x d_model]

            u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]

            u = self.dropout(u + self.W_pos[i](u))                                         # u: [bs * nvars x patch_num x d_model]
                                                            # z: [bs x nvars x patch_len x patch_num]
            
            # model
            z = self.backbone(u, nvars)                                                                # z: [bs x nvars x d_model x patch_num]
            if self.channel_cross:
                z = z.permute(0, 3, 1, 2)                             # x: [bs x patch_len x nvars x patch_num]     
                z = z[:,-1,:,:]
            z = self.heads[i](z)                                                                    # z: [bs x nvars x target_window] 
            if self.channel_cross:
                z = z.view(bsz, nvars, -1)
            # denorm
            if self.revin: 
                z = z.permute(0,2,1)
                z = self.revin_layer(z, 'denorm')
                z = z.permute(0,2,1)

            z = z.permute(0, 2, 1) # [bs x target_window x nvars]
            outputs.append(z)
            # if output is None:
            #     output = z
            # else:
            #     output = output + z

        # output = output / len(self.patch_lens)
        return outputs
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head_TCN(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x, z_tcn):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = torch.cat([x, z_tcn[:,i,:]], dim=1)
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = torch.cat([x, z_tcn], dim=-1)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class Flatten_Head_Gate(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        self.linear0 = nn.Linear(nf, target_window)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward_(self, x, z_tcn, gate):                                 # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        score = gate
        g1, g2 = score[:,:,0].unsqueeze(-1), score[:,:,1].unsqueeze(-1)
        g1 = g1.repeat(1, 1, x.shape[-1])
        g2 = g2.repeat(1, 1, z_tcn.shape[-1])
        x = x * g1 + z_tcn * g2
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
    def forward(self, x, z_tcn, gate):                                 # x: [bs x nvars x d_model x patch_num]
        r_x = x
        r_tcn = self.linear0(z_tcn)
        score = gate
        g1, g2 = score[:,:,0].unsqueeze(-1), score[:,:,1].unsqueeze(-1)
        g1 = g1.repeat(1, 1, r_x.shape[-1])
        g2 = g2.repeat(1, 1, r_tcn.shape[-1])
        res = r_x * g1 + r_tcn * g2
        
        return res

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
    
class TSTiEncoder_origin(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, mix_tcn=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder_origin(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, mix_tcn=mix_tcn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self,max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, mix_tcn=False, online=False, **kwargs):
        
        
        super().__init__()
        
        # self.patch_num = patch_num
        # self.patch_len = patch_len

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, mix_tcn=mix_tcn, online=online)

        
    def forward(self, x, n_vars) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        # n_vars = x.shape[1]
        # Input encoding
        # x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        # x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        # u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        # u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(x)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    

class TSTEncoder_origin(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, mix_tcn=False):
        super().__init__()

        if mix_tcn:
            TSTEncoderLayer_ = TSTandTCNEncoderLayer
        else:
            TSTEncoderLayer_ = TSTEncoderLayer_origin
        self.layers = nn.ModuleList([TSTEncoderLayer_(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, mix_tcn=False, online=False):
        super().__init__()

        if mix_tcn:
            TSTEncoderLayer_ = TSTandTCNEncoderLayer
        else:
            TSTEncoderLayer_ = TSTEncoderLayer
        self.layers = nn.ModuleList([TSTEncoderLayer_(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn, online=online) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class TSTEncoderLayer_origin(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False, online=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, online=online)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class TSTandTCNEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        from models.ts2vec.ncca_ import DilatedConvEncoder
        self.conv = DilatedConvEncoder(
                d_model,
                [d_model],
                kernel_size=3, gamma=0.9
            )
        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        src = src.transpose(1, 2)
        src = self.conv(src)
        src = src.transpose(1, 2)
        if self.res_attention:
            return src, scores
        else:
            return src

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, online=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        if online:
            self.W_Q = OnlineLinear(d_model, d_k * n_heads, bias=qkv_bias)
            self.W_K = OnlineLinear(d_model, d_k * n_heads, bias=qkv_bias)
            self.W_V = OnlineLinear(d_model, d_v * n_heads, bias=qkv_bias)
        else:
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
            self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
            self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        # if online:
        #     self.to_out = nn.Sequential(OnlineLinear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        # else:
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


def normalize(W):
    W_norm = torch.norm(W)
    W_norm = torch.relu(W_norm - 1) + 1
    W = W/ W_norm
    return W

class OnlineLinear(nn.Module):
    def __init__(self, in_features, out_features, gamma=0.9, bias=True):
        super().__init__()
        if out_features > in_features:
            if out_features % in_features != 0:
                raise ValueError("out_features must be divisible by in_features")
            self.conv1d = SamePadConv(in_features, out_features, kernel_size=1, gamma=gamma, bias=bias)
        else:
            self.conv1d = SamePadConv_v2(in_features, out_features, kernel_size=1, gamma=gamma, bias=bias)
    def forward(self, x): # [B, L, C]
        x = x.transpose(-2,-1)
        x = self.conv1d(x)
        x = x.transpose(-2,-1)
        return x
    
class SamePadConv_v3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, gamma=0.9, bias=True):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=False
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]),requires_grad=True)
        self.padding=padding
        self.dilation = dilation
        self.kernel_size= kernel_size
        self.need_bias = bias
        
        self.grad_dim, self.shape = [], []
        for p in self.conv.parameters():
            self.grad_dim.append(p.numel())
            self.shape.append(p.size())
        self.dim = sum(self.grad_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_features= out_channels

        self.n_chunks = out_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(out_channels*kernel_size// self.n_chunks)
        
        self.grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        self.f_grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        nh=64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, 1)
        self.calib_f = nn.Linear(nh, 1)
        dim = self.n_chunks * (self.chunk_out_d + 2)
        self.W = nn.Parameter(torch.randn(self.chunk_out_d + 2, out_channels), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = normalize(self.W.data)
        
        
        #self.calib_w = torch.nn.Parameter(torch.ones(out_channels, in_channels,1), requires_grad = True)
        #self.calib_b = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad = True)
        #self.calib_f = torch.nn.Parameter(torch.ones(1,out_channels,1), requires_grad = True)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.tau = 0
    def ctrl_params(self):
        c_iter = chain(self.controller.parameters(), self.calib_w.parameters(), 
                self.calib_b.parameters(), self.calib_f.parameters())
        for p in c_iter:
            yield p

    def store_grad(self):
        #print('storing grad')
        grad = self.conv.weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        self.f_grads = self.f_gamma * self.f_grads + (1-self.f_gamma) * grad
        if not self.training: 
            e = self.cos(self.f_grads, self.grads)
            
            if e < -self.tau:
                self.trigger = 1
        self.grads = self.gamma * self.grads + (1-self.gamma) * grad
        
    def fw_chunks(self):
        x = self.grads.view(self.n_chunks, -1) # [n_chunks, chunk_in_d]
        rep = self.controller(x) # [n_chunks, nh]
        w = self.calib_w(rep) # [n_chunks, chunk_out_d]
        b = self.calib_b(rep) # [n_chunks, 1]
        f = self.calib_f(rep) # [n_chunks, 1]
        q = torch.cat([w, b, f], dim=-1) # [n_chunks, chunk_out_d + 2]
        if not hasattr(self, 'q_ema'):
            setattr(self, 'q_ema', torch.zeros(*q.size()).float().cuda())
        else:
            self.q_ema = self.f_gamma * self.q_ema + (1-self.f_gamma)*q
            q = self.q_ema
        if self.trigger == 1:
            dim = w.size(0) # n_chunks
            self.trigger = 0
            # read      
            att = q @ self.W # [n_chunks, out_channels]
            att = F.softmax(att/0.5)
            att_out = att @ self.W.transpose(0, 1) # [n_chunks, 3]
            
            old_w = att_out.reshape(-1)

            ll = torch.split(old_w, dim)
            nw,nb, nf = w.size(1), b.size(1), f.size(1)
            o_w, o_b, o_f = torch.cat(*[ll[:nw]]), torch.cat(*[ll[nw:nw+nb]]), torch.cat(*[ll[-nf:]])
            
            try:
                w = self.tau * w + (1-self.tau)*o_w.view(w.size())
                b = self.tau * b + (1-self.tau)*o_b.view(b.size())
                f = self.tau * f + (1-self.tau)*o_f.view(f.size())
            except:
                pdb.set_trace()
        f = f.view(-1).unsqueeze(0).unsqueeze(2)
       
        return w.unsqueeze(1) ,b.view(-1),f

    def forward(self, x):
        w,b,f = self.fw_chunks()
        d0, d1 = self.conv.weight.shape[1:]
        
        cw = self.conv.weight * w
        #cw = self.conv.weight
        try:
            if self.need_bias:
                conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation, bias = self.bias * b)
            else:
                conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation)
            out =  f * conv_out
        except: pdb.set_trace()
        return out

    def representation(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

    def _forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, gamma=0.9):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def ctrl_params(self):  
        c_iter = chain(self.conv1.controller.parameters(), self.conv1.calib_w.parameters(), 
                self.conv1.calib_b.parameters(), self.conv1.calib_f.parameters(),
                self.conv2.controller.parameters(), self.conv2.calib_w.parameters(), 
                self.conv2.calib_b.parameters(), self.conv2.calib_f.parameters())

        return c_iter 
       


    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1), gamma=gamma
            )
            for i in range(len(channels))
        ])
    def ctrl_params(self):
        ctrl = []
        for l in self.net:
            ctrl.append(l.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p
    def forward(self, x):
        return self.net(x)


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, gamma=0.9, bias=True):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=False
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]),requires_grad=True)
        self.padding=padding
        self.dilation = dilation
        self.kernel_size= kernel_size
        self.need_bias = bias
        
        self.grad_dim, self.shape = [], []
        for p in self.conv.parameters():
            self.grad_dim.append(p.numel())
            self.shape.append(p.size())
        self.dim = sum(self.grad_dim)
        self.in_channels = in_channels
        self.out_features= out_channels

        self.n_chunks = in_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(in_channels*kernel_size// self.n_chunks)
        
        self.grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        self.f_grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        nh=64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, out_channels//in_channels)
        self.calib_f = nn.Linear(nh, out_channels//in_channels)
        dim = self.n_chunks * (self.chunk_out_d + 2 * out_channels // in_channels)
        self.W = nn.Parameter(torch.empty(dim, 32), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = normalize(self.W.data)
        
        
        #self.calib_w = torch.nn.Parameter(torch.ones(out_channels, in_channels,1), requires_grad = True)
        #self.calib_b = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad = True)
        #self.calib_f = torch.nn.Parameter(torch.ones(1,out_channels,1), requires_grad = True)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.tau = 0.75
    def ctrl_params(self):
        c_iter = chain(self.controller.parameters(), self.calib_w.parameters(), 
                self.calib_b.parameters(), self.calib_f.parameters())
        for p in c_iter:
            yield p

    def store_grad(self):
        #print('storing grad')
        grad = self.conv.weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        self.f_grads = self.f_gamma * self.f_grads + (1-self.f_gamma) * grad
        if not self.training: 
            e = self.cos(self.f_grads, self.grads)
            
            if e < -self.tau:
                self.trigger = 1
        self.grads = self.gamma * self.grads + (1-self.gamma) * grad
        
    def fw_chunks(self):
        x = self.grads.view(self.n_chunks, -1) # [n_chunks, chunk_in_d]
        rep = self.controller(x) # [n_chunks, nh]
        w = self.calib_w(rep) # [n_chunks, chunk_out_d]
        b = self.calib_b(rep) # [n_chunks, out_channels // in_channels]
        f = self.calib_f(rep) # [n_chunks, out_channels // in_channels]
        q = torch.cat([w.view(-1), b.view(-1), f.view(-1)]) # [n_chunks * (chunk_out_d + 2 * out_channels // in_channels )]
        if not hasattr(self, 'q_ema'):
            setattr(self, 'q_ema', torch.zeros(*q.size()).float().cuda())
        else:
            self.q_ema = self.f_gamma * self.q_ema + (1-self.f_gamma)*q
            q = self.q_ema
        if self.trigger == 1:
            dim = w.size(0) # n_chunks
            self.trigger = 0
            # read
            
            att = q @ self.W   # [32]

            att = F.softmax(att/0.5)
            
            v, idx = torch.topk(att, 2)
            ww = torch.index_select(self.W, 1, idx) # [dim, 2]
            idx = idx.unsqueeze(1).float() # [2, 1]
            old_w = ww @ idx # [dim, 1]
            # write memory
            s_att = torch.zeros(att.size(0)).cuda() # [32]
            s_att[idx.squeeze().long()] = v.squeeze()
            W = old_w @ s_att.unsqueeze(0) # [dim,1]
            mask = torch.ones(W.size()).cuda()
            mask[:, idx.squeeze().long()] = self.tau
            self.W.data = mask * self.W.data + (1-mask) * W
            self.W.data = normalize(self.W.data)   
            # retrieve
            ll = torch.split(old_w, dim) # 
            nw,nb, nf = w.size(1), b.size(1), f.size(1)
            o_w, o_b, o_f = torch.cat(*[ll[:nw]]), torch.cat(*[ll[nw:nw+nb]]), torch.cat(*[ll[-nf:]])
            try:
                w = self.tau * w + (1-self.tau)*o_w.view(w.size())
                b = self.tau * b + (1-self.tau)*o_b.view(b.size())
                f = self.tau * f + (1-self.tau)*o_f.view(f.size())
            except:
                pdb.set_trace()
        f = f.view(-1).unsqueeze(0).unsqueeze(2)
       
        return w.unsqueeze(0) ,b.view(-1),f

    def forward(self, x):
        w,b,f = self.fw_chunks()
        d0, d1 = self.conv.weight.shape[1:]
        
        cw = self.conv.weight * w
        #cw = self.conv.weight
        try:
            if self.need_bias:
                conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation, bias = self.bias * b)
            else:
                conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation)
            out =  f * conv_out
        except: pdb.set_trace()
        return out

    def representation(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

    def _forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out



class SamePadConv_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, gamma=0.9, bias=True):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=False
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]),requires_grad=True)
        self.padding=padding
        self.dilation = dilation
        self.kernel_size= kernel_size
        self.need_bias = bias
        
        self.grad_dim, self.shape = [], []
        for p in self.conv.parameters():
            self.grad_dim.append(p.numel())
            self.shape.append(p.size())
        self.dim = sum(self.grad_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_features= out_channels

        self.n_chunks = out_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(out_channels*kernel_size// self.n_chunks)
        
        self.grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        self.f_grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        nh=64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, 1)
        self.calib_f = nn.Linear(nh, 1)
        dim = self.n_chunks * (self.chunk_out_d + 0 * in_channels // out_channels + 2)
        self.W = nn.Parameter(torch.empty(dim, 32), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = normalize(self.W.data)
        
        
        #self.calib_w = torch.nn.Parameter(torch.ones(out_channels, in_channels,1), requires_grad = True)
        #self.calib_b = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad = True)
        #self.calib_f = torch.nn.Parameter(torch.ones(1,out_channels,1), requires_grad = True)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.tau = 0.75
    def ctrl_params(self):
        c_iter = chain(self.controller.parameters(), self.calib_w.parameters(), 
                self.calib_b.parameters(), self.calib_f.parameters())
        for p in c_iter:
            yield p

    def store_grad(self):
        #print('storing grad')
        grad = self.conv.weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        self.f_grads = self.f_gamma * self.f_grads + (1-self.f_gamma) * grad
        if not self.training: 
            e = self.cos(self.f_grads, self.grads)
            
            if e < -self.tau:
                self.trigger = 1
        self.grads = self.gamma * self.grads + (1-self.gamma) * grad
        
    def fw_chunks(self):
        x = self.grads.view(self.n_chunks, -1)
        rep = self.controller(x)
        w = self.calib_w(rep)
        b = self.calib_b(rep)
        f = self.calib_f(rep)
        q = torch.cat([w.view(-1), b.view(-1), f.view(-1)])
        if not hasattr(self, 'q_ema'):
            setattr(self, 'q_ema', torch.zeros(*q.size()).float().cuda())
        else:
            self.q_ema = self.f_gamma * self.q_ema + (1-self.f_gamma)*q
            q = self.q_ema
        if self.trigger == 1:
            dim = w.size(0)
            self.trigger = 0
            # read
            
            att = q @ self.W
            att = F.softmax(att/0.5)
            
            v, idx = torch.topk(att, 2)
            ww = torch.index_select(self.W, 1, idx)
            idx = idx.unsqueeze(1).float()
            old_w = ww @ idx
            # write memory
            s_att = torch.zeros(att.size(0)).cuda()
            s_att[idx.squeeze().long()] = v.squeeze()
            W = old_w @ s_att.unsqueeze(0)
            mask = torch.ones(W.size()).cuda()
            mask[:, idx.squeeze().long()] = self.tau
            self.W.data = mask * self.W.data + (1-mask) * W
            self.W.data = normalize(self.W.data)   
            # retrieve
            ll = torch.split(old_w, dim)
            nw,nb, nf = w.size(1), b.size(1), f.size(1)
            o_w, o_b, o_f = torch.cat(*[ll[:nw]]), torch.cat(*[ll[nw:nw+nb]]), torch.cat(*[ll[-nf:]])
            
            try:
                w = self.tau * w + (1-self.tau)*o_w.view(w.size())
                b = self.tau * b + (1-self.tau)*o_b.view(b.size())
                f = self.tau * f + (1-self.tau)*o_f.view(f.size())
            except:
                pdb.set_trace()
        f = f.view(-1).unsqueeze(0).unsqueeze(2)
       
        return w.unsqueeze(1) ,b.view(-1),f

    def forward(self, x):
        w,b,f = self.fw_chunks()
        d0, d1 = self.conv.weight.shape[1:]
        
        cw = self.conv.weight * w
        #cw = self.conv.weight
        try:
            if self.need_bias:
                conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation, bias = self.bias * b)
            else:
                conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation)
            out =  f * conv_out
        except: pdb.set_trace()
        return out

    def representation(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

    def _forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, gamma=0.9):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def ctrl_params(self):  
        c_iter = chain(self.conv1.controller.parameters(), self.conv1.calib_w.parameters(), 
                self.conv1.calib_b.parameters(), self.conv1.calib_f.parameters(),
                self.conv2.controller.parameters(), self.conv2.calib_w.parameters(), 
                self.conv2.calib_b.parameters(), self.conv2.calib_f.parameters())

        return c_iter 
       


    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1), gamma=gamma
            )
            for i in range(len(channels))
        ])
    def ctrl_params(self):
        ctrl = []
        for l in self.net:
            ctrl.append(l.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p
    def forward(self, x):
        return self.net(x)
