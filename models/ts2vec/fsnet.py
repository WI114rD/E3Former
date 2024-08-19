import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft

import numpy as np
from einops import rearrange, reduce, repeat

from .fsnet_ import DilatedConvEncoder
from layers.Embed import PatchEmbedding


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma=gamma
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10
    
    def ctrl_params(self):
        return self.feature_extractor.ctrl_params()

    def forward_time(self, x, mask=None):  # x: B x T x input_dims
        x = x.transpose(1, 2)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
    def forward(self, x, mask=None):  # x: B x T x input_dims
            nan_mask = ~x.isnan().any(axis=-1)
            x[~nan_mask] = 0
            x = self.input_fc(x)  # B x T x Ch
            
            # generate & apply mask
            if mask is None:
                if self.training:
                    mask = self.mask_mode
                else:
                    mask = 'all_true'
            
            if mask == 'binomial':
                mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'continuous':
                mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'all_true':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            elif mask == 'all_false':
                mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
            elif mask == 'mask_last':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
                mask[:, -1] = False
            
            mask &= nan_mask
            x[~mask] = 0
            
            # conv encoder
            x = x.transpose(1, 2)  # B x Ch x T
            x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
            x = x.transpose(1, 2)  # B x T x Co
            
            return x
    

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma=gamma
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10
    
    def ctrl_params(self):
        return self.feature_extractor.ctrl_params()

    def forward_time(self, x, mask=None):  # x: B x T x input_dims
        x = x.transpose(1, 2) 
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
    def forward(self, x, mask=None):  # x: B x T x input_dims
            nan_mask = ~x.isnan().any(axis=-1)
            x[~nan_mask] = 0
            x = self.input_fc(x)  # B x T x Ch
            
            # generate & apply mask
            if mask is None:
                if self.training:
                    mask = self.mask_mode
                else:
                    mask = 'all_true'
            
            if mask == 'binomial':
                mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'continuous':
                mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'all_true':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            elif mask == 'all_false':
                mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
            elif mask == 'mask_last':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
                mask[:, -1] = False
            
            mask &= nan_mask
            x[~mask] = 0
            
            # conv encoder
            x = x.transpose(1, 2)  # B x Ch x T
            x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
            x = x.transpose(1, 2)  # B x T x Co
            
            return x
    
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
    

class MultiModelIntegrationGate(nn.Module):
    def __init__(self, head_dropout=0):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, models_output, gate):   # models_output: 长度为num_models的模型输出列表. 每个元素形状为[bs x nvars x target_window]
                                             # gate: 形状为[bs, num_models]的权重
        
        combined_outputs = []
        for i, model_out in enumerate(models_output):
            combined_outputs.append(model_out.unsqueeze(-1))  # [bs x nvars x target_window, 1]
        
        combined_outputs = torch.cat(combined_outputs, dim=-1)  # [bs x nvars x target_window x num_models]
        
        gate = gate.unsqueeze(-2)  
        gate = gate.repeat(1, combined_outputs.shape[-2], 1)  # [bs x target_window x num_models]
        gate = gate.unsqueeze(-3)  
        gate = gate.repeat(1, combined_outputs.shape[-3], 1, 1)  # [bs x nvars x target_window x num_models]

        res = (combined_outputs * gate).sum(dim=-1)  # [bs x nvars x target_window]
        res = self.dropout(res)
        
        return res
class MIMO_TSEncoder_MOE(nn.Module):
    def __init__(self, input_dims, output_dims, seq_len, pred_len, patch_lens, strides, 
                 hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9, channel_cross=False, individual=False, dropout=0.1, head_dropout=0.1):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.seq_len = seq_len
        self.patch_lens =  patch_lens
        self.strides = strides
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.pred_len = pred_len
        # self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma=gamma
        )

        self.gate_flatten = nn.Flatten(start_dim=-2)
        self.w_gate = nn.Linear(self.input_dims * seq_len, len(self.patch_lens), bias=False)
        self.gate_softmax = nn.Softmax(-1)

        self.repr_dropout = nn.Dropout(p=0.1)

        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(d_model=hidden_dims, patch_len=patch_len, stride=stride, padding=stride, dropout=dropout)
            for patch_len, stride in zip(self.patch_lens, self.strides)
        ])

        patch_nums = [int((seq_len - patch_len)/stride + 2) for patch_len, stride in zip(patch_lens, strides)]
        self.padding_patch_layer = [nn.ReplicationPad1d((0, stride)) for stride in self.strides]

        # Head
        self.channel_cross = channel_cross
        if channel_cross:
            self.head_nf = output_dims * input_dims # use the represetations of the last patch for forecasting
            target_window = input_dims * pred_len
        else:
            self.head_nfs = [
            output_dims * patch_num
            for patch_num in patch_nums
            ]
            target_window = self.pred_len

        self.n_vars = input_dims
        self.individual = individual

        # if self.pretrain_head: 
        #     self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        # elif head_type == 'flatten': 
        #     self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        self.heads = nn.ModuleList([
            Flatten_Head(self.individual, self.n_vars, head_nf, target_window, head_dropout=head_dropout)
            for head_nf in self.head_nfs
        ])

        self.out_fc = nn.Linear(self.input_dims, self.output_dims)

        self.mimo_heads = MultiModelIntegrationGate(head_dropout=head_dropout)

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10
    
    def ctrl_params(self):
        return self.feature_extractor.ctrl_params()

    def forward(self, x, mask=None): # x: B x T x input_dims
        bsz, seq_len, input_dims = x.shape
        z = x
        nan_mask = ~z.isnan().any(axis=-1)
        z[~nan_mask] = 0

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(z.size(0), z.size(1)).to(z.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(z.size(0), z.size(1)).to(z.device)
        elif mask == 'all_true':
            mask = z.new_full((z.size(0), z.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = z.new_full((z.size(0), z.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = z.new_full((z.size(0), z.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
            
        mask &= nan_mask
        z[~mask] = 0

        z_input = z.permute(0, 2, 1)
        
        z_w = self.gate_flatten(z_input)
        gate = self.w_gate(z_w)
        gate = self.gate_softmax(gate)

        outputs = []
        for patch_embedding, head in zip(self.patch_embeddings, self.heads):
            z, _ = patch_embedding(z_input) # [bs * nvars x patch_num x hidden_dims]
            # conv encoder
            z = z.transpose(1, 2)  # [bs * nvars x hidden_dims x patch_num]

            z = self.repr_dropout(self.feature_extractor(z))  # [bs * nvars x output_dims x patch_num]
            
            # x = x.transpose(1, 2)  # 
            z = torch.reshape(
                z, (-1, input_dims, z.shape[-2], z.shape[-1])) # [bs x nvars x output_dims x patch_num]
            z = z.permute(0, 1, 3, 2)

            z = head(z) # [bs x nvars x target_window]
            z = z.permute(0, 2, 1) # [bs x target_window x nvars]

            z = self.out_fc(z) # [bs x target_window x output_dims]

            z = z.permute(0, 2, 1) # [bs x output_dims x target_window]

            outputs.append(z)
        output = self.mimo_heads(outputs, gate).permute(0, 2, 1) # [bs x target_window x output_dims]

        return output
    def forward_time(self, x, mask=None):  # x: B x T x input_dims
        x = x.transpose(1, 2)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
    

class MIMO_TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, seq_len, pred_len, patch_lens, strides, 
                 hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9, channel_cross=False, individual=False, dropout=0.1, head_dropout=0.1):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.seq_len = seq_len
        self.patch_lens =  patch_lens
        self.strides = strides
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.pred_len = pred_len
        # self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma=gamma
        )

        self.w_gate = nn.Linear(seq_len, len(self.patch_lens), bias=False)
        self.gate_softmax = nn.Softmax(-1)

        self.repr_dropout = nn.Dropout(p=0.1)

        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(d_model=hidden_dims, patch_len=patch_len, stride=stride, padding=stride, dropout=dropout)
            for patch_len, stride in zip(self.patch_lens, self.strides)
        ])

        patch_nums = [int((seq_len - patch_len)/stride + 2) for patch_len, stride in zip(patch_lens, strides)]
        self.padding_patch_layer = [nn.ReplicationPad1d((0, stride)) for stride in self.strides]

        # Head
        self.channel_cross = channel_cross
        if channel_cross:
            self.head_nf = output_dims * input_dims # use the represetations of the last patch for forecasting
            target_window = input_dims * pred_len
        else:
            self.head_nfs = [
            output_dims * patch_num
            for patch_num in patch_nums
            ]
            target_window = self.pred_len

        self.n_vars = input_dims
        self.individual = individual

        # if self.pretrain_head: 
        #     self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        # elif head_type == 'flatten': 
        #     self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        self.heads = nn.ModuleList([
            Flatten_Head(self.individual, self.n_vars, head_nf, target_window, head_dropout=head_dropout)
            for head_nf in self.head_nfs
        ])

        self.out_fc = nn.Linear(self.input_dims, self.output_dims)

        # self.mimo_heads = MultiModelIntegrationGate(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout, num_models=len(self.patch_lens))

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10
    
    def ctrl_params(self):
        return self.feature_extractor.ctrl_params()

    def forward(self, x, mask=None): # x: B x T x input_dims
        bsz, seq_len, input_dims = x.shape
        z = x
        nan_mask = ~z.isnan().any(axis=-1)
        z[~nan_mask] = 0

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(z.size(0), z.size(1)).to(z.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(z.size(0), z.size(1)).to(z.device)
        elif mask == 'all_true':
            mask = z.new_full((z.size(0), z.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = z.new_full((z.size(0), z.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = z.new_full((z.size(0), z.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
            
        mask &= nan_mask
        z[~mask] = 0

        z_input = z.permute(0, 2, 1)

        # gates = self.w_gate(z)
        # gates = self.softmax(gates)

        outputs = []
        output = None
        for patch_embedding, head in zip(self.patch_embeddings, self.heads):
            z, _ = patch_embedding(z_input) # [bs * nvars x patch_num x hidden_dims]
            # conv encoder
            z = z.transpose(1, 2)  # [bs * nvars x hidden_dims x patch_num]
            z = self.repr_dropout(self.feature_extractor(z))  # [bs * nvars x output_dims x patch_num] [46080, 320, 2]
            
            # x = x.transpose(1, 2)  # 
            z = torch.reshape(
                z, (-1, input_dims, z.shape[-2], z.shape[-1])) # [bs x nvars x output_dims x patch_num]
            z = z.permute(0, 1, 3, 2)

            z = head(z) # [bs x nvars x target_window]
            z = z.permute(0, 2, 1) # [bs x target_window x nvars]

            z = self.out_fc(z) # [bs x target_window x output_dims]
            outputs.append(z)
            output = z if output == None else output + z
        # output = self.mimo_heads(outputs, gate)
        output = output / len(self.patch_lens)
        return output
    def forward_time(self, x, mask=None):  # x: B x T x input_dims
        x = x.transpose(1, 2)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
    

# Additional components like DilatedConvEncoder should be defined as well.

class BandedFourierLayer(nn.Module):

    def __init__(self, in_channels, out_channels, band, num_bands, freq_mixing=False, bias=True, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.freq_mixing = freq_mixing

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        if self.freq_mixing:
            self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, self.total_freqs, out_channels), dtype=torch.cfloat))
        else:
            self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        else:
            self.bias = None
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        if self.freq_mixing:
            output = torch.einsum('bai,tiao->bto', input, self.weight)
        else:
            output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        if self.bias is None:
            return output
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class GlobalLocalMultiscaleTSEncoder(nn.Module):

    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 num_bands: int,
                 freq_mixing: bool,
                 length: int,
                 mode = 0,
                 hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9):
        super().__init__()

        self.mode = mode

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma = gamma
        )

        self.kernels = kernels
        self.num_bands = num_bands

        self.convs = nn.ModuleList(
            [nn.Conv1d(output_dims, output_dims//2, k, padding=k-1) for k in kernels]
        )
        self.fouriers = nn.ModuleList(
            [BandedFourierLayer(output_dims, output_dims//2, b, num_bands,
                                freq_mixing=freq_mixing, length=length) for b in range(num_bands)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        if len(self.kernels) == 0:
            local_multiscale = None
        else:
            local_multiscale = []
            for idx, mod in enumerate(self.convs):
                out = mod(x)  # b d t
                if self.kernels[idx] != 1:
                    out = out[..., :-(self.kernels[idx] - 1)]
                local_multiscale.append(out.transpose(1, 2))  # b t d
            local_multiscale = reduce(
                rearrange(local_multiscale, 'list b t d -> list b t d'),
                'list b t d -> b t d', 'mean'
            )

        x = x.transpose(1, 2)  # B x T x Co

        if self.num_bands == 0:
            global_multiscale = None
        else:
            global_multiscale = []
            for mod in self.fouriers:
                out = mod(x)  # b t d
                global_multiscale.append(out)

            global_multiscale = global_multiscale[0]

        return torch.cat([local_multiscale, global_multiscale], dim=-1)
