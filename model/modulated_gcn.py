from __future__ import absolute_import
import torch
import torch.nn as nn
from model.modulated_gcn_conv import ModulatedGraphConv
from model.graph_non_local import GraphNonLocal
from model.net.non_local_embedded_gaussian import NONLocalBlock2D

import math
from einops import rearrange
from functools import reduce

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered



class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv =  ModulatedGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out

class TimeStepBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )
        self.act = nn.SiLU()
        
    def forward(self, x, time_emb = None):
        'x : b 17 c'
        scale_shift = None
        time_emb  = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b 1 c')
        scale_shift = time_emb.chunk(2, dim = -1)
        scale, shift = scale_shift
        
        x = x * (scale + 1) + shift
        
        x = self.act(x)
        
        return x
        
class ModulatedGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(5, 5), num_layers=4, nodes_group=None, p_dropout=None):
        super(ModulatedGCN, self).__init__()
        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = ModulatedGraphConv(hid_dim, coords_dim[1], adj) 
        self.non_local = NONLocalBlock2D(in_channels=hid_dim, sub_sample=False)
        
        # time embeddings

        time_dim = hid_dim // 2

        sinu_pos_emb = SinusoidalPosEmb(hid_dim // 8)
        fourier_dim = hid_dim // 8

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.block1 = TimeStepBlock(dim = hid_dim, time_dim=time_dim)
        self.block2 = TimeStepBlock(dim = hid_dim, time_dim=time_dim)
        self.block3 = TimeStepBlock(dim = hid_dim, time_dim=time_dim)
        
        
        
    def forward(self, x, cond, time):
        x = torch.cat((x, cond), -1)
        x = x.squeeze() 
        # x = x.permute(0,2,1)
        
        t = self.time_mlp(time)
        
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        
        out = self.block1(out, t)
        
        out = out.unsqueeze(2)
        out = out.permute(0,3,2,1)
        out = self.non_local(out)
        
        out = out.permute(0,3,1,2).squeeze()
        out = self.block2(out, t)
        out = out.permute(0,2,1).unsqueeze(2)
        
        out = out.permute(0,3,1,2)
        out = out.squeeze()
        out = self.gconv_output(out)
        
        # out = out.permute(0,2,1)
        out = out.unsqueeze(1)
        # out = out.unsqueeze(4)
        return out
