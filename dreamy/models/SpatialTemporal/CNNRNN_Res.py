# this code is adopted from https://github.com/CrickWu/DL4Epi/blob/master/models/CNNRNN_Res.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .base import BaseModel

class CNNRNN_Res(BaseModel):
    def __init__(self, 
                num_nodes, 
                num_timesteps_input,
                num_timesteps_output,
                nhid,
                residual_ratio = 0,
                residual_window = 0,
                dropout = 0.5,
                device = 'cpu'):
        super(CNNRNN_Res, self).__init__()
        self.ratio = residual_ratio
        self.device = device
        self.P = num_timesteps_input
        self.n_out = num_timesteps_output
        self.m = num_nodes
        self.hidR = nhid
        self.GRU = nn.GRU(self.m, self.hidR)
        self.residual_window = residual_window
        self.mask_mat = Parameter(torch.Tensor(self.m, self.m))

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.hidR, self.m)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1)

    def forward(self, adj, x, **kargs):
        # x: batch x window (self.P) x #signal (m)
        # first transform

        masked_adj = adj * self.mask_mat
        x = x.matmul(masked_adj)
        # RNN
        # r: window (self.P) x batch x #signal (m)
        r = x.permute(1, 0, 2).contiguous()
        _, r = self.GRU(r)
        r = self.dropout(torch.squeeze(r, 0))

        res = self.linear(r)

        #residual
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(-1,self.m)
            res = res * self.ratio + z

        res = F.sigmoid(res).float()

        return res
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()