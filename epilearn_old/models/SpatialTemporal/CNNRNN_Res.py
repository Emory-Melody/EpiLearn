# this code is adopted from https://github.com/CrickWu/DL4Epi/blob/master/models/CNNRNN_Res.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .base import BaseModel

class CNNRNN_Res(BaseModel):
    """
    Combined Convolutional Neural Network and Recurrent Neural Network with Residual Connections (CNNRNN_Res)

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_features : int
        Number of features per node per timestep.
    num_timesteps_input : int
        Number of timesteps considered for each input sample (window size).
    num_timesteps_output : int
        Number of output timesteps to predict.
    nhid : int, optional
        Number of hidden units in the GRU layer. Default: 32.
    residual_ratio : float, optional
        Proportion of the residual connection compared to the GRU output. Default: 0.
    residual_window : int, optional
        Number of timesteps to include in the residual connection. Default: 0.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    device : str, optional
        The device (cpu or gpu) on which the model will be run. Default: 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output, num_nodes), representing the predicted values for each node over future timesteps.
        Each slice along the second dimension corresponds to a timestep, with each column representing a node.
    """
    def __init__(self, 
                num_nodes, 
                num_features,
                num_timesteps_input,
                num_timesteps_output,
                nhid=32,
                residual_ratio=0,
                residual_window=0,
                dropout=0.5,
                device='cpu'):
        super(CNNRNN_Res, self).__init__()
        self.nfeat = num_features
        self.ratio = residual_ratio
        self.device = device
        self.P = num_timesteps_input
        self.n_out = num_timesteps_output
        self.m = num_nodes
        self.hidR = nhid
        self.GRU = nn.GRU(self.m, self.hidR, batch_first=True)
        self.residual_window = residual_window
        self.mask_mat = Parameter(torch.Tensor(self.m, self.m))

        self.squeeze_features = nn.Linear(self.nfeat, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.hidR, self.m)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1)
        
        self.out = nn.Linear(1, self.n_out)

    def forward(self, x, adj, states=None, dynamic_adj=None, **kargs):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (batch_size, num_timesteps_input, num_nodes, num_features).
        adj : torch.Tensor
            Static adjacency matrix of the graph with shape (num_nodes, num_nodes).
        states : torch.Tensor, optional
            States of the nodes if available, with the same shape as x. Default: None.
        dynamic_adj : torch.Tensor, optional
            Dynamic adjacency matrix if available, with shape similar to adj but possibly varying over time. Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, num_timesteps_output, num_nodes),
            representing the predicted values for each node over the specified output timesteps.
        """
        # first transform
        masked_adj = adj * self.mask_mat
        x = torch.matmul(masked_adj, x)
        # RNN
        # r: window (self.P) x batch x #signal (m)
        r = self.squeeze_features(x).contiguous().view(-1, self.P, self.m)
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

        res = F.sigmoid(self.out(res.unsqueeze(-1))).float()

        return res.transpose(2,1)
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
