# this code is adopted from https://github.com/amy-deng/colagnn/blob/master/src/models.py

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

from .base import BaseModel

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 


class ColaGNN(BaseModel):  
    """
    Convolutional-Layer Graph Neural Network (ColaGNN)

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_features : int
        Number of features per node per timestep.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    nhid : int, optional
        Number of hidden units in the RNN and GNN layers. Default: 32.
    n_channels : int, optional
        Number of channels for the convolution layers. Default: 1.
    n_spatial : int, optional
        Number of spatial features from graph convolutions. Default: max(10, nhid // 2).
    rnn_model : str, optional
        Type of RNN model to use ('LSTM', 'GRU', 'RNN'). Default: 'GRU'.
    n_layer : int, optional
        Number of layers in the RNN model. Default: 1.
    bidirect : bool, optional
        Whether the RNN layers are bidirectional. Default: False.
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
                n_channels=1,
                n_spatial=None,
                rnn_model = 'GRU',
                n_layer = 1,
                bidirect = False,
                dropout = 0.5,
                device='cpu',
                **kwargs):

        super().__init__(device)
        self.x_h = num_features 
        self.m = num_nodes
        self.w = num_timesteps_input
        self.h = num_timesteps_output

        self.dropout = dropout
        self.n_hidden = nhid
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = n_channels
        # CRITICAL FIX: Use smaller kernel sizes to preserve temporal information
        # Original: kernel=w (entire window) → output length=1 (93.8% info loss!)
        # Fixed: kernel=3 with padding → preserve temporal dimension
        short_kernel = min(3, self.w)  # Use kernel=3 or smaller if window is tiny
        self.conv = nn.Conv1d(self.x_h, self.k, short_kernel, padding=short_kernel//2)
        # Long conv: use kernel=5 with dilation for larger receptive field
        long_kernel = min(5, self.w)
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernel, dilation=2, padding=2*(long_kernel//2))
        # With padding, output length ≈ input length
        long_out = self.w  # Approximate, actual may vary by ±1
        # Make n_spatial configurable with better default based on hidden size
        self.n_spatial = n_spatial if n_spatial is not None else max(10, self.n_hidden // 2)

        # CRITICAL FIX: Update input size for graph conv
        # With new padding, both convs output ~w timesteps, concat gives ~2*w timesteps
        # Flattened: 2*w*k features (approximately)
        self.conv1 = GraphConvLayer(2*self.w*self.k, self.n_hidden)
        # Add batch normalization to stabilize large feature dimensions
        self.bn1 = nn.BatchNorm1d(self.n_hidden)
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)
        self.bn2 = nn.BatchNorm1d(self.n_spatial)

        rnn_dropout = dropout if n_layer > 1 else 0
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=n_layer, dropout=rnn_dropout, batch_first=True, bidirectional=bidirect)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=n_layer, dropout=rnn_dropout, batch_first=True, bidirectional=bidirect)
        elif rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=n_layer, dropout=rnn_dropout, batch_first=True, bidirectional=bidirect)
        else:
            raise LookupError (' only support LSTM, GRU and RNN')

        hidden_size = (int(bidirect) + 1) * self.n_hidden
        self.out = nn.Linear(hidden_size + self.n_spatial, self.h)  

        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.w)
            self.residual = nn.Linear(self.residual_window, 1) 
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, states=None, dynamic_adj=None):
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
        b, _,_,_ = x.size()
        orig_x = x 

        x = x.transpose(2, 1).contiguous().flatten(0,1)
        r_out, hc = self.rnn(x, None)
        last_hid = r_out[:,-1,:]
        last_hid = last_hid.view(-1,self.m, self.n_hidden)
        out_temporal = last_hid  # [b, m, 20]

        hid_rpt_m = last_hid.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        hid_rpt_w = last_hid.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)

        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            h_tmp = h_mids[:,:,i].transpose(2,1).contiguous() 
            r = self.conv(h_tmp) # [32, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l,dim=1)
        r_long_l = torch.stack(r_long_l,dim=1)
        r_l = torch.cat((r_l,r_long_l),-1)
        r_l = r_l.view(r_l.size(0),r_l.size(1),-1)
        r_l = torch.relu(r_l)
        adjs = adj.repeat(b,1)
        adjs = adjs.view(b,self.m,self.m)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        a_mx = adjs * c + a_mx * (1-c)
        # Row-normalize learned adjacency to bound spectral radius
        row_sum = a_mx.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        adj = a_mx / row_sum
        x = r_l
        x = F.relu(self.conv1(x, adj))
        # Transpose for batch norm: (batch, nodes, features) → (batch, features, nodes)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, self.dropout, training=self.training)

        out_spatial = F.relu(self.conv2(x, adj))
        # Transpose for batch norm: (batch, nodes, features) → (batch, features, nodes)
        out_spatial = self.bn2(out_spatial.transpose(1, 2)).transpose(1, 2)
        out = torch.cat((out_spatial, out_temporal),dim=-1)
        # out shape: (batch, nodes, n_spatial + hidden_size)
        out = self.out(out)
        # out shape: (batch, nodes, horizon) — already matches target shape

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            out = out * self.ratio + z; #[batch, m]

        return out
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
