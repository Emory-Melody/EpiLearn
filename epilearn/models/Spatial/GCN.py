# this code is adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense_pyg/gcn.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

from .base import BaseModel

class GCN(BaseModel):
    """
    Graph Convolutional Network (GCN)

    Parameters
    ----------
    num_features : int
        Number of input features per node.
    hidden_dim : int, optional
        Dimension of hidden layers. Default: 16.
    num_classes : int, optional
        Number of output classes for each node. Default: 2.
    nlayers : int, optional
        Number of layers in the GCN. Default: 2.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    with_bn : bool, optional
        Specifies whether batch normalization should be included. Default: False.
    with_bias : bool, optional
        Specifies whether to include bias parameters in the GCN layers. Default: True.
    device : str
        The device (cpu or gpu) on which the model will be run.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_nodes, num_classes), representing the predicted outcomes for each node after passing through the GCN.
    """
    def __init__(self, num_features, hidden_dim=16, num_classes=2, nlayers=2, dropout=0.5,
                with_bn=False, with_bias=True, device='cpu'):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.hid = hidden_dim
        self.out_dim = num_classes

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GCNConv(num_features, hidden_dim, bias=with_bias))
        else:
            self.layers.append(GCNConv(num_features, hidden_dim, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            for i in range(nlayers-2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim, bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(GCNConv(hidden_dim, hidden_dim, bias=with_bias))
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight):
        """
        Parameters:
        x : torch.Tensor
            The input features tensor with shape (batch_size, num_nodes, num_features).
        edge_index : torch.Tensor
            The edge indices in COO format with shape (2, num_edges).
            
        Returns:
        torch.Tensor
            The output predictions for each node with shape (batch_size * num_nodes, num_classes).
        """
        #print(x.shape)
        #b, n, _= x.shape
        x = torch.reshape(x, (x.shape[0], -1))
        #print(x.shape)
    
        for i, layer in enumerate(self.layers):
            if edge_weight is not None:
                x = layer(x.float(), edge_index=edge_index, edge_weight=edge_weight)
            else:
                x = layer(x.float(), edge_index)
            if i != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[i](x)#(x.view(-1, self.hid)).view(b, n, self.hid)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x) #self.fc(x)#.squeeze() #F.log_softmax(x, dim=1)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
