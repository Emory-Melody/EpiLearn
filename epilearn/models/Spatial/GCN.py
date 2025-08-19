# this code is adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense_pyg/gcn.py

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from .base import BaseModel


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNConv(GraphConvolution):
    
    def __init__(self, in_channels, out_channels, bias=True):
        super(GCNConv, self).__init__(in_channels, out_channels, with_bias=bias)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Input node features
            edge_index: Edge indices (will be converted to adjacency matrix)
            edge_weight: Edge weights
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Input shape: [batch_size, num_nodes, in_channels]
            batch_size, num_nodes, in_channels = x.shape
            
            # Create adjacency matrix for batch processing
            adj_matrices = []
            for b in range(batch_size):
                adj = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=x.dtype)
                if edge_weight is not None:
                    adj[edge_index[0], edge_index[1]] = edge_weight
                else:
                    adj[edge_index[0], edge_index[1]] = 1.0
                
                # Add self-loops and normalize
                adj = adj + torch.eye(num_nodes, device=x.device, dtype=x.dtype)
                row_sum = adj.sum(dim=1, keepdim=True)
                row_sum[row_sum == 0] = 1  # Avoid division by zero
                adj = adj / row_sum
                adj_matrices.append(adj)
            
            # Apply GCN to each sample in the batch
            outputs = []
            for b in range(batch_size):
                x_sample = x[b]  # [num_nodes, in_channels]
                adj_sample = adj_matrices[b]  # [num_nodes, num_nodes]
                output = super(GCNConv, self).forward(x_sample, adj_sample)
                outputs.append(output)
            
            return torch.stack(outputs, dim=0)  # [batch_size, num_nodes, out_channels]
        
        else:
            # Input shape: [num_nodes, in_channels]
            num_nodes = x.size(0)
            
            # Create adjacency matrix
            adj = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=x.dtype)
            if edge_weight is not None:
                adj[edge_index[0], edge_index[1]] = edge_weight
            else:
                adj[edge_index[0], edge_index[1]] = 1.0
            
            # Add self-loops and normalize
            adj = adj + torch.eye(num_nodes, device=x.device, dtype=x.dtype)
            row_sum = adj.sum(dim=1, keepdim=True)
            row_sum[row_sum == 0] = 1  # Avoid division by zero
            adj = adj / row_sum
            
            return super(GCNConv, self).forward(x, adj)



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
