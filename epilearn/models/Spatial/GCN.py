
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

from .base import BaseModel

class GCN(BaseModel):

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
        b, n, _= x.shape
    
        for i, layer in enumerate(self.layers):
            if edge_weight is not None:
                x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            else:
                x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[i](x.view(-1, self.hid)).view(b, n, self.hid)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x).squeeze() #F.log_softmax(x, dim=1)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
