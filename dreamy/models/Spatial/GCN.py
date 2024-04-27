import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

from .base import BaseModel

class GCN(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, nlayers=2, dropout=0.5,
                with_bn=False, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GCNConv(input_dim, hidden_dim, bias=with_bias))
        else:
            self.layers.append(GCNConv(input_dim, hidden_dim, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            for i in range(nlayers-2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim, bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(GCNConv(hidden_dim, hidden_dim, bias=with_bias))
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.with_bn = with_bn

    def forward(self, data):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = torch.reshape(x, (x.shape[0], -1))
        for i, layer in enumerate(self.layers):
            if edge_weight is not None:
                x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            else:
                x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x) #F.log_softmax(x, dim=1)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()


