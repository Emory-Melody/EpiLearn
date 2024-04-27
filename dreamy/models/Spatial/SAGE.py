import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch
from torch_geometric.utils import sort_edge_index

from .base import BaseModel

class SAGE(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, nlayers=2, dropout=0.5,
                with_bn=False, with_bias=True, device=None, aggr="mean"):

        super(SAGE, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            if callable(aggr):
                try:
                    self.layers.append(SAGEConv(input_dim, hidden_dim, bias=with_bias, aggr=aggr(input_dim, input_dim)))
                except:
                    self.layers.append(SAGEConv(input_dim, hidden_dim, bias=with_bias, aggr=aggr()))
            else:
                self.layers.append(SAGEConv(input_dim, hidden_dim, bias=with_bias, aggr=aggr))
        else:
            if callable(aggr):
                try:
                    self.layers.append(SAGEConv(input_dim, hidden_dim, bias=with_bias, aggr=aggr(input_dim, input_dim)))
                except:
                    self.layers.append(SAGEConv(input_dim, hidden_dim, bias=with_bias, aggr=aggr()))
            else:
                self.layers.append(SAGEConv(input_dim, hidden_dim, bias=with_bias, aggr=aggr))

            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            for i in range(nlayers-2):
                if callable(aggr):
                    try:
                        self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=aggr(hidden_dim, hidden_dim)))
                    except:
                        self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=aggr()))
                else:
                    self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=aggr))
                
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
            if callable(aggr):
                try:
                    self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=aggr(hidden_dim, hidden_dim)))
                except:
                    self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=aggr()))
            else:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=aggr))

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.with_bn = with_bn

    def forward(self, data):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = torch.reshape(x, (x.shape[0], -1))

        edge_index = sort_edge_index(edge_index, sort_by_row=False)
        for i, layer in enumerate(self.layers):
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


