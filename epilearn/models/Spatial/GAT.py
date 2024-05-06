import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import torch

from .base import BaseModel

class GAT(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, nlayers=2, nheads=[1, 1], dropout=0.5,
                with_bn=False, with_bias=True, device=None, concat=False):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        
        assert len(nheads) == nlayers, "nheads should match nlayers!"

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()
            
        
        if concat:
            for i in range(nlayers):
                assert hidden_dim % nheads[i] == 0, "Hidden_dim should be divisible by nheads!"
                #assert output_dim % nheads[i] == 0, "Output_dim should be divisible by nheads!"
        
        if concat:
            if nlayers == 1:
                self.layers.append(GATConv(input_dim, hidden_dim//nheads[0], bias=with_bias, heads=nheads[0], concat=concat))
            else:
                self.layers.append(GATConv(input_dim, hidden_dim//nheads[0], bias=with_bias, heads=nheads[0], concat=concat))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
                        
                for i in range(nlayers-2):
                    self.layers.append(GATConv(hidden_dim, hidden_dim//nheads[i+1], bias=with_bias, heads=nheads[i+1], concat=concat))
                    if with_bn:
                        self.bns.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(GATConv(hidden_dim, hidden_dim//nheads[-1], bias=with_bias, heads=nheads[-1], concat=concat))
            
            
        else:
            if nlayers == 1:
                self.layers.append(GATConv(input_dim, hidden_dim, bias=with_bias, heads=nheads[0], concat=concat))
            else:
                self.layers.append(GATConv(input_dim, hidden_dim, bias=with_bias, heads=nheads[0], concat=concat))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
                        
                for i in range(nlayers-2):
                    self.layers.append(GATConv(hidden_dim, hidden_dim, bias=with_bias, heads=nheads[i+1], concat=concat))
                    if with_bn:
                        self.bns.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(GATConv(hidden_dim, hidden_dim, bias=with_bias, heads=nheads[-1], concat=concat))
                
            
        self.fc = nn.Linear(hidden_dim, output_dim)
            
        self.dropout = dropout
        self.with_bn = with_bn

    def forward(self, data):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.reshape(x, (x.shape[0], -1))

        for i, layer in enumerate(self.layers):
            if edge_attr is not None:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index=edge_index)
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