import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from .base import BaseModel

class GAT(BaseModel):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GATConv(nfeat, nclass, bias=with_bias))
        else:
            self.layers.append(GATConv(nfeat, nhid, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers-2):
                self.layers.append(GATConv(nhid, nhid, bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GATConv(nhid, nclass, bias=with_bias))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        self.name = 'GCN'

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if ii == len(self.layers) - 1:
                return x
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
        return x

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()