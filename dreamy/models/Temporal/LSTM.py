import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from .base_model import BaseModel

class LSTM(BaseModel):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, bias=True, device=None):

        super(LSTM, self).__init__()

        self.model = nn.LSTM()

        self.device = device

    def forward(self, x):
       pass

    def get_embed(self, x, ):
        pass

    def initialize(self):
        pass
        # for m in self.layers:
        #     m.reset_parameters()
        # if self.with_bn:
        #     for bn in self.bns:
        #         bn.reset_parameters()