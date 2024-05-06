import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn.models import MLP
import torch

from .base import BaseModel

class GIN(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, nlayers=2, dropout=0.5,
                with_bias=True, device=None):

        super(GIN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        
        mlp_one = MLP(
            [input_dim, hidden_dim, output_dim],
            act="relu",
            norm="batch_norm",
            bias=with_bias
        )
        
        mlp_first = MLP(
            [input_dim, hidden_dim, hidden_dim],
            act="relu",
            norm="batch_norm",
            bias=with_bias
        )
        
        mlp_hidden = MLP(
            [hidden_dim, hidden_dim, hidden_dim],
            act="relu",
            norm="batch_norm",
            bias=with_bias
        )
        
        mlp_last = MLP(
            [hidden_dim, hidden_dim, output_dim],
            act="relu",
            norm="batch_norm",
            bias=with_bias
        )
        
        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GINConv(mlp_first))
        else:
            self.layers.append(GINConv(mlp_first))
            for i in range(nlayers-2):
                self.layers.append(GINConv(mlp_hidden))
            self.layers.append(GINConv(mlp_hidden))
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout


    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        x = torch.reshape(x, (x.shape[0], -1))
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x) #F.log_softmax(x, dim=1)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()