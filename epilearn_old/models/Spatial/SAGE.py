import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch
from torch_geometric.utils import sort_edge_index

from .base import BaseModel

class SAGE(BaseModel):
    """
    Graph Sample and Aggregate (SAGE)

    Parameters
    ----------
    num_features : int
        Number of input features per node.
    hidden_dim : int
        Dimension of hidden layers.
    num_classes : int
        Number of output features per node.
    nlayers : int, optional
        Number of layers in the GraphSAGE model. Default: 2.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    with_bn : bool, optional
        Specifies whether batch normalization should be included. Default: False.
    with_bias : bool, optional
        Specifies whether to include bias parameters in the GraphSAGE layers. Default: True.
    device : str
        The device (cpu or gpu) on which the model will be run. Must be specified.
    aggr : str or callable, optional
        The aggregation function to use ('mean', 'sum', 'max', etc.), or a callable that returns a custom aggregation function. Default: 'mean'.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_nodes, output_dim), representing the predicted outcomes for each node after passing through the GraphSAGE model.
    """
    def __init__(self, num_features, hidden_dim, num_classes, nlayers=2, dropout=0.5,
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
                    self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=aggr(num_features, num_features)))
                except:
                    self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=aggr()))
            else:
                self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=aggr))
        else:
            if callable(aggr):
                try:
                    self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=aggr(num_features, num_features)))
                except:
                    self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=aggr()))
            else:
                self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=aggr))

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

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight):
        """
        Parameters:
        x : torch.Tensor
            Node feature matrix with shape (batch_size, num_nodes, num_features).
        edge_index : torch.Tensor
            Edge index in COO format with shape (2, num_edges).

        Returns:
        torch.Tensor
            Output from the network with shape (batch_size * num_nodes, num_classes).
        """
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
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
