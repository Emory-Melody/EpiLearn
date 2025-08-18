import torch.nn as nn
import torch.nn.functional as F
import torch

from .base import BaseModel


class SAGEConv(nn.Module):
    """GraphSAGE Convolution layer"""
    
    def __init__(self, in_channels, out_channels, bias=True, aggr='mean'):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight=None):
        """GraphSAGE forward pass"""
        if x.dim() == 3:
            # Batch processing
            batch_size, num_nodes, _ = x.shape
            outputs = []
            
            for b in range(batch_size):
                x_sample = x[b]  # [num_nodes, in_channels]
                out = self._forward_single(x_sample, edge_index, edge_weight)
                outputs.append(out)
            
            return torch.stack(outputs, dim=0)
        else:
            return self._forward_single(x, edge_index, edge_weight)
    
    def _forward_single(self, x, edge_index, edge_weight=None):
        """Forward pass for single sample"""
        num_nodes = x.size(0)
        
        # Self transformation
        out = self.lin_l(x)
        
        # Neighbor aggregation
        if edge_index.size(1) > 0:
            row, col = edge_index
            
            # Create adjacency for aggregation
            adj = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=x.dtype)
            if edge_weight is not None:
                adj[row, col] = edge_weight
            else:
                adj[row, col] = 1.0
            
            # Normalize by degree
            degree = adj.sum(dim=1, keepdim=True)
            degree[degree == 0] = 1
            adj = adj / degree
            
            # Aggregate neighbors
            neighbor_features = torch.mm(adj, x)
            out = out + self.lin_r(neighbor_features)
        
        return out


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

        # store configuration for possible lazy-rebuild when input feature dim differs
        self._init_num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.nlayers = nlayers
        self.with_bias = with_bias
        self.aggr = aggr if isinstance(aggr, str) else 'mean'

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=self.aggr))
        else:
            self.layers.append(SAGEConv(num_features, hidden_dim, bias=with_bias, aggr=self.aggr))

            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            for i in range(nlayers-2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=self.aggr))
                
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, bias=with_bias, aggr=self.aggr))

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        self.with_bn = with_bn
        # ensure parameters are on the declared device
        self.to(self.device)

    def _rebuild_layers(self, in_channels):
        """Rebuild layers to accept a different input feature size."""
        self.layers = nn.ModuleList()
        if self.with_bn:
            self.bns = nn.ModuleList()

        if self.nlayers == 1:
            self.layers.append(SAGEConv(in_channels, self.hidden_dim, bias=self.with_bias, aggr=self.aggr))
        else:
            self.layers.append(SAGEConv(in_channels, self.hidden_dim, bias=self.with_bias, aggr=self.aggr))
            if self.with_bn:
                self.bns.append(nn.BatchNorm1d(self.hidden_dim))
            for i in range(self.nlayers-2):
                self.layers.append(SAGEConv(self.hidden_dim, self.hidden_dim, bias=self.with_bias, aggr=self.aggr))
                if self.with_bn:
                    self.bns.append(nn.BatchNorm1d(self.hidden_dim))
            self.layers.append(SAGEConv(self.hidden_dim, self.hidden_dim, bias=self.with_bias, aggr=self.aggr))

        # move newly created modules to the model device
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        # If input feature dim doesn't match initialized layers, rebuild to match
        actual_in = x.shape[-1]
        if len(self.layers) > 0 and self.layers[0].in_channels != actual_in:
            self._rebuild_layers(actual_in)

        # Do not flatten nodes/features globally; SAGEConv handles both 3D (batch) and 2D inputs.
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)

            # Apply batchnorm, activation and dropout for all but last layer
            if i != len(self.layers) - 1:
                if self.with_bn:
                    # BatchNorm1d expects (N, C). If batched (B, N, C) reshape to (B*N, C)
                    if x.dim() == 3:
                        b, n, f = x.shape
                        x = x.view(b * n, f)
                        x = self.bns[i](x)
                        x = x.view(b, n, f)
                    else:
                        x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply final linear layer to the feature dimension per node
        if x.dim() == 3:
            b, n, f = x.shape
            x = x.view(b * n, f)
            x = self.fc(x)               # (B*N, num_classes)
            return x
        else:
            # single sample: (num_nodes, hidden_dim) -> (num_nodes, num_classes)
            return self.fc(x)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
