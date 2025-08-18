import torch.nn as nn
import torch.nn.functional as F
import torch

from .base import BaseModel


class GINConv(nn.Module):
    """Graph Isomorphism Network Convolution layer"""
    
    def __init__(self, nn_module, eps=0.0, train_eps=False):
        super(GINConv, self).__init__()
        self.nn = nn_module
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self.nn, 'reset_parameters'):
            self.nn.reset_parameters()
        elif hasattr(self.nn, 'children'):
            for child in self.nn.children():
                if hasattr(child, 'reset_parameters'):
                    child.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight=None):
        """GIN forward pass"""
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
        
        # Aggregate neighbors
        if edge_index.size(1) > 0:
            row, col = edge_index
            
            # Sum aggregation
            out = torch.zeros_like(x)
            for i in range(edge_index.size(1)):
                src, dst = row[i], col[i]
                if edge_weight is not None:
                    out[dst] += edge_weight[i] * x[src]
                else:
                    out[dst] += x[src]
        else:
            out = torch.zeros_like(x)
        
        # Add self-loops with learnable weight
        out = (1 + self.eps) * x + out
        
        # Apply MLP
        return self.nn(out)


# Simple MLP for GIN
class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, channels, act='relu', norm=None, bias=True, dropout=0.0):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if norm else None
        
        for i in range(len(channels) - 1):
            self.layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            
            if norm == 'batch_norm' and i < len(channels) - 2:
                self.norms.append(nn.BatchNorm1d(channels[i + 1]))
            elif self.norms is not None:
                self.norms.append(nn.Identity())
        
        self.act = getattr(F, act) if isinstance(act, str) else act
        self.dropout = dropout
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.norms:
            for norm in self.norms:
                if hasattr(norm, 'reset_parameters'):
                    norm.reset_parameters()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.norms and i < len(self.norms):
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x)
        return x


class GIN(BaseModel):
    """
    Graph Isomorphism Network (GIN)

    Parameters
    ----------
    num_features : int
        Number of input features per node.
    hidden_dim : int
        Dimension of hidden layers.
    num_classes : int
        Number of output features per node.
    nlayers : int, optional
        Number of layers in the GIN. Default: 2.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    with_bias : bool, optional
        Specifies whether to include bias parameters in the MLP layers. Default: True.
    device : str
        The device (cpu or gpu) on which the model will be run. Must be specified.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_nodes, output_dim), representing the predicted outcomes for each node after passing through the GIN. 
    """
    def __init__(self, num_features, num_classes, hidden_dim=16, nlayers=2, dropout=0.5,
                with_bias=True, device=None):

        super(GIN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        # store configuration for possible lazy-rebuild when input feature dim differs
        self._init_num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.nlayers = nlayers
        self.with_bias = with_bias
        self.dropout = dropout

        # build initial modules
        self._build_mlps(num_features)
        self.to(self.device)

    def _build_mlps(self, in_features):
        """Create MLPs and GINConv layers given input feature size."""
        mlp_first = MLP(
            [in_features, self.hidden_dim, self.hidden_dim],
            act="relu",
            norm="batch_norm",
            bias=self.with_bias
        )

        mlp_hidden = MLP(
            [self.hidden_dim, self.hidden_dim, self.hidden_dim],
            act="relu",
            norm="batch_norm",
            bias=self.with_bias
        )

        self.layers = nn.ModuleList([])
        if self.nlayers == 1:
            self.layers.append(GINConv(mlp_first))
        else:
            self.layers.append(GINConv(mlp_first))
            for i in range(self.nlayers-2):
                self.layers.append(GINConv(mlp_hidden))
            self.layers.append(GINConv(mlp_hidden))

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        # move modules to device
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Parameters:
        x : torch.Tensor
            Node feature matrix with shape (batch_size, num_nodes, num_features) or (num_nodes, num_features).
        edge_index : torch.Tensor
            Edge index in COO format with shape (2, num_edges).

        Returns:
        torch.Tensor
            Output from the network with shape (batch_size * num_nodes, num_classes) or (num_nodes, num_classes).
        """
        # If input feature dim doesn't match initialized mlps, rebuild
        actual_in = x.shape[-1]
        if hasattr(self, '_init_num_features') and self._init_num_features != actual_in:
            self._build_mlps(actual_in)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # apply final linear per-node
        if x.dim() == 3:
            b, n, f = x.shape
            x = x.view(b * n, f)
            x = self.fc(x)
            return x
        else:
            return self.fc(x)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
