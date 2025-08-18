import torch.nn as nn
import torch.nn.functional as F
import torch

from .base import BaseModel


class GATConv(nn.Module):
    """Graph Attention Network Convolution layer"""
    
    def __init__(self, in_channels, out_channels, heads=1, concat=True, bias=True, dropout=0.0):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        """GAT forward pass"""
        if x.dim() == 3:
            # Batch processing
            batch_size, num_nodes, _ = x.shape
            outputs = []
            
            for b in range(batch_size):
                x_sample = x[b]  # [num_nodes, in_channels]
                out = self._forward_single(x_sample, edge_index, edge_attr)
                outputs.append(out)
            
            return torch.stack(outputs, dim=0)
        else:
            return self._forward_single(x, edge_index, edge_attr)
    
    def _forward_single(self, x, edge_index, edge_attr=None):
        """Forward pass for single sample"""
        H, C = self.heads, self.out_channels
        
        x = self.lin(x).view(-1, H, C)  # [N, H, C]
        
        if edge_index.size(1) == 0:
            # No edges, return transformed features
            if self.concat:
                out = x.view(-1, H * C)
            else:
                out = x.mean(dim=1)
            
            if self.bias is not None:
                out = out + self.bias
            return out
        
        # Compute attention coefficients
        row, col = edge_index
        
        # Create attention features
        x_i = x[row]  # [E, H, C]
        x_j = x[col]  # [E, H, C]
        
        # Concatenate for attention computation
        att_input = torch.cat([x_i, x_j], dim=-1)  # [E, H, 2*C]
        
        # Compute attention scores
        alpha = (att_input * self.att).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Apply softmax per node
        num_nodes = x.size(0)
        alpha_sparse = torch.zeros(num_nodes, edge_index.size(1), H, device=x.device)
        alpha_sparse[row, torch.arange(edge_index.size(1)), :] = alpha
        alpha = F.softmax(alpha_sparse, dim=1)[row, torch.arange(edge_index.size(1)), :]
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate messages
        out = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            src, dst = row[i], col[i]
            out[dst] += alpha[i].unsqueeze(-1) * x_j[i]
        
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
            
        if self.bias is not None:
            out = out + self.bias
            
        return out

class GAT(BaseModel):
    """
    Graph Attention Network (GAT)
    
    Parameters
    ----------
    num_features : int
        Number of input features per node.
    hidden_dim : int
        Dimension of hidden layers.
    num_classes : int
        Number of output features per node.
    nlayers : int, optional
        Number of layers in the GAT. Default: 2.
    nheads : list of int
        Number of attention heads in each GAT layer. Length must match `nlayers`.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    with_bn : bool, optional
        Specifies whether batch normalization should be included. Default: False.
    with_bias : bool, optional
        Specifies whether to include bias parameters in the attention calculations. Default: True.
    device : torch.device
        The device (cpu or gpu) on which the model will be run.
    concat : bool, optional
        Specifies whether to concatenate the outputs of the attention heads instead of averaging them. Default: False.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_nodes, output_dim), representing the predicted values for each node over future timesteps.
    """
    def __init__(self, num_features, hidden_dim, num_classes, nlayers=2, nheads=[2, 2], dropout=0.5,
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
                #assert num_classes % nheads[i] == 0, "Output_dim should be divisible by nheads!"
        
        if concat:
            if nlayers == 1:
                self.layers.append(GATConv(num_features, hidden_dim//nheads[0], bias=with_bias, heads=nheads[0], concat=concat))
            else:
                self.layers.append(GATConv(num_features, hidden_dim//nheads[0], bias=with_bias, heads=nheads[0], concat=concat))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
                        
                for i in range(nlayers-2):
                    self.layers.append(GATConv(hidden_dim, hidden_dim//nheads[i+1], bias=with_bias, heads=nheads[i+1], concat=concat))
                    if with_bn:
                        self.bns.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(GATConv(hidden_dim, hidden_dim//nheads[-1], bias=with_bias, heads=nheads[-1], concat=concat))
            
            
        else:
            if nlayers == 1:
                self.layers.append(GATConv(num_features, hidden_dim, bias=with_bias, heads=nheads[0], concat=concat))
            else:
                self.layers.append(GATConv(num_features, hidden_dim, bias=with_bias, heads=nheads[0], concat=concat))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
                        
                for i in range(nlayers-2):
                    self.layers.append(GATConv(hidden_dim, hidden_dim, bias=with_bias, heads=nheads[i+1], concat=concat))
                    if with_bn:
                        self.bns.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(GATConv(hidden_dim, hidden_dim, bias=with_bias, heads=nheads[-1], concat=concat))
                
            
        self.fc = nn.Linear(hidden_dim, num_classes)
            
        self.dropout = dropout
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (num_nodes, num_features).
        edge_index : torch.Tensor
            Tensor defining the edges of the graph with shape (2, num_edges), where each column represents an edge as a pair of node indices.
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,). Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (num_nodes, num_classes), representing the predicted values for each node.
        """
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.reshape(x, (x.shape[0], -1))

        for i, layer in enumerate(self.layers):
            if edge_weight is not None:
                x = layer(x.float(), edge_index=edge_index, edge_attr=edge_weight)
            else:
                x = layer(x.float(), edge_index=edge_index)
            if i != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x) #F.log_softmax(x, dim=1)
        #return F.log_softmax(self.fc(x), dim=1)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
