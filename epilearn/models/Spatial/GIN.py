import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn.models import MLP
import torch

from .base import BaseModel

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
    def __init__(self, num_features, hidden_dim, num_classes, nlayers=2, dropout=0.5,
                with_bias=True, device=None):

        super(GIN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        
        mlp_one = MLP(
            [num_features, hidden_dim, num_classes],
            act="relu",
            norm="batch_norm",
            bias=with_bias
        )
        
        mlp_first = MLP(
            [num_features, hidden_dim, hidden_dim],
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
            [hidden_dim, hidden_dim, num_classes],
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
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout


    def forward(self, x, edge_index, edge_weight):
        """
        Parameters:
        x : torch.Tensor
            The input features tensor with shape (batch_size, num_nodes, num_features).
        edge_index : torch.Tensor
            The edge indices in COO format with shape (2, num_edges).
            
        Returns:
        torch.Tensor
            The output predictions for each node with shape (batch_size * num_nodes, num_classes).
        """
        #x, edge_index = data.x, data.edge_index
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
