# this code is adopted from https://github.com/HySonLab/pandemic_tgnn/blob/main/code/models_multiresolution.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .base import BaseModel


class MPNN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, dropout):
        super(MPNN_Encoder, self).__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.fc1 = nn.Linear(nfeat+2*nhid, nhid )
        self.fc2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, adj, x):
        
        lst = list()
        weight = adj.to_sparse().values()
        adj = adj.to_sparse().indices()
        lst.append(x)

        x = self.relu(self.conv1(x,adj,edge_weight=weight))
        x = self.bn1(x.permute((0,2,1))).permute((0,2,1))
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x.permute((0,2,1))).permute((0,2,1))
        x = self.dropout(x)
        lst.append(x)

        x = torch.cat(lst, dim=2)
        x = self.relu(self.fc1(x.contiguous().view(-1, x.size(2))))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x.view(-1, self.n_nodes, x.size(1))
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()





class ATMGNN(BaseModel):
    """
    Attention-based Temporal Multiresolution Graph Neural Network (ATMGNN) 

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_features : int
        Number of features per node per timestep.
    num_timesteps_input : int
        Number of timesteps considered for each input sample (window size).
    num_timesteps_output : int
        Number of output timesteps to predict.
    nhid : int, optional
        Number of hidden units in the network and the output size of graph convolution layers. Default: 256.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    nhead : int, optional
        Number of heads in the multi-head attention mechanism. Default: 1.
    num_clusters : list, optional
        List of integers defining the number of clusters for each multiresolution layer. Default: [10, 5].
    use_norm : bool, optional
        Whether to use normalization on outputs from each layer. Default: False.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output, num_nodes), representing the predicted values for each node over future timesteps.
        Each slice along the second dimension corresponds to a timestep, with each column representing a node.
    """
    def __init__(self, 
                num_nodes,
                num_features,
                num_timesteps_input,
                num_timesteps_output,
                nhid = 256,
                dropout = 0.5,
                nhead = 1, 
                num_clusters = [10, 5], 
                use_norm = False):
        
        super(ATMGNN, self).__init__()
        
        self.window = num_timesteps_input
        self.nout = num_timesteps_output
        self.n_nodes = num_nodes
        self.nhid = nhid
        self.nfeat = num_features
        self.nhead = nhead
        self.use_norm = use_norm

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+

        # Bottom encoder
        self.bottom_encoder = MPNN_Encoder(self.nfeat, nhid, nhid, self.n_nodes, dropout)

        # Number of clusters
        self.num_clusters = num_clusters

        # Multiresolution construction
        self.middle_linear = nn.ModuleList()
        self.middle_encoder = nn.ModuleList()

        for size in self.num_clusters:
            self.middle_linear.append(nn.Linear(nhid, size))
            self.middle_encoder.append(nn.Linear(nhid, nhid))

        # Mixing multiple resolutions together
        self.mix_1 = nn.Linear((len(self.num_clusters) + 1) * nhid, 512)
        self.mix_2 = nn.Linear(512, (len(self.num_clusters) + 1) * nhid)

        # +--------------------------+
        # | Attention/Self-Attention |
        # +--------------------------+

        self.self_attention = nn.MultiheadAttention((len(self.num_clusters) + 1) * nhid, self.nhead, dropout=dropout)
        self.linear_reduction = nn.Linear(self.window, 1)
        
        self.fc1 = nn.Linear((len(self.num_clusters) + 1) * nhid + self.window * self.nfeat, nhid)
        self.fc2 = nn.Linear(nhid, self.nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, x, adj, states=None, dynamic_adj=None, **kargs):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (batch_size, num_timestamps, num_nodes, num_features).
        adj : torch.Tensor
            Static adjacency matrix of the graph with shape (num_nodes, num_nodes).
        states : torch.Tensor, optional
            States of the nodes if available, with the same shape as x. Default: None.
        dynamic_adj : torch.Tensor, optional
            Dynamic adjacency matrix if available, with shape similar to adj but possibly varying over time. Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, num_timesteps_output, num_nodes),
            representing the predicted values for each node over the specified output timesteps.
        """
        lst = list()
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+
        x = x.view(-1, self.n_nodes, self.nfeat)

        # All latents
        all_latents = []

        # Bottom encoder
        bottom_latent = self.bottom_encoder(adj, x)

        all_latents.append(bottom_latent)

        # Product of all assignment matrices
        product = None

        # Multiresolution construction
        adj = adj.to_dense()
        latent = bottom_latent
        num_nodes = [self.n_nodes] + self.num_clusters
        for level in range(len(self.num_clusters)):
            size = self.num_clusters[level]

            # Assignment matrix
            assign = self.middle_linear[level](latent.view(-1, bottom_latent.shape[-1]))
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1).view(-1, num_nodes[level], assign.shape[-1])

            # Update product
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign)

            # Coarsen node features
            x = torch.matmul(assign.transpose(1, 2), latent)
            x = F.normalize(x, dim = 2)

            # Coarsen the adjacency
            adj = torch.matmul(torch.matmul(assign.transpose(1, 2), adj), assign)
            adj = adj / torch.sum(adj)

            # New latent by graph convolution
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x))) # h(AX)

            # Extended latent
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim = 2)
        x = representation

        # Mixing multiple resolutions
        x = torch.relu(self.mix_1(x.view(-1, x.shape[-1])))
        x = torch.relu(self.mix_2(x))

        # +------------------------------------------------------------------------------------------+
        # | Attention model module, self-attention variant with query, key, and value the same input |
        # +------------------------------------------------------------------------------------------+
        x = x.view(-1, self.window, self.n_nodes, x.shape[-1]) 
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes

        x, _ = self.self_attention(x, x, x)
        x = torch.transpose(x, 0, 2)
        x = self.linear_reduction(x)
        x = x.squeeze()
        x = torch.transpose(x, 0, 1)

        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = x.view(-1, x.shape[-1], self.n_nodes)

        return x
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    

class MPNN_LSTM(BaseModel):
    """
    Message Passing Neural Network combined with Long Short-Term Memory (MPNN_LSTM)
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_features : int
        Number of features in each timestep of the input data.
    num_timesteps_input : int
        Window size; number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    nhid : int, optional
        Number of hidden units in the LSTM and the number of output channels in GCN layers. Default: 256.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    device : str, optional
        The device (cpu or gpu) on which the model will be run. Default: 'cpu'.

    
    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output, num_nodes), representing the predicted values for each node over future timesteps.
        Each slice along the second dimension corresponds to a timestep, with each column representing a node.

    """
    def __init__(self, 
                num_nodes,
                num_features,
                num_timesteps_input,
                num_timesteps_output,
                nhid = 256,
                dropout = 0.5,
                device = 'cpu'):
        """
        Parameters:
        nfeat (int): Number of features
        nhid (int): Hidden size
        nout (int): Number of output features
        n_nodes (int): Number of nodes
        window (int): Window size
        dropout (float): Dropout rate
        
        Returns:
        x (torch.Tensor): Output of the model
        """
        super(MPNN_LSTM, self).__init__()
        self.window = num_timesteps_input
        self.n_nodes = num_nodes
        self.nhid = nhid
        self.nfeat = num_features
        self.nout = num_timesteps_output
        self.conv1 = GCNConv(self.nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm2d(nhid) # originally 1d
        self.bn2 = nn.BatchNorm2d(nhid)  
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2*nhid+self.window*self.nfeat, nhid)
        self.fc2 = nn.Linear( nhid, self.nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, x, adj, states=None, dynamic_adj=None, **kargs):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (batch_size, num_timestamps, num_nodes, num_features).
        adj : torch.Tensor
            Static adjacency matrix of the graph with shape (num_nodes, num_nodes).
        states : torch.Tensor, optional
            States of the nodes if available, with the same shape as x. Default: None.
        dynamic_adj : torch.Tensor, optional
            Dynamic adjacency matrix if available, with shape similar to adj but possibly varying over time. Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, num_timesteps_output, num_nodes),
            representing the predicted values for each node over the specified output timesteps.
        """
        lst = list()
        weight = adj.to_sparse().values()
        adj = adj.to_sparse().indices()

        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)
       
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)
        
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        x = self.bn1(x.view(-1, self.nhid, self.window, self.n_nodes)).view(-1, self.window, self.n_nodes, self.nhid)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        x = self.bn2(x.view(-1, self.nhid, self.window, self.n_nodes)).view(-1, self.window, self.n_nodes, self.nhid)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=-1)
        
        # reshape to (seq_len, batch_size , hidden) to fit the lstm
        x = x.view(-1, self.window, self.n_nodes, x.size(3))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
   
        x, (hn1, cn1) = self.rnn1(x)
        
        out2, (hn2,  cn2) = self.rnn2(x)
        
        # use the hidden states of both rnns 
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1, self.nout, self.n_nodes)
        
        return x
    
    def initialize(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.rnn1.reset_parameters()
        self.rnn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
