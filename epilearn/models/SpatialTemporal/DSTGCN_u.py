# this code is adopted from https://github.com/LeronQ/STGCN-Pytorch/blob/main/stgcn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from copy import deepcopy

from torch_geometric.nn import GATConv
# from .gat_conv import GATConv
from torch_geometric.utils import dense_to_sparse

from .base import BaseModel


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        padding = (0, kernel_size // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # import ipdb; ipdb.set_trace()
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()




class GraphLearningLayer(nn.Module):
    """
    Layer to learn a graph structure (adjacency matrix) from node embeddings.
    """
    def __init__(self, num_nodes, embedding_dim, device='cpu'):
        """
        :param num_nodes: Number of nodes in the graph.
        :param embedding_dim: Dimension of the node embeddings.
        :param device: Device to run the model on.
        """
        super(GraphLearningLayer, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.device = device
        # Learnable node embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.node_embeddings.data.uniform_(-stdv, stdv)

    def forward(self, x=None):
        """
        Computes the learned adjacency matrix.
        :return: Learned adjacency matrix (num_nodes, num_nodes).
        """
        if x is None:
            # Compute adjacency matrix using dot product similarity
            adj_raw = torch.matmul(self.node_embeddings, self.node_embeddings.t())
        
            # Apply activation (e.g., ReLU) to get non-negative edge weights
            # You might explore other activations like Softmax for probabilistic edges
            adj_learned = F.relu(adj_raw)
        else:
            # non-learning based: compute adjacency from input features
            # x shape: (batch_size, num_nodes, num_timesteps, num_features)
            # average over batch and time to obtain node feature vectors
            feat = x.mean(dim=(0, 2))  # (num_nodes, num_features)
            # compute raw adjacency via feature similarity
            adj_raw = torch.matmul(feat, feat.t())
            adj_learned = F.relu(adj_raw)
        
        # Ensure self-loops (optional, but common)
        adj_learned = adj_learned + torch.eye(self.num_nodes, device=self.device)
        
        # Normalization (optional, e.g., row normalization)
        # D = torch.diag(torch.sum(adj_learned, dim=1)**(-0.5))
        # adj_normalized = torch.matmul(torch.matmul(D, adj_learned), D)
        # return adj_normalized
        
        return adj_learned




class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            concat=False, # Outputs [N, out_channels]
            dropout=0.5,
            bias=True
        )
        # Store batched edge index if adj is static (optimization)
        self._cached_batched_edge_index = None
        self._cached_batched_edge_weight = None
        self._cached_b_t_n = None # To check if cache is valid

    def _create_batched_graph(self, edge_index, edge_weight, b, t, n, device):
        """Helper to create batched edge_index and edge_weight."""
        num_edges = edge_index.size(1)
        
        # Create offsets: [0, N, 2N, ..., (B*T-1)N]
        offset = torch.arange(0, b * t, device=device) * n
        
        # Repeat edge_index B*T times and add offset
        # Shape: [2, E] -> [2, B*T*E]
        batched_edge_index = edge_index.repeat(1, b * t) 
        # Shape: [B*T] -> [1, B*T] -> [1, B*T*E] after repeat_interleave
        offset_repeated = offset.repeat_interleave(num_edges) 
        # Add offset to both source and target node indices
        batched_edge_index += offset_repeated.unsqueeze(0) 

        batched_edge_weight = None
        if edge_weight is not None:
            # Shape: [E] -> [B*T*E]
            batched_edge_weight = edge_weight.repeat(b * t)
            
        return batched_edge_index, batched_edge_weight

    def forward(self, adj, h):
        # adj shape: (num_nodes, num_nodes) - assuming static
        # h shape: (batch, num_nodes, timesteps, features)
        # import ipdb; ipdb.set_trace()
        b, n, t, f_in = h.size() # Original dimensions: B, N, T, F_in
        device = h.device

        # 1. Prepare node features for batching
        # Transpose to (B, T, N, F_in)
        h = h.transpose(1, 2).contiguous()
        # Reshape to (B*T, N, F_in) -> For intermediate checks if needed
        h_flat_intermediate = h.view(-1, n, f_in) 
        # Reshape for GATConv: (B*T*N, F_in)
        h_reshaped = h_flat_intermediate.view(-1, f_in) 

        # 2 & 3. Prepare batched graph structure (edge_index, edge_weight)
        # Optimization: Cache if B, T, N don't change and adj is static
        # Note: This caching assumes adj doesn't change between calls. 
        # If adj can change, remove caching or invalidate cache when adj changes.
        cache_key = (b, t, n)
        if self._cached_batched_edge_index is None or self._cached_b_t_n != cache_key:
            # Convert dense adj only once if needed
            edge_index, edge_weight = dense_to_sparse(adj) 
            
            self._cached_batched_edge_index, self._cached_batched_edge_weight = \
                self._create_batched_graph(edge_index, edge_weight, b, t, n, device)
            self._cached_b_t_n = cache_key
            
        batched_edge_index = self._cached_batched_edge_index
        batched_edge_weight = self._cached_batched_edge_weight

        # 4. Single GATConv Call
        # Input x: [B*T*N, F_in]
        # Input edge_index: [2, B*T*E]
        # Input edge_weight: [B*T*E] or None
        # Output: [B*T*N, F_out] where F_out = out_channels
        # import ipdb; ipdb.set_trace()

        out_reshaped = self.conv1(
            x=h_reshaped, 
            edge_index=batched_edge_index, 
            # edge_weight=batched_edge_weight
        )
        # Apply activation
        out_reshaped = F.elu(out_reshaped)
        
        f_out = out_reshaped.size(-1) # Get the output feature dimension

        # 5. Reshape Output back
        # [B*T*N, F_out] -> [B*T, N, F_out]
        h_processed_flat = out_reshaped.view(-1, n, f_out)
        # [B*T, N, F_out] -> [B, T, N, F_out]
        h_processed_batched = h_processed_flat.view(b, t, n, f_out)

        # Transpose back to original format expectation: (B, N, T, F_out)
        return h_processed_batched.transpose(1, 2).contiguous()






class DSTGCNBlock(nn.Module):
    """
    Dynamic Spatio-Temporal Graph Convolutional Network (DSTGCN)

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_features : int
        Number of features at each node per timestep.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    device : str, optional
        The device (cpu or gpu) on which the model will be run. Default: 'cpu'.

    Attributes
    ----------
    block1 : DSTGCNBlock
        First DSTGCN block which applies spatial and temporal convolutions.
    block2 : DSTGCNBlock
        Second DSTGCN block which applies further spatial and temporal convolutions.
    last_temporal : TimeBlock
        Temporal convolution block that processes the output of the last DSTGCN block.
    fully : torch.nn.Linear
        Fully connected layer to reshape the output into the desired number of future time steps.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_nodes, num_timesteps_output), representing the predicted values for each node over future timesteps.
        Each slice along the second dimension corresponds to a timestep, with each column representing a node.
    """
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, gat_heads=1, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step. (Also output of temporal1 and input to GCN).
        :param num_nodes: Number of nodes in the graph.
        """
        super(DSTGCNBlock, self).__init__()
        self.in_channels = in_channels
        self.spatial_channels = spatial_channels
        self.out_channels = out_channels # Output of temporal blocks & input to GCN's matrix multiply part

        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels, kernel_size=kernel_size) # Outputs 'out_channels' features
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
        #                                              spatial_channels)) # Maps 'out_channels' to 'spatial_channels'
        # self.bias1 = nn.Parameter(torch.FloatTensor(spatial_channels))
        self.Theta1 = nn.Parameter(torch.empty(out_channels, spatial_channels)) # Maps 'out_channels' to 'spatial_channels'
        self.bias1 = nn.Parameter(torch.empty(spatial_channels))
        # import ipdb; ipdb.set_trace()
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels, kernel_size=kernel_size) # Outputs 'out_channels' features
        self.batch_norm = nn.BatchNorm2d(num_nodes)

        # self.dropout = nn.Dropout(p=0.3)

        # self.gat = GAT(in_channels=spatial_channels, out_channels=out_channels, heads=gat_heads)
        
        # Residual connection for the GCN part
        # Input to GCN (after temporal1) has 'out_channels' features.
        # Output of GCN matrix multiplication has 'spatial_channels' features.
        self.residual_conv_gcn = None
        # self.residual_conv_gcn = nn.Conv2d(self.out_channels, self.spatial_channels, kernel_size=1)
    
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

        # Initialize bias1 to avoid NaNs with deterministic algorithms
        if self.bias1 is not None:
            fan_in = self.Theta1.shape[0]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias1.data.uniform_(-bound, bound)

        self.temporal1.reset_parameters()
        self.temporal2.reset_parameters()


    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # import ipdb; ipdb.set_trace()
        # First temporal block
        # t has shape (batch_size, num_nodes, num_timesteps, self.out_channels)
        t = self.temporal1(X)
        
        # GCN part
        if A_hat.dim() == 2:
            lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        elif A_hat.dim() == 3:
            # import ipdb; ipdb.set_trace()
            lfs = torch.einsum("kij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        elif A_hat.dim() == 4:
            # Average across the second dimension (horizon) to get shape [batch, N, N]
            A_hat_avg = A_hat.mean(dim=1)
            
            # Original computation for comparison
            lfs_original = torch.einsum("kij,jklm->kilm", [A_hat_avg, t.permute(1, 0, 2, 3)])
            
            # Decouple self and neighbor influence
            # Create diagonal mask for self-loops
            diag_mask = torch.eye(A_hat_avg.size(-1), device=A_hat_avg.device, dtype=A_hat_avg.dtype)
            
            # Split adjacency matrix
            A_self = A_hat_avg * diag_mask.unsqueeze(0)  # Self-loops only
            A_neighbor = A_hat_avg * (1 - diag_mask.unsqueeze(0))  # Neighbors only
            
            # Compute self and neighbor influences separately
            lfs_self = torch.einsum("kij,jklm->kilm", [A_self, t.permute(1, 0, 2, 3)])
            lfs_neighbor = torch.einsum("kij,jklm->kilm", [A_neighbor, t.permute(1, 0, 2, 3)])
            
            # Combine to get the same result as original
            lfs = lfs_self + lfs_neighbor
            
            # Sanity check: ensure decoupled computation matches original
            if not torch.allclose(lfs, lfs_original, atol=1e-6):
                raise RuntimeError(f"Sanity check failed: decoupled computation differs from original. "
                                f"Max difference: {torch.max(torch.abs(lfs - lfs_original)).item():.2e}")
            
        # import ipdb; ipdb.set_trace()
        gcn_matrix_mult = torch.matmul(lfs, self.Theta1) + self.bias1
        
        # Add residual to GCN output, then ReLU
        t2 = F.relu(gcn_matrix_mult)
        t3 = self.temporal2(t2)
        
        return self.batch_norm(t3)
    
    def freeze_mapping(self):
        """
        Freeze the temporal1.
        """
        for param in self.temporal1.parameters():
            param.requires_grad = False


class DSTGCN(BaseModel):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, nhids = 128, gat_heads=1, **kwargs):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        self.nhid = nhids
        self.spatial_nhid = nhids
        self.gsl_embed_dim = 1
        kernel_size = 1 # Kernel size for temporal convolutions
        # import ipdb; ipdb.set_trace()

        super(DSTGCN, self).__init__(device=kwargs.get('device', 'cpu'))
        self.graph_learner = GraphLearningLayer(num_nodes, self.gsl_embed_dim, self.device).to(self.device)
        self.block1 = DSTGCNBlock(in_channels=num_features, out_channels=self.nhid,
                                 spatial_channels=self.spatial_nhid, num_nodes=num_nodes, gat_heads=gat_heads, kernel_size=kernel_size).to(self.device)
        self.last_temporal = TimeBlock(in_channels=self.nhid, out_channels=self.nhid, kernel_size=kernel_size).to(self.device)
        # import ipdb; ipdb.set_trace()
        self.fully = nn.Linear((num_timesteps_input ) * self.nhid, num_timesteps_output).to(self.device)
        self.std_estimate = nn.Linear((num_timesteps_input ) * self.nhid, num_timesteps_output).to(self.device)


    def forward(self, X, adj=None, states=None, dynamic_adj=None, debug=False, y=None, **kargs):
        """
        Parameters
        ----------
        X : torch.Tensor
            Shape (batch_size, num_nodes, num_timesteps_input, num_features)
        adj : torch.Tensor
            Shape (num_nodes, num_nodes)

        Returns
        -------
        torch.Tensor
            Output shape (batch_size, num_timesteps_output, num_nodes)
        """
        if dynamic_adj is not None:
            dynamic_adj.diagonal(dim1=-2, dim2=-1).fill_(1)
            adj = dynamic_adj
        else:
            assert adj is not None, "Either static or dynamic adjacency matrix must be provided."
            adj.diagonal().fill_(1)

        # import ipdb; ipdb.set_trace()
        X = X.transpose(1, 2)

        # Debugging: Check gradients after block1
        out1 = self.block1(X, adj)
        final = self.last_temporal(out1)
        output = self.fully(final.reshape((final.shape[0], final.shape[1], -1)))
        std = torch.exp(self.std_estimate(final.reshape((final.shape[0], final.shape[1], -1))))  # Use exp to ensure positive values

        return {"mean": output, "std": std}

    def initialize(self):
        self.block1.reset_parameters()
        self.last_temporal.reset_parameters()
        self.fully.reset_parameters()



def is_diagonal(t, tol=1e-6):
    diag = torch.diag(torch.diag(t))
    return torch.allclose(t, diag, atol=tol)