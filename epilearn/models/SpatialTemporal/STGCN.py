# this code is adopted from https://github.com/LeronQ/STGCN-Pytorch/blob/main/stgcn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
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


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GAT, self).__init__()

        self.conv1 = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            concat=False,
            dropout=0.5,
            bias=True
        )
    
    # def forward(self, adj, h):
    #     # import ipdb; ipdb.set_trace()
    #     edge_index, edge_weight = dense_to_sparse(adj)
    #     shapes = h.size()
    #     h = h.transpose(1, 2)
    #     h = h.contiguous().view(-1, h.size(2), h.size(3))
    #     for t in range(h.shape[0]):
    #         h[t] = F.elu(self.conv1(h[t], edge_index))
    #     h = h.view(shapes)
    #     return h  # Reshape back to original dimensions
    
    def forward(self, adj, h):
        edge_index, _ = dense_to_sparse(adj)
        original_shape = h.size()  # (batch, num_nodes, timesteps, features)
        # Change to: (batch, timesteps, num_nodes, features)
        h = h.transpose(1, 2).contiguous()
        # Flatten batch and temporal dimensions ==> (batch * timesteps, num_nodes, features)
        h_flat = h.view(-1, h.size(2), h.size(3))
        # Apply GATConv for each time step without in-place modifications
        out_list = [F.elu(self.conv1(h_flat[t], edge_index)) for t in range(h_flat.size(0))]
        h_processed = torch.stack(out_list, dim=0)
        # Restore original dimensions
        h_processed = h_processed.view(h.size(0), h.size(1), h.size(2), h.size(3))
        # Return tensor with original shape: (batch, num_nodes, timesteps, features)
        return h_processed.transpose(1, 2)



class STGCNBlock(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.

    Spatio-Temporal Graph Convolutional Network (STGCN)

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_features : int
        Number of features per node per timestep.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    device : str, optional
        The device (cpu or gpu) on which the model will be run. Default: 'cpu'.

    Attributes
    ----------
    block1 : STGCNBlock
        First STGCN block which applies spatial and temporal convolutions.
    block2 : STGCNBlock
        Second STGCN block which applies further spatial and temporal convolutions.
    last_temporal : TimeBlock
        Temporal convolution block that processes the output of the last STGCN block.
    fully : torch.nn.Linear
        Fully connected layer to reshape the output into the desired number of future time steps.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_nodes, num_timesteps_output), representing the predicted values for each node over future timesteps.
        Each slice along the second dimension corresponds to a timestep, with each column representing a node.
    """
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.gat = GAT(in_channels=spatial_channels, out_channels=out_channels, heads=4)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

        self.temporal1.reset_parameters()
        self.temporal2.reset_parameters()
        self.batch_norm.reset_parameters()


    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        # import ipdb; ipdb.set_trace()
        #---------GCN
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        #---------GAT
        # t2 = self.gat(A_hat, t)

        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(BaseModel):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, nhids = 128, device='cpu', **kwargs):
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
        super(STGCN, self).__init__(device=device)
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=self.nhid,
                                 spatial_channels=self.spatial_nhid, num_nodes=num_nodes).to(self.device)
        self.block2 = STGCNBlock(in_channels=self.nhid, out_channels=self.nhid,
                                 spatial_channels=self.spatial_nhid, num_nodes=num_nodes).to(self.device)
        # self.block3 = STGCNBlock(in_channels=self.nhid, out_channels=self.nhid,
        #                          spatial_channels=self.spatial_nhid, num_nodes=num_nodes).to(self.device)
        self.last_temporal = TimeBlock(in_channels=self.nhid, out_channels=self.nhid).to(self.device)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * self.nhid,
                               num_timesteps_output).to(self.device)
        # import ipdb; ipdb.set_trace()   

    def forward(self, X, adj, states=None, dynamic_adj=None, **kargs):
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
        # # import ipdb; ipdb.set_trace()
        # out1 = self.block1(X, adj)
        # out2 = self.block2(out1, adj)
        # out3 = self.block3(out2, adj)
        # out4 = self.last_temporal(out3)
        # # import ipdb; ipdb.set_trace()
        # out5 = self.fully(out4.reshape((out4.shape[0], out4.shape[1], -1)))
        # return out5

        # import ipdb; ipdb.set_trace()
        adj.diagonal().fill_(1)
        # import ipdb; ipdb.set_trace()
        out1 = self.block1(X, adj)
        out2 = self.block2(out1, adj)
        final = self.last_temporal(out2)
        # import ipdb; ipdb.set_trace()
        output = self.fully(final.reshape((final.shape[0], final.shape[1], -1)))
        return output

                # import ipdb; ipdb.set_trace()
        # adj.diagonal().fill_(1)
        # # import ipdb; ipdb.set_trace()
        # out1 = self.block1(X, adj)
        # final = self.last_temporal(out1)
        # # import ipdb; ipdb.set_trace()
        # output = self.fully(final.reshape((final.shape[0], final.shape[1], -1)))
        # return output
    
    def initialize(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        self.last_temporal.reset_parameters()
        self.fully.reset_parameters()
