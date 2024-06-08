# this code is adopted from https://github.com/LeronQ/STGCN-Pytorch/blob/main/stgcn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
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
                 num_timesteps_output, device = 'cpu'):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__(device=device)
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)


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
        out1 = self.block1(X, adj)
        out2 = self.block2(out1, adj)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4
    
    def initialize(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        self.last_temporal.reset_parameters()
        self.fully.reset_parameters()
