# this code is adopted from https://github.com/v1xerunt/STAN/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from .base import BaseModel


class WeightedSumConv(MessagePassing):
    def __init__(self):
        super(WeightedSumConv, self).__init__(aggr='add')  # 'add' 表示聚合方式为加法
    
    def forward(self, x, edge_index, edge_attr):
        # 触发消息传递
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        return x_j * edge_attr  # 消息 = 源节点特征 * 边权重
    
    def update(self, aggr_out):
        return aggr_out  # 聚合输出


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.conv = WeightedSumConv()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def forward(self, adj, h):
        z = self.fc(h)

        sparse_adj = adj.to_sparse_coo()
        src_index = sparse_adj.indices()[0]
        dst_index = sparse_adj.indices()[1]
        try:
            att_feat = torch.cat([z[:,src_index],z[:,dst_index]], dim = -1)
        except:
            att_feat = torch.cat([z[src_index],z[dst_index]], dim = -1)
        att_edge = F.leaky_relu(self.attn_fc(att_feat))

        output = self.conv(z, sparse_adj.indices(), att_edge)

        return output  #self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, adj, h):
        head_outs = [attn_head(adj, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=-1)
        else:
            return torch.mean(torch.stack(head_outs))
        
class STAN(BaseModel):
    """
    Spatio-Temporal Attention Network (STAN)

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
    population : float, optional
        Total population considered in the model. Default: 1e10.
    gat_dim1 : int
        Dimension of the output space for the first GAT layer. Default: 32.
    gat_dim2 : int
        Dimension of the output space for the second GAT layer. Default: 32.
    gru_dim : int
        Dimension of the hidden state for the GRU layer. Default: 32.
    num_heads : int
        Number of attention heads in the first GAT layer. Default: 1.
    device : str, optional
        The device (cpu or gpu) on which the model will be run. Default: 'cpu'.

    Returns
    -------
    tuple of torch.Tensor
        A tuple containing two tensors, each of shape (batch_size, num_timesteps_output, num_nodes, 1), representing the predicted values for newly infected and recovered individuals respectively for each node over future timesteps. The second tensor contains the physical model predictions.
    """
    def __init__(self, 
                 num_nodes, 
                 num_features, 
                 num_timesteps_input, 
                 num_timesteps_output, 
                 population=1e10, 
                 gat_dim1=32, 
                 gat_dim2=32, 
                 gru_dim=32, 
                 num_heads=1, 
                 device = 'cpu'):
        super(STAN, self).__init__()
        self.n_nodes = num_nodes
        self.nfeat = num_features
        self.history = num_timesteps_input
        self.horizon = num_timesteps_output
        self.pop = population
        
        self.layer1 = MultiHeadGATLayer(self.nfeat*self.history, gat_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(gat_dim1 * num_heads, gat_dim2, 1)

        self.pred_window = self.horizon
        # self.gru = nn.GRUCell(gat_dim2, gru_dim)
        self.gru = nn.GRU(gat_dim2, gru_dim, num_layers=1, dropout=0.5)
    
        self.nn_res_I = nn.Linear(gru_dim+2, self.horizon)
        self.nn_res_R = nn.Linear(gru_dim+2, self.horizon)

        self.nn_res_sir = nn.Linear(gru_dim+2, 2)
        
        self.hidden_dim2 = gat_dim2
        self.gru_dim = gru_dim
        self.device = device

    def forward(self, X, adj, states, dynamic_adj=None, N = None, h = None):
        """
        Parameters
        ----------
        X : torch.Tensor
            Input feature tensor with shape (batch_size, num_timesteps_input, num_nodes, num_features).
        adj : torch.Tensor
            Static adjacency matrix with shape (num_nodes, num_nodes).
        states : torch.Tensor
            States of the nodes, with the same shape as X, containing current infection and recovery data.
        dynamic_adj : torch.Tensor, optional
            Dynamic adjacency matrix, with shape similar to adj but possibly varying over time. Default: None.
        N : float, optional
            Total population considered in the model. Default: None.
        h : torch.Tensor, optional
            Hidden states for the GRU layer, used if provided. Default: None.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing two tensors:
            1. Predicted new infections and recoveries, shape (batch_size, num_timesteps_output, num_nodes, 2).
            2. Physical model predictions based on current states, shape (batch_size, num_timesteps_output, num_nodes, 2).
        """
        last_diff_I = X[:, -1, :, 1].unsqueeze(2)
        last_diff_R = X[:, -1, :, 2].unsqueeze(2)
        X = X.transpose(1,2).flatten(2,3)
        

        # if h is None:
        #     h = torch.zeros(X.shape[0], self.gru_dim).to(self.device)
        #     gain = nn.init.calculate_gain('relu')
        #     nn.init.xavier_normal_(h, gain=gain)  

        new_I = []
        new_R = []
        phy_I = []
        phy_R = []
        self.alpha_list = []
        self.beta_list = []
        self.alpha_scaled = []
        self.beta_scaled = [] 

        cur_h = self.layer1(adj, X)
        cur_h = F.elu(cur_h)
        cur_h = self.layer2(adj, cur_h)
        cur_h = F.elu(cur_h)
        
        h, last_h = self.gru(cur_h)
        hc = torch.cat((h, last_diff_I, last_diff_R) , dim = -1)

        pred_I = self.nn_res_I(hc).transpose(1,2).unsqueeze(-1)
        pred_R = self.nn_res_R(hc).transpose(1,2).unsqueeze(-1)

        pred_res = self.nn_res_sir(hc)

        alpha = torch.sigmoid(pred_res[..., 0])
        beta =  torch.sigmoid(pred_res[..., 1])

        phy_I = []
        phy_R = []
        if N is None:
            N = self.pop
        for i in range(self.horizon):
            last_I = states[:,-1,:, 1] if i == 0 else last_I + dI.detach()
            last_R = states[:,-1,:, 2] if i == 0 else last_R + dR.detach()

            last_S = N - last_I - last_R
            
            dI = alpha * last_I * (last_S/N) - beta * last_I
            dR = beta * last_I
            phy_I.append(dI)
            phy_R.append(dR)
        phy_I = torch.stack(phy_I).to(self.device).transpose(1,0).unsqueeze(-1)
        phy_R = torch.stack(phy_R).to(self.device).transpose(1,0).unsqueeze(-1)

        return torch.cat([pred_I, pred_R], dim=-1), torch.cat([phy_I, phy_R], dim=-1)
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
