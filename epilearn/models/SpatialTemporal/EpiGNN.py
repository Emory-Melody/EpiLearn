# this code is adopted from https://github.com/Xiefeng69/EpiGNN/blob/main/src/models.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .base import BaseModel

def getLaplaceMat(batch_size, m, adj):
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj>0, o_mat, adj)
    '''
    d_mat = torch.bmm(adj, adj.permute(0, 2, 1))
    d_mat = torch.where(i_mat>0, d_mat, i_mat)
    print('d_mat version 1', d_mat)
    '''
    d_mat_in = torch.sum(adj, dim=1)
    d_mat_out = torch.sum(adj, dim=2)
    d_mat = torch.sum(adj, dim=2) # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    #d_mat = torch.pow(d_mat, -0.5) if is 1/2
    d_mat = torch.pow(d_mat, -1)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    #laplace_mat = torch.bmm(laplace_mat, d_mat)
    return laplace_mat


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class ConvBranch(nn.Module):
    def __init__(self,
                 m,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_factor,
                 hidP=1,
                 isPool=True):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), dilation=(dilation_factor,1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        #self.activate = nn.Tanh()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x

class RegionAwareConv(nn.Module):
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.nfeat = nfeat
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(m=self.m, in_channels=nfeat, out_channels=self.k, kernel_size=3, dilation_factor=1, hidP=self.hidP)
        self.conv_l2 = ConvBranch(m=self.m, in_channels=nfeat, out_channels=self.k, kernel_size=5, dilation_factor=1, hidP=self.hidP)
        self.conv_p1 = ConvBranch(m=self.m, in_channels=nfeat, out_channels=self.k, kernel_size=3, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_p2 = ConvBranch(m=self.m, in_channels=nfeat, out_channels=self.k, kernel_size=5, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_g = ConvBranch(m=self.m, in_channels=nfeat, out_channels=self.k, kernel_size=self.P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()
    
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = x.contiguous().view(-1, self.nfeat, self.P, self.m)
        batch_size = x.shape[0]
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x


class EpiGNN(BaseModel):
    """
    Epidemiological Graph Neural Network (EpiGNN)

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
    k : int, optional
        Number of local neighborhoods to consider in the graph learning layer. Default: 8.
    hidA : int, optional
        Dimension of attention in the model. Default: 64.
    hidR : int, optional
        Dimension of hidden layers in the recurrent neural network part. Default: 40.
    hidP : int, optional
        Dimension of positional encoding in the model. Default: 1.
    n_layer : int, optional
        Number of layers in the graph neural network. Default: 2.
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
                k = 8, 
                hidA = 128, 
                hidR = 32, 
                hidP=1, 
                n_layer = 2, 
                dropout = 0, 
                nhids=None,
                device = 'cpu'):
        super().__init__(device)
        # arguments setting
        self.nfeat = num_features
        self.m = num_nodes
        self.w = num_timesteps_input
        self.droprate = dropout
        self.hidR = hidR
        self.hidA = hidA
        self.hidP = hidP
        self.k =k
        self.n = n_layer
        self.dropout = nn.Dropout(self.droprate)

        # unavailable settings
        self.res = 0
        self.hw = 0
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.extra = False

        # if args.extra:
        #     self.extra = True
        #     self.external = data.external
        # else:
        #     self.extra = False

        # Feature embedding
        self.hidR = self.k*4*self.hidP + self.k
        self.backbone = RegionAwareConv(nfeat = num_features, P=self.w, m=self.m, k=self.k, hidP=self.hidP)

        # global
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local
        self.s_enc = nn.Linear(1, self.hidR)

        # external resources
        self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for i in range(self.n)])
        #self.GCNBlock1 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
        #self.GCNBlock2 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)

        # prediction
        # import ipdb; ipdb.set_trace()
        if self.res == 0:
            self.output = nn.Linear(self.hidR*2, num_timesteps_output)
        else:
            self.output = nn.Linear(self.hidR*(self.n+1), num_timesteps_output)

        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def forward(self, X, adj, states=None, dynamic_adj=None, index=None):
        """
        Parameters
        ----------
        X : torch.Tensor
            Input features tensor with shape (batch_size, num_timesteps_input, num_nodes, num_features).
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
        adj = adj.bool().float()
        #print(index.shape) batch_size
        batch_size = X.shape[0] # batchsize, w, m, feat

        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        temp_emb = self.backbone(X)

        # step 2: generate global transmission risk encoding.
        query = self.WQ(temp_emb) # batch, N, hidden
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        #attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # step 3: generate local transmission risk encoding.
        # print(self.degree.shape) [self.m]
        d = adj.sum(0).unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion.
        feat_emb = temp_emb + t_enc + s_enc

        # step 4: Region-Aware Graph Learner
        # load external resource
        if self.extra:
            extra_adj_list=[]
            zeros_mt = torch.zeros((self.m, self.m)).to(self.adj.device)
            #print(self.external.shape)
            for i in range(batch_size):
                offset = 20
                if i-offset>=0:
                    idx = i-offset
                    extra_adj_list.append(self.external[index[i],:,:].unsqueeze(0))
                else:
                    extra_adj_list.append(zeros_mt.unsqueeze(0))
            extra_info = torch.concat(extra_adj_list, dim=0) # [1872, 52]
            extra_info = extra_info
            #print(extra_info.shape) # batch_size, self.m self.m
            external_info = torch.mul(self.external_parameter, extra_info)
            external_info = F.relu(external_info)
            #print(self.external_parameter)

        # apply Graph Learner to generate a graph
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, adj)
        adj = self.graphGen(temp_emb)
        
        # if additional information => fusion
        if self.extra:
            adj = adj + spatial_adj + external_info
        else:
            adj = adj + spatial_adj

        # get laplace adjacent matrix
        laplace_adj = getLaplaceMat(batch_size, self.m, adj)
        
        # Graph Convolution Network
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)
        '''
        if self.res == 1:
            node_state = torch.cat(node_state_list, dim=-1)
        
        node_state = self.GCNBlock1(feat_emb, laplace_adj)
        node_state = self.dropout(node_state)
        node_state = self.GCNBlock2(node_state, laplace_adj)
        node_state = self.dropout(node_state)
        '''

        # Final prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state)
        # highway means autoregressive model
        if self.hw > 0:
            z = X[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        

        # import ipdb; ipdb.set_trace()

        return res
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
