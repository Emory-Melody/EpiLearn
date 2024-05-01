import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ...utils.utils import Degree_Matrix, Static_full, kronecker
from .base import BaseModel


#--------------For Fine Module---------------

class TimeEffect(nn.Module):
    def __init__(self, node_hid_size, out_size, num_timestamps, num_nodes):
        """
        Time-specific effect: Compute the fixed spatial dependency of each specific time slice
        """
        super(TimeEffect, self).__init__()

        self.gru = nn.GRU(input_size=node_hid_size, hidden_size=out_size, batch_first=True)

        self.num_timestamps = num_timestamps
        self.tot_nodes = num_nodes
        self.dim = self.num_timestamps * self.tot_nodes
        self.input_size = node_hid_size

    def forward(self, raw_features):
        node_features = raw_features.transpose(0, 1)
        temp_out, temp_hid = self.gru(node_features)

        temp = temp_out.transpose(0, 1)
        spatial_scores = torch.matmul(temp, temp.transpose(1, 2).contiguous()) / math.sqrt(self.input_size)
        spatial_weights = torch.sigmoid(spatial_scores)

        ST_weights = spatial_weights.repeat(self.num_timestamps, 1, 1)
        ST_weights = ST_weights.view(self.num_timestamps, self.num_timestamps, self.tot_nodes, self.tot_nodes)
        ST_weights = ST_weights.permute(0, 2, 1, 3)              # affected by fixed spatial dependency of each historical time slice
        ST_weights = ST_weights.reshape(self.dim, self.dim)

        return ST_weights


'''
Space-specific effect: Compute the fixed temporal weights at each specific region
'''
class GAT(nn.Module):

    def __init__(self, input_size):
        super(GAT, self).__init__()

        self.input_size = input_size

    def forward(self, input, adj):

        spatial_scores = torch.matmul(input, input.transpose(1, 2))/math.sqrt(self.input_size)
        zero_vec = -1e12 * torch.ones_like(spatial_scores)
        attention = torch.where(adj > 0, spatial_scores, zero_vec)
        attention = F.softmax(attention, dim=2)
        out_hid = torch.matmul(attention, input)

        return out_hid


class SpaceEffect(nn.Module):

    def __init__(self, node_input_size, num_nodes, num_timestamps):
        super(SpaceEffect, self).__init__()

        self.gat = GAT(node_input_size)

        self.num_timestamps = num_timestamps
        self.tot_nodes = num_nodes
        self.dim = self.num_timestamps * self.tot_nodes
        self.input_size = node_input_size

    def forward(self, raw_features, adj):
        node_features = self.gat(raw_features, adj)

        ## temporal dependency at each specific region
        spatial_features = node_features.transpose(0, 1)
        temp_scores = torch.matmul(spatial_features, spatial_features.transpose(1, 2).contiguous())/math.sqrt(self.input_size)
        spatial_weights = torch.sigmoid(temp_scores)

        ST_weights = spatial_weights.repeat(self.tot_nodes, 1, 1)
        ST_weights = ST_weights.view(self.tot_nodes, self.tot_nodes, self.num_timestamps, self.num_timestamps)
        ST_weights = ST_weights.permute(3, 1, 2, 0)            # affected by temporal dependency of self region
        ST_weights = ST_weights.reshape(self.dim, self.dim)

        return ST_weights


'''
Direct interaction effect: directly compute the weight between any two space-time points
'''
class DirectEffect(nn.Module):

    def __init__(self, input_size, num_nodes, num_timestamps):
        super(DirectEffect, self).__init__()

        self.num_timestamps = num_timestamps
        self.tot_nodes = num_nodes
        self.input_size = input_size


    def forward(self, raw_features):
        node_features = raw_features.view(raw_features.size()[0] * raw_features.size()[1], -1)
        st_scores = torch.matmul(node_features, node_features.transpose(0, 1).contiguous()) / math.sqrt(self.input_size)
        st_scores = torch.sigmoid(st_scores)

        return st_scores


'''
Integrate three NT*NT dimensional spatio-temporal weight matrices generated by different space-time effects
'''
class MultiEffectFusion(nn.Module):

    def __init__(self, input_size, num_nodes, num_timestamps):
        super(MultiEffectFusion, self).__init__()

        self.input_size = input_size
        self.tot_nodes = num_nodes
        self.num_timestamps = num_timestamps
        self.dim = num_timestamps * self.tot_nodes

        self.Time = TimeEffect(input_size, input_size, num_timestamps, num_nodes)
        self.Space = SpaceEffect(input_size, num_nodes, num_timestamps)
        self.Direct = DirectEffect(input_size, num_nodes, num_timestamps)

        self.TW = nn.Parameter(torch.FloatTensor(1, 1))
        self.SW = nn.Parameter(torch.FloatTensor(1, 1))
        self.DW = nn.Parameter(torch.FloatTensor(1, 1))


    def forward(self, his_raw_features, adj):     # dim(his_raw_features)=(timestamp, num_vertices, num_features)
        filter = Static_full(self.tot_nodes, self.num_timestamps, adj)

        temporal_matrix = self.Time(his_raw_features) * filter
        temporal_gate = self.TW * temporal_matrix

        spatial_matrix = self.Space(his_raw_features, adj) * filter
        spatial_gate = self.SW * spatial_matrix

        direct_matrix = self.Direct(his_raw_features) * filter
        direct_gate = self.DW * direct_matrix

        gates = torch.cat((temporal_gate.unsqueeze(0), spatial_gate.unsqueeze(0), direct_gate.unsqueeze(0)), 0)
        gates = F.softmax(gates, dim=0)

        ST_matrix = gates[0, :, :] * temporal_matrix + gates[1, :, :] * spatial_matrix + gates[2, :, :] * direct_matrix

        return ST_matrix


#--------------For Coarase Module---------------


'''
A general function to construct space-time neighboring blocks / space-time dependency structures
'''
def Static(n, t, A, rho_IT, rho_CT1, rho_CT2):
    """
    :param n: the dimension of the spatial adjacency matrix
    :param t: the length of periods
    :param A: the spatial adjacency matrix
    :param rho_IT: the trainable paramter of the current states of neighbors
    :param rho_CT1: the trainable paramter of the historical states of self
    :param rho_CT2: the trainable paramter of the historical states of neighbors
    :return: a space-time dependency structure matrix
    """
    I_S = torch.diag_embed(torch.ones(n))
    I_T = torch.diag_embed(torch.ones(t))

    C_S = A
    C_T = torch.tril(torch.ones(t, t), diagonal=-1)

    A_ST = kronecker(rho_IT * I_T, C_S) + kronecker(rho_CT1 * C_T, I_S) + kronecker(rho_CT2 * C_T, C_S)
    # A_ST = rho_CT1 * kronecker(C_T, I_S) + rho_CT2 * kronecker(C_T, C_S) + rho_IT * kronecker(I_T, C_S)

    return A_ST


'''
Coarse-grained module based on interventions
'''
class STNB_layer(nn.Module):
    # return NT * NT dimensional weight matrix for each type of space-time neighbor blocks
    def __init__(self, tot_nodes, num_timestamps, input_size):
        super(STNB_layer, self).__init__()
        self.dim = tot_nodes * num_timestamps
        self.num_timestamps = num_timestamps

        self.w1 = nn.Parameter(torch.FloatTensor(1, 1))
        self.w2 = nn.Parameter(torch.FloatTensor(1, 1))
        self.gate = nn.Sequential(nn.Linear(input_size, 1), nn.Sigmoid())

    def forward(self, features, interven, block_matrix, adj):
        judge = torch.sum(adj) * self.num_timestamps
        # process cumulative/current intervention
        block_sum = torch.sum(block_matrix)
        if block_sum > judge:
            interven_cum = torch.cumsum(interven, dim=0) - interven           # cumulative historical interventions
            interven_adjust = interven_cum.squeeze().view(self.dim, 1)
        else:
            interven_adjust = interven.squeeze().view(self.dim, 1)            # current intervention


        Infor = torch.mm(block_matrix, features)
        feat = self.w1 * Infor + interven_adjust.repeat(1,features.shape[1]) + self.w2 * features

        rho = self.gate(feat)
        # weights of all impact points in the same block are identical
        block_weight = torch.mul(rho.repeat(1, self.dim), block_matrix)

        return block_weight


class Coarse_module(nn.Module):
    # return NT * NT dimensional weight matrix of three types of space-time neighbor blocks
    def __init__(self, tot_nodes, num_timestamps, input_size):

        super(Coarse_module, self).__init__()
        self.tot_nodes = tot_nodes
        self.num_timestamps = num_timestamps
        self.input_size = input_size

        self.Gate_IT = STNB_layer(self.tot_nodes, num_timestamps, input_size)
        self.Gate_CS = STNB_layer(self.tot_nodes, num_timestamps, input_size)
        self.Gate_CT = STNB_layer(self.tot_nodes, num_timestamps, input_size)

    def forward(self, his_raw_features, interven, adj):
        # ipdb.set_trace()

        features = his_raw_features.contiguous().view(-1, self.input_size)
        A_IT = Static(self.tot_nodes, self.num_timestamps, adj, rho_IT=1, rho_CT1=0, rho_CT2=0)     # current states of neighbors
        A_CS = Static(self.tot_nodes, self.num_timestamps, adj, rho_IT=0, rho_CT1=1, rho_CT2=0)     # historical states of self
        A_CT = Static(self.tot_nodes, self.num_timestamps, adj, rho_IT=0, rho_CT1=0, rho_CT2=1)     # historical states of neighbors

        gate_IT = self.Gate_IT(features, interven, A_IT, adj)
        gate_CS = self.Gate_CS(features, interven, A_CS, adj)
        gate_CT = self.Gate_CT(features, interven, A_CT, adj)

        gate = gate_IT + gate_CS + gate_CT

        return gate

#--------------Downstream Model---------------

class Regression(nn.Module):

    def __init__(self, emb_size, out_size):
        super(Regression, self).__init__()

        self.layer = nn.Sequential(nn.Linear(emb_size, emb_size),
                                   nn.ReLU(),
                                   nn.Linear(emb_size, out_size),
                                   nn.ReLU())

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embds):
        logists = self.layer(embds)
        return logists


#--------------Main Model---------------

'''
Spatio-Temporal GNN
'''
class GConv(nn.Module):

    def __init__(self, D_temporal, A_temporal, tot_nodes, tw, fw):
        super(GConv, self).__init__()

        self.tot_nodes = tot_nodes
        self.sp_temp = torch.mm(D_temporal, torch.mm(A_temporal, D_temporal))

        self.his_temporal_weight = tw
        self.his_final_weight = fw

    def forward(self, his_raw_features):
        his_self = his_raw_features
        his_temporal = self.his_temporal_weight.repeat(self.tot_nodes, 1) * his_raw_features
        his_temporal = torch.mm(self.sp_temp, his_temporal)

        his_combined = torch.cat([his_self, his_temporal], dim=1)
        his_raw_features = torch.relu(his_combined.mm(self.his_final_weight))

        return his_raw_features
    

class DASTGN(BaseModel):
    def __init__(self, num_nodes,
                num_features,
                num_timesteps_input,
                num_timesteps_output,
                GNN_layers = 2,
                device = 'cpu'):
        super(DASTGN, self).__init__()

        self.device = device
        self.num_timestamps = num_timesteps_input
        self.input_size = num_features
        self.tot_nodes = num_nodes
        self.device = device
        self.GNN_layers = GNN_layers
        self.dim = self.num_timestamps * self.tot_nodes
        self.embd_size = self.num_timestamps * self.input_size
        self.out_size = num_timesteps_output

        self.his_temporal_weight = nn.Parameter(torch.FloatTensor(self.num_timestamps, self.input_size))
        self.his_final_weight = nn.Parameter(torch.FloatTensor(2 * self.input_size, self.input_size))
        self.final_weight = nn.Parameter(torch.FloatTensor(self.embd_size, self.embd_size))

        # a trainable parameter of the time-decaying interventions after quarantine
        self.theta = nn.Parameter(torch.FloatTensor(1, 1))

        self.Coarse_module = Coarse_module(self.tot_nodes, self.num_timestamps, self.input_size)
        self.Fine_module = MultiEffectFusion(self.input_size, self.tot_nodes, self.num_timestamps)

        self.output_module = Regression(emb_size=self.embd_size, out_size=self.out_size) 

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, x, adj, states = None, **kwargs):     # dim(his_raw_features)=(Time, Space, Feat)
        batch, w, n, f = x.shape
        x = x[..., :self.input_size]              # delete time and intervention variables

        if self.input_size == x.shape[-1]:
            interven_adjust = torch.zeros_like(x)[..., :1]
        else:
            time_interval = x[..., self.input_size:(self.input_size+1)]    # construct a piece-wise function for intervention
            interven = x[..., (self.input_size+1):(self.input_size+2)]
            interven_decay = torch.sigmoid(-self.theta * time_interval)
            interven_adjust = torch.where(interven == 0.5, interven_decay, interven)

        batch_embds = []
        for b in range(batch):
            features =  x[b]
            intervention = interven_adjust[b]
            for i in range(self.GNN_layers):
                features = features.contiguous().view(self.num_timestamps, self.tot_nodes, self.input_size)

                coarse_matrix = self.Coarse_module(features, intervention, adj)            # NT * NT
                fine_matrix = self.Fine_module(features, adj)                                 # NT * NT

                A_temporal = coarse_matrix * fine_matrix                                        # final ST weighted matrix
                D_temporal = Degree_Matrix(A_temporal)

                features = features.contiguous().view(-1, self.input_size)

                GCN = GConv(D_temporal, A_temporal, self.tot_nodes, self.his_temporal_weight, self.his_final_weight)
                features = GCN(features)

            his_list = []

            for timestamp in range(self.num_timestamps):
                st = timestamp * self.tot_nodes
                en = (timestamp + 1) * self.tot_nodes
                his_list.append(features[st:en, :])

            his_embds = torch.cat(his_list, dim=1)
            embds = his_embds
            embds = torch.relu(self.final_weight.mm(embds.t()).t())

            batch_embds.append(embds)

            
        batch_embds = torch.stack(batch_embds)

        out = self.output_module(batch_embds).transpose(1,2)

        return out
    
    def initialize(self):
        self.init_params()
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()



