import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy.sparse as sp

from .base import BaseModel

def scatter_mul(src, index, dim_size):
    """
    Custom implementation of scatter multiplication using PyTorch
    """
    result = torch.ones(dim_size)
    # Convert index to numpy for indexing if it's not already
    if isinstance(index, torch.Tensor):
        index = index.numpy()
    # Group values by index
    for idx in range(dim_size):
        mask = (index == idx)
        if mask.any():
            result[idx] = torch.prod(src[mask])
    return result

class DMP(nn.Module):
    def __init__(self, 
                num_nodes,
                recover_rate=None,
                horizon=1,
                seed_list=[14,8],
                device = 'cpu'):
        super(DMP, self).__init__()

        self.n_nodes = num_nodes
        self.steps = horizon

        self.seed_list = seed_list
        self.nodes_gamma = recover_rate

    def cave_index(self, src_nodes, tar_nodes):
        edge_list = [(int(s), int(t)) for s, t in zip(src_nodes, tar_nodes)]
        E = len(edge_list)
        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        attr = {edge:w for edge, w in zip(edge_list, range(E))}
        nx.set_edge_attributes(G, attr, "idx")

        cave = []
        for edge in edge_list:
            if G.has_edge(*edge[::-1]):
                cave.append(G.edges[edge[::-1]]["idx"])
            else:
                cave.append(E)
        return np.array(cave)
    
    def edgeList(self, weight_adj):
        sp_mat = sp.coo_matrix(weight_adj)
        weight = sp_mat.data
        cave = self.cave_index(sp_mat.row, sp_mat.col)
        edge_list = np.vstack((sp_mat.row, sp_mat.col, weight, cave))
        return edge_list

    def mulmul(self, Theta_t):
        Theta = scatter_mul(Theta_t, self.tar_nodes, self.N) # [N]
        Theta = Theta[self.src_nodes] #[E]
        Theta_cav = scatter_mul(Theta_t, self.cave_index, self.E+1)[:self.E]

        mul = Theta / Theta_cav
        return mul

    def iteration(self):
        self.Theta_ij_t = self.Theta_ij_t - self.weights * self.Phi_ij_t
        new_Ps_ij_t = self.Ps_i_0 * self.mulmul(self.Theta_ij_t)
        self.Ps_ij_t_1 = self.Ps_ij_t
        self.Ps_ij_t = new_Ps_ij_t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_t - (self.Ps_ij_t-self.Ps_ij_t_1)

        # marginals
        self.Ps_t = self.Ps_0 * scatter_mul(self.Theta_ij_t, self.tar_nodes, self.N)
        self.Pr_t = self.Pr_t + self.nodes_gamma*self.Pi_t
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append([self.Ps_t, self.Pi_t, self.Pr_t])


    def _stop(self):
        I_former, R_former = self.marginals[-2][1:]
        I_later , R_later  = self.marginals[-1][1:]

        I_delta = torch.sum(torch.abs(I_former-I_later))
        R_delta = torch.sum(torch.abs(R_former-R_later))
        if I_delta>0.01 or R_delta>0.01:
            return False
        else:
            return True


    def forward(self, x, adj):
        """
        Parameters
        ----------
        x : torch.Tensor
            Expected shape (batch_size, num_timesteps_input, num_nodes, num_features).
        adj : torch.Tensor
            Adjacency matrix of the graph with shape (num_nodes, num_nodes), indicating connections between nodes.
    
        Returns
        -------
        torch.Tensor
            The output tensor of shape (horizon, num_nodes, 3), representing the predicted number of susceptible, infected, and recovered individuals at each timestep.
        """
        self.edge_list = self.edgeList(adj)
        # edge_list with size [3, E], (src_node, tar_node, weight) 
        self.src_nodes = torch.LongTensor(self.edge_list[0])
        self.tar_nodes = torch.LongTensor(self.edge_list[1])
        self.weights   = torch.FloatTensor(self.edge_list[2])
        self.cave_index = torch.LongTensor(self.edge_list[3])
        self.gamma = torch.FloatTensor(self.nodes_gamma)[self.src_nodes]
        self.nodes_gamma = torch.FloatTensor(self.nodes_gamma)
        
        self.N = max([torch.max(self.src_nodes), torch.max(self.tar_nodes)]).item()+1
        self.E = len(self.src_nodes)
        self.marginals = []

        self.seeds = torch.zeros(self.N)
        self.seeds[self.seed_list] = 1

        # initial
        self.Ps_0 = 1 - self.seeds
        self.Pi_0 = self.seeds
        self.Pr_0 = torch.zeros_like(self.seeds)

        self.Ps_i_0 = self.Ps_0[self.src_nodes]
        self.Pi_i_0 = self.Pi_0[self.src_nodes]
        self.Pr_i_0 = self.Pr_0[self.src_nodes]
        
        self.Phi_ij_0 = 1 - self.Ps_i_0
        self.Theta_ij_0 = torch.ones(self.E)      

        # first iteration, t=1
        self.Theta_ij_t = self.Theta_ij_0 - self.weights * self.Phi_ij_0 + 1E-10 # get rid of NaN
        self.Ps_ij_t_1 = self.Ps_i_0 # t-1
        self.Ps_ij_t = self.Ps_i_0 * self.mulmul(self.Theta_ij_t) # t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_0 - (self.Ps_ij_t-self.Ps_ij_t_1)

        # marginals
        self.Ps_t = self.Ps_0 * scatter_mul(self.Theta_ij_t, self.tar_nodes, self.N)
        self.Pr_t = self.Pr_0 + self.nodes_gamma*self.Pi_0
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append([self.Ps_t, self.Pi_t, self.Pr_t])
        
        while True:
            self.iteration()
            if self._stop():
                break
        # Output a size of [T, N, 3] Tensor， T starts from t=1
        marginals = [torch.stack(m, dim=1) for m in self.marginals]
        marginals = torch.stack(marginals, dim=0) 


        return marginals
