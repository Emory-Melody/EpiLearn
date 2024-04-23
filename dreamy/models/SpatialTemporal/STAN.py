# this code is adopted from https://github.com/v1xerunt/STAN/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))



# self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, device = 'cpu'

class STAN(BaseModel):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device = 'cpu'):
        super(STAN, self).__init__()
        self.g = g
        
        self.layer1 = MultiHeadGATLayer(self.g, in_dim, hidden_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(self.g, hidden_dim1 * num_heads, hidden_dim2, 1)

        self.pred_window = pred_window
        self.gru = nn.GRUCell(hidden_dim2, gru_dim)
    
        self.nn_res_I = nn.Linear(gru_dim+2, pred_window)
        self.nn_res_R = nn.Linear(gru_dim+2, pred_window)

        self.nn_res_sir = nn.Linear(gru_dim+2, 2)
        
        self.hidden_dim2 = hidden_dim2
        self.gru_dim = gru_dim
        self.device = device

    def forward(self, dynamic, cI, cR, N, I, R, h=None):
        num_loc, timestep, n_feat = dynamic.size()
        N = N.squeeze()

        if h is None:
            h = torch.zeros(1, self.gru_dim).to(self.device)
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(h, gain=gain)  

        new_I = []
        new_R = []
        phy_I = []
        phy_R = []
        self.alpha_list = []
        self.beta_list = []
        self.alpha_scaled = []
        self.beta_scaled = [] 

        for each_step in range(timestep):        
            cur_h = self.layer1(dynamic[:, each_step, :])
            cur_h = F.elu(cur_h)
            cur_h = self.layer2(cur_h)
            cur_h = F.elu(cur_h)
            
            cur_h = torch.max(cur_h, 0)[0].reshape(1, self.hidden_dim2)
            
            h = self.gru(cur_h, h)
            hc = torch.cat((h, cI[each_step].reshape(1,1), cR[each_step].reshape(1,1)),dim=1)
            
            pred_I = self.nn_res_I(hc)
            pred_R = self.nn_res_R(hc)
            new_I.append(pred_I)
            new_R.append(pred_R)

            pred_res = self.nn_res_sir(hc)
            alpha = pred_res[:, 0]
            beta =  pred_res[:, 1]
            
            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
            alpha = torch.sigmoid(alpha)
            beta = torch.sigmoid(beta)
            self.alpha_scaled.append(alpha)
            self.beta_scaled.append(beta)
            
            cur_phy_I = []
            cur_phy_R = []
            for i in range(self.pred_window):
                last_I = I[each_step] if i == 0 else last_I + dI.detach()
                last_R = R[each_step] if i == 0 else last_R + dR.detach()

                last_S = N - last_I - last_R
                
                dI = alpha * last_I * (last_S/N) - beta * last_I
                dR = beta * last_I
                cur_phy_I.append(dI)
                cur_phy_R.append(dR)
            cur_phy_I = torch.stack(cur_phy_I).to(self.device).permute(1,0)
            cur_phy_R = torch.stack(cur_phy_R).to(self.device).permute(1,0)

            phy_I.append(cur_phy_I)
            phy_R.append(cur_phy_R)

        new_I = torch.stack(new_I).to(self.device).permute(1,0,2)
        new_R = torch.stack(new_R).to(self.device).permute(1,0,2)
        phy_I = torch.stack(phy_I).to(self.device).permute(1,0,2)
        phy_R = torch.stack(phy_R).to(self.device).permute(1,0,2)

        self.alpha_list = torch.stack(self.alpha_list).squeeze()
        self.beta_list = torch.stack(self.beta_list).squeeze()
        self.alpha_scaled = torch.stack(self.alpha_scaled).squeeze()
        self.beta_scaled = torch.stack(self.beta_scaled).squeeze()
        return new_I, new_R, phy_I, phy_R, h