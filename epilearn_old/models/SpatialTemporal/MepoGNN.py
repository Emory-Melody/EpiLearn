# this code is adopted from https://github.com/deepkashiwa20/MepoGNN/blob/main/model/MepoGNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        else:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = []
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class stcell(nn.Module):
    def __init__(self, num_nodes, dropout, in_dim, out_len, residual_channels, dilation_channels, skip_channels,
                 end_channels, kernel_size, blocks, layers):
        super(stcell, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1
        self.supports_len = 2

        for b in range(blocks):
            additional_scope = 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes, (2 ** layers - 1) * blocks + 2 - receptive_field]))
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))


        self.end_conv_b1 = nn.Conv2d(in_channels=skip_channels * blocks * layers,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_b2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_len,
                                    kernel_size=(1,1),
                                    bias=True)

        self.end_conv_g1 = nn.Conv2d(in_channels=skip_channels* blocks * layers,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_g2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_len,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, adp_g):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            res = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = torch.cat((s, skip[:, :, :,  -s.size(3):]), dim=1)
            except:
                skip = s

            x = self.gconv[i](x, adp_g)

            try:
                dense = dense[:, :, :, -x.size(3):]
            except:
                dense = 0
            dense = res[:, :, :, -x.size(3):] + dense

            gate = torch.sigmoid(x)
            x = x * gate + dense * (1 - gate)
            x = self.ln[i](x)

        param_b = F.relu(skip)
        param_b = F.relu(self.end_conv_b1(param_b))
        param_b = torch.sigmoid(self.end_conv_b2(param_b))

        param_g = F.relu(skip)
        param_g = F.relu(self.end_conv_g1(param_g))
        param_g = torch.sigmoid(self.end_conv_g2(param_g))

        return param_b, param_g


class SIRcell(nn.Module):
    def __init__(self):
        super(SIRcell, self).__init__()

    def forward(self, param_b: torch.Tensor, param_g: torch.Tensor, mob: torch.Tensor, SIR: torch.Tensor):
        if len(mob.shape) == 2:
            batch_size = SIR.shape[0]
            mob = mob.unsqueeze(0).expand(batch_size, -1, -1)
        num_node = SIR.shape[-2]
        S = SIR[..., [0]]
        I = SIR[..., [1]]
        R = SIR[..., [2]]
        pop = (S + I + R).expand(-1, num_node, num_node)
        propagtion = (mob/pop * I.expand(-1, num_node, num_node)).sum(1) +\
                     (mob/pop * I.expand(-1, num_node, num_node).transpose(1, 2)).sum(2)
        propagtion = propagtion.unsqueeze(2)

        I_new = param_b * propagtion
        R_t = I * param_g + R
        I_t = I + I_new - I * param_g
        S_t = S - I_new

        Ht_SIR = torch.cat((I_new, S_t, I_t, R_t), dim=-1)

        return Ht_SIR


class MepoGNN(BaseModel):
    """
    Meta-Population Graph Neural Network (MepoGNN)

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
    glm_type : str, optional
        Type of graph learning model ('Dynamic', 'Adaptive'). Default: 'Dynamic'.
    adapt_graph : tensor, optional
        Initial tensor for adaptive adjacency matrix. Only needed if glm_type is 'Adaptive'. Default: None.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    residual_channels : int
        Number of channels in residual layers.
    dilation_channels : int
        Number of channels in dilation layers.
    skip_channels : int
        Number of channels in skip connection layers.
    end_channels : int
        Number of channels in the final convolution layers.
    kernel_size : int
        Kernel size for the convolution layers.
    blocks : int
        Number of blocks in the WaveNet architecture.
    layers : int
        Number of layers in each block.
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
                 glm_type='Dynamic', 
                 adapt_graph=None, 
                 dropout=0.5, 
                 residual_channels=32,
                 dilation_channels=32, 
                 skip_channels=256, 
                 end_channels=512, 
                 kernel_size=2, 
                 blocks=2, 
                 layers=3, 
                 device = 'cpu'):
        super(MepoGNN, self).__init__()
        self.stcell = stcell(num_nodes, dropout, num_features, num_timesteps_output, residual_channels, dilation_channels,
                             skip_channels, end_channels, kernel_size, blocks, layers)
        self.SIRcell = SIRcell()
        self.out_dim = num_timesteps_output
        self.glm_type = glm_type
        self.device = device

        if self.glm_type == 'Adaptive':
            # To prevent parameter magnitude being too big
            log_g = torch.log(adapt_graph+1.0)
            self.max_log = log_g.max()
            # initialize g
            self.g_rescaled = nn.Parameter(log_g/self.max_log, requires_grad=True)

        elif self.glm_type == 'Dynamic':
            self.inc_init = nn.Parameter(torch.empty(num_timesteps_output, num_timesteps_input), requires_grad=True)
            nn.init.normal_(self.inc_init, 1, 0.01)
            self.od_scale_factor = 3
        else:
            raise NotImplementedError('Invalid graph type.')

    def forward(self, x, adj, states, dynamic_adj, max_od=1e6):
        """
        Parameters
        ----------
        x : torch.Tensor
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
        x_node = x
        od = dynamic_adj
        SIR = states

        x_node = x_node.transpose(1,3)
        if self.glm_type == 'Adaptive':
            mob = torch.exp(torch.relu(self.g_rescaled*self.max_log))
            g_adp = [mob / mob.sum(1, True), mob.T / mob.T.sum(1, True)]
            param_b, param_g = self.stcell(x_node, g_adp)
            outputs_SIR = []
            SIR = SIR[:, -1, ...]
            for i in range(self.out_dim):
                NSIR = self.SIRcell(param_b[:, i, ...], param_g[:, i, ...], mob, SIR)
                SIR = NSIR[..., 1:]
                outputs_SIR.append(NSIR[..., [0]])

        if self.glm_type == 'Dynamic':
            incidence = torch.softmax(self.inc_init, dim=1)
            mob = torch.einsum('kl,blnmc->bknmc', incidence, od).squeeze(-1)
            g = mob.mean(1)
            g_t = g.permute(0, 2, 1)
            g_dyn = [g / g.sum(2, True), g_t / g_t.sum(2, True)]
            param_b, param_g = self.stcell(x_node, g_dyn)
            outputs_SIR = []
            SIR = SIR[:, -1, ...]
            for i in range(self.out_dim):
                NSIR = self.SIRcell(param_b[:,i,...], param_g[:,i,...], mob[:,i,...]*max_od*self.od_scale_factor, SIR)
                SIR = NSIR[...,1:]
                outputs_SIR.append(NSIR[...,[0]])

        outputs = torch.stack(outputs_SIR, dim=1)

        return outputs.squeeze()
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
