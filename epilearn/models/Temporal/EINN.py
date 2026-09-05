"""EINN: Epidemiologically-Informed Neural Networks.

Rodriguez et al. (AAAI 2023), adapted from
https://github.com/AdityaLab/EINNs.

Simplified architecture preserving the core innovation:
- GRU with attention encoder (from original feature module)
- GRU decoder predicting SEIR compartment states
- Learnable SEIR ODE as soft finite-difference regularizer
- No autograd-through-time; no separate time module

The ODE regularizer encourages predicted state trajectories to satisfy
SEIR dynamics without the gradient conflicts of the original autograd approach.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn.parameter import Parameter

from .base import BaseModel


class TransformerAttn(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super().__init__()
        self.q_proj = nn.Linear(dim_in, dim_k)
        self.k_proj = nn.Linear(dim_in, dim_k)
        self.v_proj = nn.Linear(dim_in, dim_v)
        self.scale = math.sqrt(dim_k)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v.transpose(0, 1)).transpose(0, 1)
        return out


class EmbedAttenSeq(nn.Module):
    def __init__(self, dim_seq_in, rnn_out=40, dim_out=50, n_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_size=dim_seq_in, hidden_size=rnn_out,
                          num_layers=n_layers, dropout=dropout)
        self.attn_layer = TransformerAttn(rnn_out, rnn_out, rnn_out)
        self.out_layer = nn.Sequential(
            nn.Linear(rnn_out, dim_out), nn.Tanh(), nn.Dropout(dropout)
        )

    def forward(self, seqs):
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        return self.out_layer(latent_seqs)


class SEIRModule(nn.Module):
    """Learnable SEIR compartmental model.

    Parameters constrained to [0, 1] via tanh (matching original).
    Computes expected state derivatives given current states.
    """
    def __init__(self):
        super().__init__()
        self.log_alpha = Parameter(torch.tensor(0.0))
        self.log_beta = Parameter(torch.tensor(0.0))
        self.log_gamma = Parameter(torch.tensor(0.0))
        self.log_mu = Parameter(torch.tensor(-2.0))

    def get_params(self):
        alpha = (torch.tanh(self.log_alpha) + 1) * 0.5
        beta = (torch.tanh(self.log_beta) + 1) * 0.5
        gamma = (torch.tanh(self.log_gamma) + 1) * 0.5
        mu = (torch.tanh(self.log_mu) + 1) * 0.5
        return alpha, beta, gamma, mu

    def forward(self, state):
        """Compute SEIR derivatives from normalized states.

        state: (batch, 5) representing (S, E, I, R, M) in normalized units.
        Returns: (batch, 5) of (dS, dE, dI, dR, dM).
        """
        alpha, beta, gamma, mu = self.get_params()
        S, E, I, R, M = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]

        S_pos = F.softplus(S)
        E_pos = F.softplus(E)
        I_pos = F.softplus(I)

        dSE = beta * S_pos * I_pos
        dEI = alpha * E_pos
        dIR = gamma * I_pos
        dIM = mu * I_pos

        dS = -dSE
        dE = dSE - dEI
        dI = dEI - dIR - dIM
        dR = dIR
        dM = dIM

        return torch.stack([dS, dE, dI, dR, dM], dim=1)


class EINNModel(BaseModel):
    def __init__(self, device='cpu', lookback=16, horizon=4, n_features=1,
                 num_features=None, num_timesteps_input=None, num_timesteps_output=None,
                 hidden_dim=40, n_states=5, n_layers=1,
                 ode_lambda=0.1, lr=0.001, **kwargs):
        super().__init__(device=device)
        if num_features is not None:
            n_features = num_features
        if num_timesteps_input is not None:
            lookback = num_timesteps_input
        if num_timesteps_output is not None:
            horizon = num_timesteps_output
        self.lookback = lookback
        self.horizon = horizon
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_states = n_states
        self.ode_lambda = ode_lambda
        self.lr = lr

        self.encoder = EmbedAttenSeq(
            dim_seq_in=n_features, rnn_out=hidden_dim,
            dim_out=hidden_dim, n_layers=n_layers
        )

        self.decoder = nn.GRU(
            input_size=1, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True
        )
        self.n_layers_dec = n_layers

        self.state_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_states),
        )

        self.incidence_proj = nn.Linear(n_states, 1)

        self.ode_module = SEIRModule()

        self.to(device)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        self.ode_module.log_alpha.data.fill_(0.0)
        self.ode_module.log_beta.data.fill_(0.0)
        self.ode_module.log_gamma.data.fill_(0.0)
        self.ode_module.log_mu.data.fill_(-2.0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)

        encoded = self.encoder(x.transpose(0, 1))

        t_input = torch.linspace(0, 1, self.horizon, device=x.device)
        t_input = t_input.view(1, self.horizon, 1).expand(batch_size, -1, -1)

        h0 = encoded.unsqueeze(0).expand(self.n_layers_dec, -1, -1).contiguous()
        dec_out, _ = self.decoder(t_input, h0)

        states = self.state_proj(dec_out)
        incidence = self.incidence_proj(states).squeeze(-1)

        return incidence

    def _forward_with_states(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)

        encoded = self.encoder(x.transpose(0, 1))

        t_input = torch.linspace(0, 1, self.horizon, device=x.device)
        t_input = t_input.view(1, self.horizon, 1).expand(batch_size, -1, -1)

        h0 = encoded.unsqueeze(0).expand(self.n_layers_dec, -1, -1).contiguous()
        dec_out, _ = self.decoder(t_input, h0)

        states = self.state_proj(dec_out)
        incidence = self.incidence_proj(states).squeeze(-1)

        return incidence, states

    def _compute_ode_reg(self, states):
        """Soft ODE regularizer: finite-difference vs SEIR equations."""
        if self.horizon < 2:
            return torch.tensor(0.0, device=states.device)

        delta_states = states[:, 1:, :] - states[:, :-1, :]
        seir_expected = self.ode_module(states[:, :-1, :].reshape(-1, self.n_states))
        seir_expected = seir_expected.reshape(states.size(0), self.horizon - 1, self.n_states)

        ode_reg = F.mse_loss(delta_states, seir_expected)
        return ode_reg

    def _compute_loss(self, x, target):
        incidence, states = self._forward_with_states(x)

        data_loss = F.mse_loss(incidence, target)
        ode_reg = self._compute_ode_reg(states)

        return data_loss + self.ode_lambda * ode_reg

    def fit(self, train_input, train_target, train_states=None, train_graph=None,
            train_dynamic_graph=None, val_input=None, val_target=None,
            val_states=None, val_graph=None, val_dynamic_graph=None,
            loss='mse', epochs=1000, batch_size=32, lr=None,
            weight_decay=0, initialize=True, verbose=False, patience=10, **kwargs):
        if initialize:
            self.initialize()
        if lr is None:
            lr = self.lr

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_val = float('inf')
        best_weights = deepcopy(self.state_dict())
        patience_counter = patience

        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(len(train_input))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_input), batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = train_input[batch_idx].to(self.device)
                y_batch = train_target[batch_idx].to(self.device)

                total_loss = self._compute_loss(x_batch, y_batch)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += total_loss.item()
                n_batches += 1

            if val_input is not None and val_input.numel():
                self.eval()
                with torch.no_grad():
                    val_pred = self.forward(val_input.to(self.device))
                    val_loss = F.mse_loss(val_pred, val_target.to(self.device)).item()

                if val_loss < best_val:
                    best_val = val_loss
                    best_weights = deepcopy(self.state_dict())
                    self.output = val_pred
                    patience_counter = patience
                else:
                    patience_counter -= 1
                    if patience_counter <= 0:
                        break

        if val_input is not None:
            self.load_state_dict(best_weights)

    def predict(self, feature, **kwargs):
        self.eval()
        with torch.no_grad():
            feature = feature.to(self.device)
            return self.forward(feature)
