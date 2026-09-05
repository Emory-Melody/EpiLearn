"""CALI-Net: Calibration-Aware Learning for epidemic forecasting.

Kamarthi et al. (AAAI 2022), adapted from
https://github.com/AdityaLab/CALI-Net.

Architecture:
- Pre-trained EpiDeep as source domain encoder
- Lightweight target GRU module (replaces CAEM for univariate setting)
- Calibration modules for domain adaptation via knowledge distillation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .base import BaseModel
from .EpiDeep import EpiDeepModel, _build_network


class CALINetModel(BaseModel):
    def __init__(self, device='cpu', lookback=16, horizon=4, n_features=1,
                 num_features=None, num_timesteps_input=None, num_timesteps_output=None,
                 hidden_dim=20, n_clusters=4, calib_dim=16, seq_len=5,
                 pretrain_epochs=200, alpha_kd=0.1, alpha_recon=0.1,
                 kd_warmup=10, finetune_source=True,
                 lr=0.001, **kwargs):
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
        self.n_clusters = n_clusters
        self.calib_dim = calib_dim
        self.seq_len = min(seq_len, lookback)
        self.pretrain_epochs = pretrain_epochs
        self.alpha_kd = alpha_kd
        self.alpha_recon = alpha_recon
        self.kd_warmup = kd_warmup
        self.finetune_source = finetune_source
        self.lr = lr

        self.epideep = EpiDeepModel(
            device=device, lookback=lookback, horizon=horizon,
            n_features=n_features, hidden_dim=hidden_dim,
            n_clusters=n_clusters, seq_len=seq_len,
            pretrain_epochs=pretrain_epochs, lr=lr,
        )

        epideep_emb_dim = hidden_dim + hidden_dim

        self.target_encoder = nn.GRU(
            input_size=n_features, hidden_size=hidden_dim * 2,
            num_layers=1, batch_first=True
        )

        target_emb_dim = hidden_dim * 2

        self.module_g = _build_network([epideep_emb_dim, calib_dim])
        self.module_h = _build_network([target_emb_dim, calib_dim])
        self.module_f1 = _build_network([calib_dim, calib_dim, calib_dim], activation="leakyrelu")
        self.module_f2 = nn.Linear(calib_dim, horizon)
        self.module_g_prime = _build_network([calib_dim, epideep_emb_dim])
        self.module_h_prime = _build_network([calib_dim, target_emb_dim])

        self.to(device)

    def initialize(self):
        self.epideep.initialize()
        for m in [self.module_g, self.module_h, self.module_f1,
                  self.module_f2, self.module_g_prime, self.module_h_prime,
                  self.target_encoder]:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_normal_(sub.weight)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    def _get_epideep_embedding(self, x):
        batch_size = x.size(0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        query = x[:, -self.seq_len:, :].reshape(batch_size, -1)
        z1 = self.epideep.first_encoder(query)
        translated_emb = self.epideep.mapper(z1)

        rnn_input = x[:, -self.seq_len:, :]
        rnn_out = self.epideep.rnn_encoder(rnn_input)

        return torch.cat([rnn_out, translated_emb], dim=1)

    def _get_target_embedding(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, hidden = self.target_encoder(x)
        return hidden.squeeze(0)

    def forward(self, x):
        target_emb = self._get_target_embedding(x)
        h_out = self.module_h(target_emb)
        f1_out = self.module_f1(h_out)
        pred = self.module_f2(f1_out)
        return pred

    def fit(self, train_input, train_target, train_states=None, train_graph=None,
            train_dynamic_graph=None, val_input=None, val_target=None,
            val_states=None, val_graph=None, val_dynamic_graph=None,
            loss='mse', epochs=1000, batch_size=32, lr=None,
            weight_decay=0, initialize=True, verbose=False, patience=10, **kwargs):
        if initialize:
            self.initialize()
        if lr is None:
            lr = self.lr

        self.epideep._pretrain_autoencoders(train_input)

        if not self.finetune_source:
            for param in self.epideep.first_encoder.parameters():
                param.requires_grad = False
            for param in self.epideep.second_encoder.parameters():
                param.requires_grad = False
            for param in self.epideep.mapper.parameters():
                param.requires_grad = False
            for param in self.epideep.rnn_encoder.parameters():
                param.requires_grad = False

        params = (
            list(self.target_encoder.parameters()) +
            list(self.module_g.parameters()) +
            list(self.module_h.parameters()) +
            list(self.module_f1.parameters()) +
            list(self.module_f2.parameters()) +
            list(self.module_g_prime.parameters()) +
            list(self.module_h_prime.parameters())
        )
        if self.finetune_source:
            params += list(self.epideep.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        best_val = float('inf')
        best_weights = deepcopy(self.state_dict())
        patience_counter = patience

        for epoch in range(epochs):
            self.train()
            if not self.finetune_source:
                self.epideep.eval()
            indices = torch.randperm(len(train_input))
            epoch_loss = 0.0
            n_batches = 0

            kd_weight = self.alpha_kd * min(1.0, epoch / max(self.kd_warmup, 1))

            for i in range(0, len(train_input), batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = train_input[batch_idx].to(self.device)
                y_batch = train_target[batch_idx].to(self.device)

                if self.finetune_source:
                    source_emb = self._get_epideep_embedding(x_batch)
                else:
                    with torch.no_grad():
                        source_emb = self._get_epideep_embedding(x_batch)

                target_emb = self._get_target_embedding(x_batch)

                g_out = self.module_g(source_emb)
                h_out = self.module_h(target_emb)

                f1_out = self.module_f1(h_out)
                pred = self.module_f2(f1_out)

                source_recon = self.module_g_prime(g_out)
                target_recon = self.module_h_prime(h_out)

                pred_loss = F.mse_loss(pred, y_batch)
                kd_loss = F.mse_loss(h_out, g_out.detach())
                recon_loss = F.mse_loss(source_recon, source_emb.detach()) + \
                             F.mse_loss(target_recon, target_emb.detach())

                total_loss = pred_loss + kd_weight * kd_loss + self.alpha_recon * recon_loss

                optimizer.zero_grad()
                total_loss.backward()
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
