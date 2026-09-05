"""EpiDeep: Deep learning with clustering for epidemic forecasting.

Adhikari et al. (KDD 2019), adapted from the CALI-Net implementation
(Kamarthi et al. 2022, https://github.com/AdityaLab/CALI-Net).

Architecture:
- Dual autoencoder with KMeans clustering for regime-aware representations
- LSTM with attention for temporal encoding
- Mapper network bridges historical patterns to current context
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from sklearn.cluster import KMeans

from .base import BaseModel


def _build_network(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "leakyrelu":
            net.append(nn.LeakyReLU())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


def _target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, (hidden, _) = self.lstm(x, (h0, c0))
        hidden_state = hidden[-1].unsqueeze(2)
        weights = torch.bmm(out, hidden_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        attended = torch.bmm(out.transpose(1, 2), weights).squeeze(2)
        return self.dropout(attended)


class EpiDeepModel(BaseModel):
    def __init__(self, device='cpu', lookback=16, horizon=4, n_features=1,
                 num_features=None, num_timesteps_input=None, num_timesteps_output=None,
                 hidden_dim=20, n_clusters=4, seq_len=5,
                 encode_layers=None, mapping_layers=None,
                 pretrain_epochs=200, alpha_cluster=0.1, alpha_pred=10.0,
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
        self.seq_len = min(seq_len, lookback)
        self.pretrain_epochs = pretrain_epochs
        self.alpha_cluster = alpha_cluster
        self.alpha_pred = alpha_pred
        self.lr = lr

        if encode_layers is None:
            encode_layers = [256, 128]
        if mapping_layers is None:
            mapping_layers = [64, 128, 64]

        query_dim = self.seq_len * n_features
        full_dim = lookback * n_features

        self.first_encoder = _build_network([query_dim] + encode_layers + [hidden_dim])
        self.first_decoder = _build_network([hidden_dim] + list(reversed(encode_layers)) + [query_dim])
        self.first_cluster_layer = nn.Parameter(torch.randn(n_clusters, hidden_dim))

        self.second_encoder = _build_network([full_dim] + encode_layers + [hidden_dim])
        self.second_decoder = _build_network([hidden_dim] + list(reversed(encode_layers)) + [full_dim])
        self.second_cluster_layer = nn.Parameter(torch.randn(n_clusters, hidden_dim))

        self.mapper = _build_network([hidden_dim] + mapping_layers + [hidden_dim], activation="leakyrelu")

        self.rnn_encoder = AttentionLSTM(n_features, hidden_dim, num_layers=2)

        decoder_in = hidden_dim + hidden_dim
        self.decoder = _build_network([decoder_in, hidden_dim, hidden_dim], activation="leakyrelu")
        self.regressor = _build_network([hidden_dim, hidden_dim, hidden_dim, horizon], activation="leakyrelu")

        self.alpha = 1.0
        self.to(device)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.first_cluster_layer.data)
        nn.init.xavier_normal_(self.second_cluster_layer.data)

    def _cluster_soft_assign(self, z, cluster_layer):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        query = x[:, -self.seq_len:, :].reshape(batch_size, -1)
        full = x.reshape(batch_size, -1)

        z1 = self.first_encoder(query)
        translated_emb = self.mapper(z1)

        rnn_input = x[:, -self.seq_len:, :]
        rnn_out = self.rnn_encoder(rnn_input)

        combined = torch.cat([rnn_out, translated_emb], dim=1)
        decoded = self.decoder(combined)
        pred = self.regressor(decoded)
        return pred

    def _compute_loss(self, x, pred, target):
        batch_size = x.size(0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        query = x[:, -self.seq_len:, :].reshape(batch_size, -1)
        full = x.reshape(batch_size, -1)

        z1 = self.first_encoder(query)
        x1_bar = self.first_decoder(z1)
        q1 = self._cluster_soft_assign(z1, self.first_cluster_layer)

        z2 = self.second_encoder(full)
        x2_bar = self.second_decoder(z2)
        q2 = self._cluster_soft_assign(z2, self.second_cluster_layer)

        translated_emb = self.mapper(z1)

        recon_loss = F.mse_loss(x1_bar, query) + F.mse_loss(x2_bar, full)
        mapping_loss = F.mse_loss(translated_emb, z2.detach())

        p1 = _target_distribution(q1).detach()
        p2 = _target_distribution(q2).detach()
        cluster_loss = F.kl_div(q1.log(), p1, reduction='batchmean') + \
                       F.kl_div(q2.log(), p2, reduction='batchmean')

        pred_loss = F.mse_loss(pred, target)

        total = self.alpha_cluster * (recon_loss + mapping_loss + cluster_loss) + \
                self.alpha_pred * pred_loss
        return total

    def fit(self, train_input, train_target, train_states=None, train_graph=None,
            train_dynamic_graph=None, val_input=None, val_target=None,
            val_states=None, val_graph=None, val_dynamic_graph=None,
            loss='mse', epochs=1000, batch_size=32, lr=None,
            weight_decay=0, initialize=True, verbose=False, patience=10, **kwargs):
        if initialize:
            self.initialize()
        if lr is None:
            lr = self.lr

        self._pretrain_autoencoders(train_input)

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

                pred = self.forward(x_batch)
                loss_val = self._compute_loss(x_batch, pred, y_batch)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                epoch_loss += loss_val.item()
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

    def _pretrain_autoencoders(self, train_input):
        if train_input.dim() == 2:
            train_input = train_input.unsqueeze(-1)

        batch_size = train_input.size(0)
        query_data = train_input[:, -self.seq_len:, :].reshape(batch_size, -1).to(self.device)
        full_data = train_input.reshape(batch_size, -1).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.first_encoder.parameters()) +
            list(self.first_decoder.parameters()) +
            list(self.second_encoder.parameters()) +
            list(self.second_decoder.parameters()),
            lr=self.lr
        )

        for _ in range(self.pretrain_epochs):
            z1 = self.first_encoder(query_data)
            x1_bar = self.first_decoder(z1)
            z2 = self.second_encoder(full_data)
            x2_bar = self.second_decoder(z2)

            loss = F.mse_loss(x1_bar, query_data) + F.mse_loss(x2_bar, full_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            z1 = self.first_encoder(query_data).cpu().numpy()
            z2 = self.second_encoder(full_data).cpu().numpy()

        kmeans1 = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans1.fit(z1)
        self.first_cluster_layer.data = torch.tensor(
            kmeans1.cluster_centers_, dtype=torch.float32, device=self.device
        )

        kmeans2 = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans2.fit(z2)
        self.second_cluster_layer.data = torch.tensor(
            kmeans2.cluster_centers_, dtype=torch.float32, device=self.device
        )

    def predict(self, feature, **kwargs):
        self.eval()
        with torch.no_grad():
            feature = feature.to(self.device)
            return self.forward(feature)
