import torch
import os
import matplotlib.pyplot as plt
from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.ATMGNN import MPNN_LSTM, ATMGNN
from epilearn.models.SpatialTemporal.CNNRNN_Res import CNNRNN_Res
from epilearn.models.SpatialTemporal.STAN import STAN

from epilearn.data import UniversalDataset
from epilearn.utils import utils

# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 13 # inputs size
horizon = 3 # predicts size

permute = False
target_feat_idx = None
target_idx = None

epochs = 15 # training epochs
batch_size = 15 # training batch size

# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()

# preprocessing
features = utils.diff(dataset.states)
features, mean, std = utils.normalize(features)
adj_norm = utils.normalize_adj(dataset.graph)

features = features.to(device)
adj_norm = adj_norm.to(device)

# prepare datasets
train_rate = 0.6 
val_rate = 0.2

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))

train_original_data = features[:split_line1, :, :]
val_original_data = features[split_line1:split_line2, :, :]
test_original_data = features[split_line2:, :, :]

train_original_states = dataset.states[:split_line1, :, :]
val_original_states = dataset.states[split_line1:split_line2, :, :]
test_original_states = dataset.states[split_line2:, :, :]

train_input, train_target, train_states = dataset.generate_dataset(X = train_original_data, Y = train_original_data[..., 1:], states = train_original_states, lookback_window_size = lookback, horizon_size = horizon, permute = permute, feat_idx = target_feat_idx, target_idx = target_idx)
val_input, val_target, val_states = dataset.generate_dataset(X = val_original_data, Y = val_original_data[..., 1:], states = val_original_states, lookback_window_size = lookback, horizon_size = horizon, permute = permute, feat_idx = target_feat_idx, target_idx = target_idx)
test_input, test_target, test_states = dataset.generate_dataset(X = test_original_data, Y = test_original_data[..., 1:], states = test_original_states, lookback_window_size = lookback, horizon_size = horizon, permute = permute, feat_idx = target_feat_idx, target_idx = target_idx)

# prepare model

# model = STGCN(
#             num_nodes = adj_norm.shape[0],
#             num_features = train_input.shape[3],
#             num_timesteps_input = lookback,
#             num_timesteps_output = horizon
#             ).to(device = device)

# model = MPNN_LSTM(
#                 num_nodes = adj_norm.shape[0],
#                 num_features = train_input.shape[3],
#                 num_timesteps_input = lookback,
#                 num_timesteps_output = horizon,
#                 nhid = 4
#                 ).to(device = device)

# model = ATMGNN(
#                 num_nodes = adj_norm.shape[0],
#                 num_features = train_input.shape[3],
#                 num_timesteps_input = lookback,
#                 num_timesteps_output = horizon,
#                 nhid = 4
#                 ).to(device = device)

# model = CNNRNN_Res(
#                 num_nodes = adj_norm.shape[0],
#                 num_timesteps_input = lookback,
#                 num_timesteps_output = horizon,
#                 nhid = 4
#                 ).to(device = device)


model = STAN(
        num_nodes=adj_norm.shape[0],
        num_features=train_input.shape[3],
        num_timesteps_input=lookback,
        num_timesteps_output=horizon,
        population=1e10,
        gat_dim1=32, 
        gat_dim2=32, 
        gru_dim=32, 
        num_heads=1,
        device=device
        ).to(device=device)


# training
model.fit(
        train_input=train_input, 
        train_target=train_target,
        train_states = train_states, 
        graph=adj_norm, 
        val_input=val_input, 
        val_target=val_target, 
        val_states=val_states,
        loss='stan',
        verbose=True,
        batch_size=batch_size,
        epochs=epochs)