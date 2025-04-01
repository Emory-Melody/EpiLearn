#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import matplotlib.pyplot as plt
os.chdir("..")

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.ATMGNN import MPNN_LSTM, ATMGNN
from epilearn.models.SpatialTemporal.STAN import STAN
from epilearn.models.SpatialTemporal.DCRNN import DCRNN
from epilearn.models.SpatialTemporal.GraphWaveNet import GraphWaveNet


from epilearn.data import UniversalDataset
from epilearn.utils import utils, metrics
from epilearn.utils import transforms

# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 12 # inputs size
horizon = 3 # predicts size

# permutation is True when using STGCN
permute = False

epochs = 50 # training epochs
batch_size = 50 # training batch size


# In[2]:


# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()

# initialize transforms
transformation = transforms.Compose({
                                    'features': [
                                                    transforms.normalize_feat(),

                                                ],
                                    "target": [transforms.normalize_feat()],
                                    'graph': [
                                                transforms.normalize_adj(),
                                                    
                                            ],
                                    'dynamic_graph': [
                                                        transforms.normalize_adj(),
                                                    
                                                    ],
                                    'states': []
                                    })


# preprocessing dataset
dataset.transforms = transformation

features, target, adj_norm, adj_dynamic_norm, states = dataset.get_transformed().values()
mean, std = dataset.transforms.feat_mean, dataset.transforms.feat_std

features = features.to(device)
adj_norm = adj_norm.to(device)
adj_dynamic_norm = adj_dynamic_norm.to(device)

# split data
train_rate = 0.6 
val_rate = 0.2

target_feat_idx = None
target_idx = None

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))

train_original_input = features[:split_line1, :, :]
val_original_input = features[split_line1:split_line2, :, :]
test_original_input = features[split_line2:, :, :]

train_original_target = target[:split_line1, :]
val_original_target = target[split_line1:split_line2, :]
test_original_target = target[split_line2:, :]

train_original_states = dataset.states[:split_line1, :, :]
val_original_states = dataset.states[split_line1:split_line2, :, :]
test_original_states = dataset.states[split_line2:, :, :]


train_input, train_target, train_states, train_adj = dataset.generate_dataset(X = train_original_input, Y = train_original_target, states = train_original_states, dynamic_adj = adj_dynamic_norm, lookback_window_size = lookback, horizon_size = horizon, permute = permute)
val_input, val_target, val_states, val_adj = dataset.generate_dataset(X = val_original_input, Y = val_original_target, states = val_original_states, dynamic_adj = adj_dynamic_norm, lookback_window_size = lookback, horizon_size = horizon, permute = permute)
test_input, test_target, test_states, test_adj = dataset.generate_dataset(X = test_original_input, Y = test_original_target, states = test_original_states, dynamic_adj = adj_dynamic_norm, lookback_window_size = lookback, horizon_size = horizon, permute = permute)


# prepare model

# model = STGCN(
#             num_nodes=adj_norm.shape[0],
#             num_features=train_input.shape[3],
#             num_timesteps_input=lookback,
#             num_timesteps_output=horizon
#             ).to(device=device)

model = MPNN_LSTM(
                num_nodes=adj_norm.shape[0],
                num_features=train_input.shape[3],
                num_timesteps_input=lookback,
                num_timesteps_output=horizon,
                nhid=4
                ).to(device=device)

# model = ATMGNN(
#                 num_nodes=adj_norm.shape[0],
#                 num_features=train_input.shape[3],
#                 num_timesteps_input=lookback,
#                 num_timesteps_output=horizon,
#                 nhid=4
#                 ).to(device=device)

# model = STAN(
#             num_nodes=adj_norm.shape[0],
#             num_features=train_input.shape[3],
#             num_timesteps_input=lookback,
#             num_timesteps_output=horizon
#             ).to(device=device)


'''model = DCRNN(num_features=train_input.shape[3],
              num_timesteps_input=lookback,
              num_timesteps_output=horizon,
              num_classes=1,
              max_diffusion_step=2, 
              filter_type="laplacian",
              num_rnn_layers=1, 
              rnn_units=1,
              nonlinearity="tanh",
              dropout=0.5,
              device=device)'''


'''model = GraphWaveNet(num_timesteps_input=train_input.shape[3], num_timesteps_output=horizon, 
                     adj_m=adj_norm, gcn_bool=True, 
                     addaptadj=True, aptinit=None, 
                     blocks=4, nlayers=2,
                     residual_channels=32, dilation_channels=32, 
                     skip_channels=32 * 8, end_channels=32 * 16,dropout=0.3, device=device)'''


# In[3]:


# training
model.fit(
        train_input=train_input, 
        train_target=train_target,
        train_states = train_states, 
        train_graph=adj_norm,  # for dynamic graph, use val_adj
        val_input=val_input, 
        val_target=val_target, 
        val_states=val_states,
        val_graph=adj_norm,  # for dynamic graph, use val_adj
        loss='mse',
        verbose=True,
        batch_size=batch_size,
        epochs=epochs)


# In[4]:


# evaluate
out = model.predict(feature=test_input, graph=adj_norm)
preds = out.detach().cpu()*std[0]+mean[0]
targets = test_target.detach().cpu()*std[0]+mean[0]
# MAE
mae = metrics.get_MAE(preds, targets)
print(f"MAE: {mae.item()}")


# In[5]:


# # visualization
# out = model.predict(feature=train_input, graph=adj_norm).detach().cpu()

# sample = 28

# plt.figure(figsize=(15 ,5))
# for i in range(1, 4):
#     sample_input=train_input[sample, i, :, 0]
#     sample_output=out[sample, i, :]
#     sample_target=train_target[sample, i, :]

#     vis_data = torch.cat([sample_input, sample_target]).numpy()
    
#     plt.subplot(1, 3, i)
#     rng = list(range(lookback+horizon))
#     plt.plot(rng, vis_data, label="ground truth")
#     plt.plot(rng[lookback:lookback+horizon], sample_output.numpy(), label="prediction")
#     plt.legend()


# plt.show()

