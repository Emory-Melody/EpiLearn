#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import os
import matplotlib.pyplot as plt
import sys
import os

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move to the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Change the working directory (optional, if needed)
os.chdir(parent_dir)

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.GAT import GAT
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.models.Spatial.GIN import GIN



from epilearn.data import UniversalDataset
from epilearn.utils import utils, metrics
from epilearn.utils import transforms

# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 12 # inputs size
horizon = 3 # predicts size

# permutation is True when using STGCN
permute = True

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

model = GCN(num_features=train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=2, with_bn=True,
        dropout=0.3, device=device)

'''model = GAT(num_features=train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=2, with_bn=True, nheads=[2,4], concat=True,
        dropout=0.3, device=device)'''

'''model = SAGE(num_features=train_input.shape[2]*train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=1, with_bn=True, aggr=torch_geometric.nn.GRUAggregation,
        dropout=0.3, device=device)'''

'''model = GIN(num_features=train_input.shape[2]*train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=2, 
        dropout=0.3, device=device)'''


model = model.to(device)


# In[3]:


# training
model.fit(
        train_input=train_input[..., 0,:], 
        train_target=train_target, 
        train_states=None, 
        train_graph=adj_norm, 
        train_dynamic_graph=None,
        val_input=val_input[..., 0,:], 
        val_target=val_target,
        val_states=None, 
        val_graph=adj_norm, 
        val_dynamic_graph=None,
        loss='mse', 
        epochs=5, 
        batch_size=10,
        lr=1e-3, 
        weight_decay=1e-3,
        initialize=True, 
        verbose=True, 
        patience=10, 
        shuffle=False,
        )


# In[4]:


# evaluate
out = model.predict(feature=test_input[..., 0,:], 
                    graph=adj_norm, 
                    states=None, 
                    dynamic_graph=None, 
                    batch_size=1, 
                    device = device, 
                    shuffle=False)

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

