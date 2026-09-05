#!/usr/bin/env python
# coding: utf-8
"""
Spatial-temporal models without a Task (EpiLearn 0.1.0).

Same manual pipeline as examples/temporal.py and examples/spatial.py, for the
models in ``epilearn.models.SpatialTemporal``. They take a whole window of node
features, shaped (samples, lookback, nodes, channels), plus an adjacency matrix,
and predict (samples, nodes, horizon).

0.1.0 notes:
  * ``Dataset.get_transformed()`` is gone -> ``set_transforms(..., apply_now=True)``
  * ``Compose.feat_mean`` / ``.feat_std`` are gone -> ``get_process_history()``
  * ``generate_dataset`` returns a DICT, and its ``permute`` flag flipped meaning:
    the new default ``permute=False`` is the old ``permute=True``.

Run it with::

    python examples/spatial_temporal.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
# load_toy_dataset() resolves './datasets' relative to the working directory.
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch
import matplotlib.pyplot as plt

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.ATMGNN import ATMGNN
from epilearn.models.SpatialTemporal.DCRNN import DCRNN
from epilearn.models.SpatialTemporal.GraphWaveNet import GraphWaveNet

from epilearn.data import Dataset
from epilearn.utils import metrics, transforms


# ### Configs

device = torch.device('cpu')
torch.manual_seed(7)

lookback = 12   # input size
horizon = 3     # prediction size

epochs = 20
batch_size = 32


# ### Load and transform the dataset

dataset = Dataset()              # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
# The toy dynamic graph ships as (T, N, N, 1); pass
# dynamic_adj=dataset.dynamic_graph.squeeze(-1) below if your model needs it.
dataset.dynamic_graph = None

transformation = transforms.Compose({
    'features': [transforms.normalize_feat()],
    'target': [transforms.normalize_target()],
    'graph': [transforms.normalize_adj()],
    'states': [],
})
dataset.set_transforms(transformation, apply_now=True)

stats = dataset.get_process_history()
target_mean = float(stats['target_mean'])
target_std = float(stats['target_std'])

features = dataset.x.to(device)
target = dataset.y.to(device)
states = dataset.states.to(device)
adj_norm = dataset.graph.to(device)


# ### Split the data by time

train_rate = 0.6
val_rate = 0.2

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))


def make_split(start, end):
    return dataset.generate_dataset(X=features[start:end],
                                    Y=target[start:end],
                                    states=states[start:end],
                                    adj=adj_norm,
                                    lookback_window_size=lookback,
                                    horizon_size=horizon)


train_split = make_split(0, split_line1)
val_split = make_split(split_line1, split_line2)
test_split = make_split(split_line2, features.shape[0])
print({k: tuple(v.shape) for k, v in train_split.items() if v is not None})

num_nodes = adj_norm.shape[0]
num_features = train_split['features'].shape[3]


# ### Prepare the model

model = STGCN(num_nodes=num_nodes,
              num_features=num_features,
              num_timesteps_input=lookback,
              num_timesteps_output=horizon).to(device)

# model = ATMGNN(num_nodes=num_nodes,
#                num_features=num_features,
#                num_timesteps_input=lookback,
#                num_timesteps_output=horizon,
#                nhid=4).to(device)

# model = DCRNN(num_features=num_features,
#               num_timesteps_input=lookback,
#               num_timesteps_output=horizon,
#               num_classes=1,
#               max_diffusion_step=2,
#               filter_type="laplacian",
#               num_rnn_layers=1,
#               rnn_units=1,
#               nonlinearity="tanh",
#               dropout=0.5,
#               device=device).to(device)

# model = GraphWaveNet(num_nodes=num_nodes,
#                      num_features=num_features,
#                      num_timesteps_input=lookback,
#                      num_timesteps_output=horizon,
#                      adj_m=adj_norm, gcn_bool=True,
#                      addaptadj=True, aptinit=None,
#                      blocks=2, nlayers=2,
#                      residual_channels=8, dilation_channels=8,
#                      skip_channels=32, end_channels=64,
#                      dropout=0.3, device=device).to(device)


# ### Train
# For a dynamic-graph model pass train_dynamic_graph=train_split['dynamic_graph'].

model.fit(train_input=train_split['features'],
          train_target=train_split['targets'],
          train_states=train_split['states'],
          train_graph=adj_norm,
          val_input=val_split['features'],
          val_target=val_split['targets'],
          val_states=val_split['states'],
          val_graph=adj_norm,
          loss='mse',
          verbose=True,
          batch_size=batch_size,
          epochs=epochs)


# ### Evaluate
# Metrics are NOT auto-denormalized in 0.1.0: undo normalize_target() by hand.

out = model.predict(feature=test_split['features'],
                    graph=adj_norm,
                    states=test_split['states'])
if isinstance(out, tuple):       # some models also return a physics term
    out = out[0]

preds = out.detach().cpu() * target_std + target_mean
targets = test_split['targets'].detach().cpu() * target_std + target_mean
print(f"MAE: {metrics.get_MAE(preds, targets).item():.4f}")


# ### Visualize one node across the test window

node = 0
plt.figure(figsize=(12, 4))
plt.plot(preds[:, node, -1].numpy(), 'r-', label='Prediction')
plt.plot(targets[:, node, -1].numpy(), 'b--', label='Ground Truth')
plt.xlabel("Test window index")
plt.ylabel("Value")
plt.title(f"Node {node}, horizon step {horizon}")
plt.legend()
plt.tight_layout()
plt.show()
