#!/usr/bin/env python
# coding: utf-8
"""
Spatial models without a Task (EpiLearn 0.1.0).

Same manual pipeline as examples/temporal.py, but for the graph models in
``epilearn.models.Spatial``: they consume one timestep of node features at a time,
shaped (samples, nodes, channels), plus a static adjacency matrix.

0.1.0 notes:
  * ``Dataset.get_transformed()`` is gone -> ``set_transforms(..., apply_now=True)``
  * ``Compose.feat_mean`` / ``.feat_std`` are gone -> ``get_process_history()``
  * ``generate_dataset`` returns a DICT, and its ``permute`` flag flipped meaning:
    the new default ``permute=False`` is the old ``permute=True``.

Run it with::

    python examples/spatial.py
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

from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.GAT import GAT
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.models.Spatial.GIN import GIN

from epilearn.data import Dataset
from epilearn.utils import metrics, transforms


# ### Configs

device = torch.device('cpu')
torch.manual_seed(7)

lookback = 12   # input size
horizon = 3     # prediction size

epochs = 10
batch_size = 32


# ### Load and transform the dataset

dataset = Dataset()              # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
dataset.dynamic_graph = None     # spatial models only use the static graph

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
adj_norm = dataset.graph.to(device)


# ### Split the data by time

train_rate = 0.6
val_rate = 0.2

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))


def make_split(start, end):
    return dataset.generate_dataset(X=features[start:end],
                                    Y=target[start:end],
                                    adj=adj_norm,
                                    lookback_window_size=lookback,
                                    horizon_size=horizon)


train_split = make_split(0, split_line1)
val_split = make_split(split_line1, split_line2)
test_split = make_split(split_line2, features.shape[0])
print({k: tuple(v.shape) for k, v in train_split.items() if v is not None})

# features: (samples, lookback, nodes, channels) -> take the first timestep of
# each window, giving the (samples, nodes, channels) a spatial model expects.
# targets: (samples, nodes, horizon)
train_input, train_target = train_split['features'][:, 0], train_split['targets']
val_input, val_target = val_split['features'][:, 0], val_split['targets']
test_input, test_target = test_split['features'][:, 0], test_split['targets']
print(f"train_input {tuple(train_input.shape)} (samples, nodes, channels)")


# ### Prepare the model

model = GCN(num_features=train_input.shape[-1],
            hidden_dim=16,
            num_classes=horizon,
            nlayers=2, with_bn=True,
            dropout=0.3, device=device)

# model = GAT(num_features=train_input.shape[-1],
#             hidden_dim=16,
#             num_classes=horizon,
#             nlayers=2, with_bn=True, nheads=[2, 4], concat=True,
#             dropout=0.3, device=device)

# model = SAGE(num_features=train_input.shape[-1],
#              hidden_dim=16,
#              num_classes=horizon,
#              nlayers=1, with_bn=True, aggr="mean",
#              dropout=0.3, device=device)

# model = GIN(num_features=train_input.shape[-1],
#             hidden_dim=16,
#             num_classes=horizon,
#             nlayers=2,
#             dropout=0.3, device=device)

model = model.to(device)


# ### Train

model.fit(train_input=train_input,
          train_target=train_target,
          train_states=None,
          train_graph=adj_norm,
          train_dynamic_graph=None,
          val_input=val_input,
          val_target=val_target,
          val_states=None,
          val_graph=adj_norm,
          val_dynamic_graph=None,
          loss='mse',
          epochs=epochs,
          batch_size=batch_size,
          lr=1e-3,
          weight_decay=1e-3,
          initialize=True,
          verbose=True,
          patience=10,
          shuffle=False)


# ### Evaluate
# Metrics are NOT auto-denormalized in 0.1.0: undo normalize_target() by hand.

out = model.predict(feature=test_input,
                    graph=adj_norm,
                    states=None,
                    dynamic_graph=None,
                    batch_size=32,
                    device=device,
                    shuffle=False)

preds = out.detach().cpu() * target_std + target_mean
targets = test_target.detach().cpu() * target_std + target_mean
print(f"MAE: {metrics.get_MAE(preds, targets).item():.4f}")


# ### Visualize a few nodes at the last horizon step

node_ids = range(min(20, preds.shape[1]))
plt.figure(figsize=(12, 4))
plt.plot([preds[0, n, -1].item() for n in node_ids], 'r-s', label='Prediction')
plt.plot([targets[0, n, -1].item() for n in node_ids], 'b--o', label='Ground Truth')
plt.xlabel("Node")
plt.ylabel("Value")
plt.title(f"First test window, horizon step {horizon}")
plt.legend()
plt.tight_layout()
plt.show()
