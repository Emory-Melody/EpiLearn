#!/usr/bin/env python
# coding: utf-8
"""
Plugging your own model into EpiLearn (0.1.0).

A custom model needs two things:

  * a ``forward(self, feature, graph=None, states=None, dynamic_graph=None, **kwargs)``
  * an ``initialize()`` used to (re-)init the weights

Subclass the right base — ``models.Temporal.base.BaseModel`` for a model that only
sees a time series, ``models.SpatialTemporal.base.BaseModel`` for anything that
also takes a graph — and the base class provides ``fit`` / ``predict`` / ``evaluate``.

Shapes the tasks hand you (with ``generate_dataset``'s new default ``permute=False``):

  temporal        feature (samples, lookback, channels)         -> (samples, horizon)
  spatiotemporal  feature (samples, lookback, nodes, channels)  -> (samples, nodes, horizon)
  spatial         feature (samples, 1, nodes, channels)         -> (samples, nodes, classes)

Run it with::

    python examples/model_customization.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.forecast import Forecast
from epilearn.tasks.detection import Detection
from epilearn.models.Temporal.base import BaseModel as TemporalBaseModel
from epilearn.models.SpatialTemporal.base import BaseModel as SpatialTemporalBaseModel


torch.manual_seed(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lookback = 36
horizon = 3


# ## 1. A customized temporal model

class CustomizedTemporal(TemporalBaseModel):
    def __init__(self,
                 num_features,
                 num_timesteps_input,
                 num_timesteps_output,
                 hidden_size=16,
                 num_layers=2,
                 bidirectional=False,
                 device='cpu',
                 **kwargs):
        super(CustomizedTemporal, self).__init__(device=device)
        self.num_feats = num_features
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lookback = num_timesteps_input
        self.horizon = num_timesteps_output
        self.device = device

        self.lstm = nn.LSTM(input_size=self.num_feats, hidden_size=self.hidden,
                            num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden, self.horizon)

    def forward(self, feature, graph=None, states=None, dynamic_graph=None, **kwargs):
        # feature: (samples, lookback, channels)
        out, _ = self.lstm(feature)
        return self.fc(out[:, -1, :])            # (samples, horizon)

    def initialize(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


# sample data: a cosine wave
t = torch.linspace(0, 1, 500)
cos_wave = torch.cos(2 * torch.pi * 3 * t)

plt.figure(figsize=(10, 3))
plt.plot(cos_wave.numpy())
plt.title("input series")
plt.tight_layout()
plt.show()

inputs = cos_wave.reshape(-1, 1)                 # (timesteps, channels)
dataset = Dataset(x=inputs, y=inputs)
dataset.set_transforms(transforms.Compose({
    "features": [transforms.normalize_feat()],
    "target": [transforms.normalize_target()],
}))

task = Forecast(prototype=CustomizedTemporal,
                dataset=None,
                lookback=lookback,
                horizon=horizon,
                device=device)

# model_args carries the hyperparameters the task cannot infer; num_features /
# num_timesteps_input / num_timesteps_output / device are filled in for you.
model_args = {"hidden_size": 16, "num_layers": 2, "bidirectional": False}

result = task.rolling_train(dataset=dataset,
                            train_size=300,
                            val_size=60,
                            test_size=60,
                            train_loss='mse',
                            epochs=40,
                            batch_size=8,
                            lr=1e-3,
                            model_args=model_args)
print("temporal aggregate:", {k: round(float(v), 4) for k, v in result['aggregate_metrics'].items()})

# plot the last fold (plot_forecasts was renamed to plot_preds in 0.1.0)
last_fold = result['fold_results'][-1]
evaluation = task.evaluate_model(dataset=last_fold['test_split'],
                                 process_history=last_fold['process_history'],
                                 inverse_normalize=True)
task.plot_preds(evaluation, region_idx=0, horizon_idx=-1)


# ## 2. A customized spatial-temporal model
# Note the output shape: (samples, nodes, horizon), matching the targets that
# generate_dataset produces with its new default permute=False.

class CustomizedSpatialTemporal(SpatialTemporalBaseModel):
    def __init__(self,
                 num_nodes,
                 num_features,
                 num_timesteps_input,
                 num_timesteps_output,
                 hidden_size=16,
                 num_layers=2,
                 bidirectional=False,
                 device='cpu',
                 **kwargs):
        super(CustomizedSpatialTemporal, self).__init__(device=device)
        self.num_nodes = num_nodes
        self.num_feats = num_features
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lookback = num_timesteps_input
        self.horizon = num_timesteps_output
        self.device = device

        self.gcn = GCNConv(in_channels=self.num_feats, out_channels=self.hidden)
        self.lstm = nn.LSTM(input_size=self.hidden, hidden_size=self.hidden,
                            num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden, self.horizon)

    def forward(self, feature, graph, states=None, dynamic_graph=None, **kwargs):
        # feature: (samples, lookback, nodes, channels)
        edge_index, _ = dense_to_sparse(graph)
        x = self.gcn(feature.float(), edge_index=edge_index)             # (S, L, N, H)
        x = x.transpose(1, 2).reshape(-1, self.lookback, self.hidden)    # (S*N, L, H)
        out, _ = self.lstm(x)
        out = out[:, -1, :].reshape(-1, self.num_nodes, self.hidden)     # (S, N, H)
        return self.fc(out)                                             # (S, N, horizon)

    def initialize(self):
        pass


# sample data: one cosine per node
num_nodes = 25
frequencies = torch.randint(low=1, high=10, size=[num_nodes])
cos_waves = torch.cos(2 * torch.pi * frequencies.unsqueeze(1) * t)
node_inputs = cos_waves.unsqueeze(-1).transpose(0, 1)        # (timesteps, nodes, channels)
graph = torch.round(torch.rand([num_nodes, num_nodes]))

dataset = Dataset(x=node_inputs, y=node_inputs.squeeze(-1), graph=graph)
dataset.set_transforms(transforms.Compose({
    "features": [transforms.normalize_feat()],
    "target": [transforms.normalize_target()],
    "graph": [transforms.normalize_adj()],
}))

task = Forecast(prototype=CustomizedSpatialTemporal,
                dataset=None,
                lookback=lookback,
                horizon=horizon,
                device='cpu')

result = task.rolling_train(dataset=dataset,
                            train_size=300,
                            val_size=60,
                            test_size=60,
                            train_loss='mse',
                            epochs=10,
                            batch_size=16,
                            model_args={"hidden_size": 16, "num_layers": 2,
                                        "bidirectional": False})
print("spatiotemporal aggregate:",
      {k: round(float(v), 4) for k, v in result['aggregate_metrics'].items()})


# ## 3. A customized spatial model, used for detection
# With lookback=1 the task hands the model (samples, 1, nodes, channels), and
# the target is one class label per node. For a graph model without a time axis
# the task supplies num_nodes / num_features / num_classes / device, so those are
# the arguments to accept.

class CustomizedSpatial(SpatialTemporalBaseModel):
    def __init__(self,
                 num_nodes,
                 num_features,
                 num_classes=2,
                 hidden_size=16,
                 device='cpu',
                 **kwargs):
        super(CustomizedSpatial, self).__init__(device=device)
        self.num_nodes = num_nodes
        self.num_feats = num_features
        self.hidden = hidden_size
        self.num_classes = num_classes
        self.device = device

        self.gcn = GCNConv(in_channels=self.num_feats, out_channels=self.hidden)
        self.fc = nn.Linear(self.hidden, self.num_classes)

    def forward(self, feature, graph, states=None, dynamic_graph=None, **kwargs):
        # feature: (samples, 1, nodes, channels) -> (samples, nodes, channels)
        x = feature.float().transpose(1, 2).reshape(-1, self.num_nodes, self.num_feats)
        edge_index, _ = dense_to_sparse(graph)
        x = self.gcn(x, edge_index=edge_index)
        return self.fc(x)                                    # (samples, nodes, classes)

    def initialize(self):
        pass


num_classes = 2
node_features = torch.rand(200, num_nodes, 1)
node_labels = (node_features[:, :, 0] > 0.5).long()

detect_ds = Dataset(x=node_features, graph=torch.round(torch.rand(num_nodes, num_nodes)))
detect_ds.y = node_labels        # assign after construction to keep integer labels


def make_split(start, end):
    return detect_ds.generate_dataset(X=detect_ds.x[start:end],
                                      Y=detect_ds.y[start:end],
                                      adj=detect_ds.graph,
                                      lookback_window_size=1,
                                      horizon_size=1)


task = Detection(prototype=CustomizedSpatial,
                 dataset=detect_ds,
                 lookback=1,
                 horizon=num_classes,
                 device='cpu')

task.train_model(train_split=make_split(0, 120),
                 val_split=make_split(120, 160),
                 test_split=make_split(160, 200),
                 train_loss='ce',
                 val_loss='ce',
                 epochs=25,
                 batch_size=16,
                 model_args={"hidden_size": 16})
evaluation = task.evaluate_model(dataset=make_split(160, 200), compute_bootstrap_ci=False)
print(f"custom spatial detection accuracy: {evaluation['accuracy']:.4f}")
