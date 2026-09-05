#!/usr/bin/env python
# coding: utf-8
"""
Getting your own data into EpiLearn (0.1.0).

Covers the four shapes of input EpiLearn understands:

  * spatiotemporal tensors (timesteps, nodes, channels) + a graph
  * a plain univariate series (timesteps, channels)
  * per-node labels, for the detection task
  * a long-format CSV, via ``Dataset.from_csv``

``Dataset.from_csv`` renamed its keywords in 0.1.0:
``feature_csv``->``file_path``, ``node_id_col``->``region_col``,
``time_col``->``timestamp_col``, ``edge_csv``->``graph_file``.

Run it with::

    python examples/dataset_customization.py
"""

import datetime
import os
import sys
import tempfile

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch
import numpy as np
import matplotlib.pyplot as plt

from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.forecast import Forecast
from epilearn.tasks.detection import Detection
from epilearn.models.SpatialTemporal import STGCN, ColaGNN
from epilearn.models.Temporal import LSTMModel, DlinearModel, GRUModel
from epilearn.models.Spatial.GCN import GCN


torch.manual_seed(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ## 1. What your data looks like

data = torch.load("examples/example.pt", weights_only=False)
print("keys:", list(data.keys()))
print(f"Node Features [timesteps, nodes, channels]: {tuple(data['features'].shape)}")
print(f"Static Graph  [nodes, nodes]              : {tuple(data['graph'].shape)}")
print(f"Dynamic Graph [timesteps, nodes, nodes]   : {tuple(data['dynamic_graph'].shape)}")
print(f"Target        [timesteps, nodes]          : {tuple(data['targets'].shape)}")
print(f"Node States   [timesteps, nodes]          : {tuple(data['states'].shape)}")

node_features = data['features']
static_graph = torch.Tensor(data['graph'])
dynamic_graph = data['dynamic_graph']
targets = data['targets']

# in this dataset the target series is also channel 0 of the features
print("target == channel 0:", bool((node_features[:, :, 0] == targets).all()))

plt.figure(figsize=(12, 3))
plt.plot(np.array(targets[:, 0]), label='node 0 target')
plt.plot(np.array(node_features[:, 0, 1]), label='node 0, channel 1')
plt.legend()
plt.tight_layout()
plt.show()


# ## 2. Spatiotemporal forecasting
# Minimum ingredients: node features, a target per node, and a graph.

lookback = 36
horizon = 3

# Optional extras: states=data['states'] for SIR-style node states, and
# dynamic_graph=dynamic_graph for models that consume a per-timestep graph.
dataset = Dataset(x=node_features,
                  y=targets,
                  graph=static_graph)
print("\nspatiotemporal dataset:", dataset)

dataset.set_transforms(transforms.Compose({
    "features": [transforms.normalize_feat()],
    "target": [transforms.normalize_target()],
    "graph": [transforms.normalize_adj()],
}))

task = Forecast(prototype=STGCN,
                dataset=None,
                lookback=lookback,
                horizon=horizon,
                device=device)

result = task.rolling_train(dataset=dataset,
                            train_size=350,
                            val_size=60,
                            test_size=60,
                            train_loss='mse',
                            epochs=5,
                            batch_size=16,
                            lr=1e-3,
                            max_folds=2)
print("aggregate:", {k: round(float(v), 4) for k, v in result['aggregate_metrics'].items()})

# Re-score any fold: evaluate_model takes the fold's split dict, and
# inverse_normalize=True reports the numbers in original units.
for fold in result['fold_results']:
    evaluation = task.evaluate_model(dataset=fold['test_split'],
                                     process_history=fold['process_history'],
                                     inverse_normalize=True)
    print(f"fold {fold['fold']}: predictions {tuple(evaluation['predictions'].shape)}, "
          f"targets {tuple(evaluation['targets'].shape)}")


# ## 3. Temporal forecasting
# A univariate series is (timesteps, channels); use the same tensor as x and y.

inputs = targets[:, 0].unsqueeze(-1)
print(f"\nunivariate series: {tuple(inputs.shape)} (timesteps, channels)")

dataset = Dataset(x=inputs, y=inputs)
dataset.set_transforms(transforms.Compose({
    "features": [transforms.normalize_feat()],
    "target": [transforms.normalize_target()],
}))

task = Forecast(prototype=DlinearModel,
                dataset=None,
                lookback=lookback,
                horizon=horizon,
                device='cpu')

# model_args passes the hyperparameters your model needs on top of the ones the
# task infers (num_features / num_timesteps_input / num_timesteps_output / device).
result = task.rolling_train(dataset=dataset,
                            train_size=350,
                            val_size=60,
                            test_size=60,
                            train_loss='mse',
                            epochs=40,
                            batch_size=8,
                            lr=1e-3,
                            model_args={'moving_avg_window': 25})
print("aggregate:", {k: round(float(v), 4) for k, v in result['aggregate_metrics'].items()})

last_fold = result['fold_results'][-1]
evaluation = task.evaluate_model(dataset=last_fold['test_split'],
                                 process_history=last_fold['process_history'],
                                 inverse_normalize=True)
# plot_forecasts was renamed to plot_preds in 0.1.0
task.plot_preds(evaluation, region_idx=0, horizon_idx=-1)


# ## 4. Per-node labels: the detection task
# Detection classifies every node, so it uses explicit splits from
# generate_dataset rather than rolling_train.

labels = (targets > targets.median()).long()        # (timesteps, nodes), 2 classes

detect_ds = Dataset(x=node_features, graph=static_graph)
detect_ds.y = labels                                 # assign after construction to keep it integer


def make_split(lookback_size, start, end):
    return detect_ds.generate_dataset(X=detect_ds.x[start:end],
                                      Y=detect_ds.y[start:end],
                                      adj=detect_ds.graph,
                                      lookback_window_size=lookback_size,
                                      horizon_size=1)


# 4a. spatiotemporal input: a whole window of node features
task = Detection(prototype=STGCN, dataset=detect_ds, lookback=16, horizon=2, device='cpu')
task.train_model(train_split=make_split(16, 0, 350),
                 val_split=make_split(16, 350, 420),
                 test_split=make_split(16, 420, 539),
                 train_loss='ce', val_loss='ce', epochs=5, batch_size=16)
evaluation = task.evaluate_model(dataset=make_split(16, 420, 539), compute_bootstrap_ci=False)
print(f"\nSTGCN detection accuracy: {evaluation['accuracy']:.4f}")

# 4b. spatial input: a single timestep per sample (lookback=1)
task = Detection(prototype=GCN, dataset=detect_ds, lookback=1, horizon=2, device='cpu')
task.train_model(train_split=make_split(1, 0, 350),
                 val_split=make_split(1, 350, 420),
                 test_split=make_split(1, 420, 539),
                 train_loss='ce', val_loss='ce', epochs=5, batch_size=16)
evaluation = task.evaluate_model(dataset=make_split(1, 420, 539), compute_bootstrap_ci=False)
print(f"GCN detection accuracy  : {evaluation['accuracy']:.4f}")


# ## 5. Loading from CSV
# Long format: one row per (timestamp, region). from_csv builds the tensors for you.

csv_dataset = Dataset.from_csv(file_path='./datasets/toy_features.csv',
                               timestamp_col='time',
                               region_col='node',
                               feature_cols=['f0', 'f1', 'f2', 'f3'],
                               target_cols=['y'],
                               graph_file='./datasets/toy_edges.csv')
print("\nfrom_csv (spatiotemporal):", csv_dataset)
print("  feature_names:", csv_dataset.feature_names, "target_names:", csv_dataset.target_names)
print("  timestamps   :", csv_dataset.timestamps[:5], "...")

# Without region_col you get a temporal dataset, one row per timestamp.
tmp_dir = tempfile.mkdtemp(prefix='epilearn_example_')
tmp_csv = os.path.join(tmp_dir, 'series.csv')
with open(tmp_csv, 'w') as handle:
    handle.write("date,cases,tests\n")
    for day in range(120):
        date = datetime.date(2021, 1, 1) + datetime.timedelta(days=day)
        handle.write(f"{date.isoformat()},{100 + day},{500 + 2 * day}\n")

temporal_csv = Dataset.from_csv(file_path=tmp_csv,
                                timestamp_col='date',
                                feature_cols=['cases', 'tests'],
                                target_cols=['cases'])
print("from_csv (temporal)      :", temporal_csv)
os.remove(tmp_csv)
os.rmdir(tmp_dir)
