#!/usr/bin/env python
# coding: utf-8
"""
Source-detection pipeline (EpiLearn 0.1.0).

Detection is a per-node classification task, so it uses the low-level
``train_model`` entry point with explicit train/val/test splits built by
``Dataset.generate_dataset`` rather than the rolling-origin ``rolling_train``.

Run it with::

    python examples/detection_task.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
# load_toy_dataset() resolves './datasets' relative to the working directory.
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.NetworkSIR import NetSIR

from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.models.Spatial.GAT import GAT
from epilearn.models.Spatial.GIN import GIN

from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.utils import simulation
from epilearn.tasks.detection import Detection


# ### Configs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(7)

lookback = 1    # input size
horizon = 2     # number of classes

epochs = 20
batch_size = 16


# ### Initialize dataset
# The toy dataset is a regression dataset, so binarize the target into two
# per-node classes to turn it into a detection problem.

dataset = Dataset()              # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
dataset.y = (dataset.y > dataset.y.median()).long()
print(dataset)


# ### Add transformations
# apply_now=True applies them immediately, because here we build the splits by
# hand instead of letting rolling_train refit the stats per fold.

# transformation = transforms.Compose({
#     'features': [transforms.normalize_feat()],
#     'graph': [transforms.normalize_adj()],
# })
transformation = transforms.Compose({
    'features': [],
    'graph': [],
})
dataset.set_transforms(transformation, apply_now=True)


# ### Build the splits
# generate_dataset returns a DICT (features / targets / states / graph /
# dynamic_graph). Always pass adj=..., otherwise the graph never reaches the model.

def make_split(start, end):
    return dataset.generate_dataset(X=dataset.x[start:end],
                                    Y=dataset.y[start:end],
                                    adj=dataset.graph,
                                    lookback_window_size=lookback,
                                    horizon_size=1)


train_split = make_split(0, 300)
val_split = make_split(300, 400)
test_split = make_split(400, 539)
print({k: tuple(v.shape) for k, v in train_split.items() if v is not None})


# ### Initialize model and task

task = Detection(prototype=GCN,
                 dataset=dataset,
                 lookback=lookback,
                 horizon=horizon,
                 device=device)


# ### Train model

result = task.train_model(train_split=train_split,
                          val_split=val_split,
                          test_split=test_split,
                          train_loss='ce',
                          val_loss='ce',
                          epochs=epochs,
                          batch_size=batch_size)
print(f"test ce loss: {result['loss']:.4f}")


# ### Evaluate model
# evaluate_model now takes a split dict, not a Dataset.

evaluation = task.evaluate_model(dataset=test_split)
print(f"accuracy: {evaluation['accuracy']:.4f}, macro f1: {evaluation['macro_f1']:.4f}")

# plot_preds indexes targets as (samples, nodes), while generate_dataset returns
# (samples, nodes, 1) for a one-step horizon, so drop the trailing axis first.
evaluation['targets'] = evaluation['targets'].squeeze(-1)
task.plot_preds(evaluation, sample_idx=0)


# ### Train on a simulated dataset
# Simulate an outbreak on a random graph, then ask the model to recover which
# node was the seed from the final state of the network.

num_nodes = 25
initial_graph = simulation.get_random_graph(num_nodes=num_nodes, connect_prob=0.15)

x, y = [], []
for _ in range(60):
    # one seed per sample: [S, I, R] one-hot states
    initial_states = torch.zeros(num_nodes, 3)
    initial_states[:, 0] = 1
    source = torch.randint(0, num_nodes, (1,)).item()
    initial_states[source, 0] = 0
    initial_states[source, 1] = 1

    sim = NetSIR(num_nodes=num_nodes,
                 horizon=60,
                 infection_rate=0.01,
                 recovery_rate=0.0384)
    preds = sim(initial_states, initial_graph, steps=None)

    # input: final network state; label: which node was the seed
    x.append(torch.nn.functional.one_hot(preds[-1].argmax(1), num_classes=3))
    y.append(initial_states.argmax(1))

sim_dataset = Dataset(x=torch.stack(x).float(), graph=initial_graph)
sim_dataset.y = torch.stack(y).long()   # keep integer class labels
sim_dataset.set_transforms(transformation, apply_now=True)


def make_sim_split(start, end):
    return sim_dataset.generate_dataset(X=sim_dataset.x[start:end],
                                        Y=sim_dataset.y[start:end],
                                        adj=sim_dataset.graph,
                                        lookback_window_size=lookback,
                                        horizon_size=1)


task = Detection(prototype=GCN,
                 dataset=sim_dataset,
                 lookback=lookback,
                 horizon=horizon,
                 device=device)

result = task.train_model(train_split=make_sim_split(0, 36),
                          val_split=make_sim_split(36, 48),
                          test_split=make_sim_split(48, 60),
                          train_loss='ce',
                          val_loss='ce',
                          epochs=epochs,
                          batch_size=8)
evaluation = task.evaluate_model(dataset=make_sim_split(48, 60),
                                 compute_bootstrap_ci=False)
print(f"simulated accuracy: {evaluation['accuracy']:.4f}")
