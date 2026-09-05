#!/usr/bin/env python
# coding: utf-8
"""
Loading data (EpiLearn 0.1.0).

``UniversalDataset`` is now ``Dataset`` (the old name is a deprecated alias). A few
attributes were renamed at the same time:

    dataset.features  ->  dataset.feature_names
    dataset.timestamp ->  dataset.timestamps
    dataset.index / .anual_population / .coordinates  ->  dataset.metadata
    dataset.save()    ->  dataset.save(path)          (path is now required)

Run it with::

    python examples/data_loading.py
"""

import os
import sys
import tempfile

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
# load_toy_dataset() resolves './datasets' relative to the working directory.
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch

from epilearn.data import Dataset


# ## 1. Built-in datasets, by name
# The name picks the loader; root is where the download is cached.

covid = Dataset(name='Covid_Austria', root='./datasets/')
print("Covid_Austria      :", covid)
print("  feature_names    :", covid.feature_names)     # was dataset.features
print("  timestamps       :", covid.timestamps[:3], "...", len(covid.timestamps), "total")

measles = Dataset(name='Measles', root='./datasets')
print("Measles            :", measles)
print("  feature_names    :", measles.feature_names[:4], "...")
# index / anual_population / coordinates now live in one metadata dict
print("  metadata keys    :", sorted(measles.metadata.keys()))

jhu = Dataset(name='JHU_covid', root='./datasets')
print("JHU_covid          :", jhu)

# Tycho_v1 bundles several diseases whose series have different lengths, so load
# the one you want yourself instead of asking Dataset to stack them all. The
# name= call still downloads the archive before it fails on the ragged stack.
tycho_path = './datasets/Tycho_v1.pt'
if not os.path.exists(tycho_path):
    try:
        Dataset(name='Tycho_v1', root='./datasets')
    except ValueError:
        pass  # expected: the diseases have different lengths
tycho_raw = torch.load(tycho_path, weights_only=False)
print("Tycho_v1 diseases  :", list(tycho_raw.keys()))
measles_series = tycho_raw['MEASLES'].float().reshape(-1, 1)
tycho = Dataset(x=measles_series, y=measles_series, feature_names=['MEASLES'])
print("Tycho_v1/MEASLES   :", tycho)


# ## 2. The toy dataset

dataset = Dataset()
dataset.load_toy_dataset()
print("\ntoy dataset        :", dataset)
print(f"  x              : {tuple(dataset.x.shape)} (timesteps, nodes, channels)")
print(f"  y              : {tuple(dataset.y.shape)} (timesteps, nodes)")
print(f"  states         : {tuple(dataset.states.shape)} (timesteps, nodes, SIR)")
print(f"  graph          : {tuple(dataset.graph.shape)}")
print(f"  dynamic_graph  : {tuple(dataset.dynamic_graph.shape)}")
print(f"  edge_index     : {tuple(dataset.edge_index.shape)}")
print(f"  edge_weight    : {tuple(dataset.edge_weight.shape)}")
print(f"  n_timesteps={dataset.n_timesteps}, n_regions={dataset.n_regions}, "
      f"n_features={dataset.n_features}, spatiotemporal={dataset.is_spatiotemporal}")


# ## 3. Building a Dataset from tensors you already have

custom = Dataset(x=dataset.x,
                 y=dataset.y,
                 graph=dataset.graph,
                 dynamic_graph=dataset.dynamic_graph.squeeze(-1),   # (T, N, N)
                 states=dataset.states,
                 feature_names=['cases', 'f1', 'f2', 'f3'],
                 target_names=['cases'])
print("\nfrom tensors       :", custom)


# ## 4. Saving and reloading
# Dataset.save() now requires an explicit path.

tmp_dir = tempfile.mkdtemp(prefix='epilearn_example_')
path = os.path.join(tmp_dir, 'toy_dataset.pt')
custom.save(path)
reloaded = Dataset.load(path)
print("reloaded           :", reloaded)
print("  feature_names    :", reloaded.feature_names)
print("  identical x      :", torch.equal(custom.x, reloaded.x))
os.remove(path)
os.rmdir(tmp_dir)


# ## 5. Slicing by time and by region
# get_slice / rolling_splits replace the removed ganerate_splits / get_splits.

train = dataset.get_slice(end_rate=0.7)
test = dataset.get_slice(start_rate=0.7)
print(f"\nget_slice          : train {train.n_timesteps} timesteps, test {test.n_timesteps}")

subset = dataset.get_slice(start=0, end=99, end_inclusive=True, regions=[0, 1, 2])
print("get_slice(regions) :", subset)

for fold, (tr, va, te) in enumerate(dataset.rolling_splits(train_size=400,
                                                           val_size=50,
                                                           test_size=50), start=1):
    print(f"rolling fold {fold}   : train={tr.n_timesteps} val={va.n_timesteps} test={te.n_timesteps}")
