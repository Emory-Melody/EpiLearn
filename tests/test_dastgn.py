"""DASTGN demo: a dynamic/adaptive space-time graph network trained through the
low-level model.fit path, with the sliding windows built by hand.

Run it from the repo root with:

    python tests/test_dastgn.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # load_toy_dataset() resolves './datasets' from the cwd

import torch

from epilearn.models.SpatialTemporal.DASTGN import DASTGN
from epilearn.data import Dataset
from epilearn.utils import metrics, utils


# ### Configs

device = torch.device('cpu')
torch.manual_seed(7)

lookback = 8    # input size
horizon = 3     # prediction size

epochs = 6      # kept small: see the note on cost below
# DASTGN loops over the batch in Python, so a small batch costs the same per epoch as
# a large one but takes many more optimizer steps -- which this model needs.
batch_size = 4

# DASTGN builds a (nodes * lookback) x (nodes * lookback) space-time weight matrix
# for every sample, one Python loop iteration at a time, so its cost grows quickly
# with the number of regions. This demo runs on a 12-region slice of the toy
# dataset to stay fast; use range(47) to take all of them.
regions = list(range(12))


# ### Load the toy dataset

dataset = Dataset()                 # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
print(dataset)

features, _, _ = utils.normalize(dataset.x.clone())
# The toy graph is a raw OD matrix (values up to 5.4e6): normalize before use.
adj_norm = utils.normalize_adj(dataset.graph.clone())

features = features[:, regions, :].to(device)
adj_norm = adj_norm[regions][:, regions].to(device)
# DASTGN's regression head ends in a ReLU, so it can only fit a non-negative
# target: predict log1p(cases) rather than z-scored cases.
target = torch.log1p(dataset.y[:, regions]).to(device)


# ### Build the sliding windows
# 0.1.0: generate_dataset returns a DICT (features/targets/states/dynamic_graph/graph),
# it needs adj= passed explicitly, and its `permute` flag flipped meaning -- the new
# default permute=False is the old permute=True.

def make_split(start, end):
    split = dataset.generate_dataset(X=features[start:end],
                                     Y=target[start:end],
                                     adj=adj_norm,
                                     lookback_window_size=lookback,
                                     horizon_size=horizon)
    # DASTGN returns (samples, horizon, nodes); generate_dataset's default layout
    # puts horizon last, so transpose the targets to match the model.
    split['targets'] = split['targets'].transpose(1, 2)
    return split


train_split = make_split(0, 180)
val_split = make_split(180, 250)
test_split = make_split(250, 330)
print({k: tuple(v.shape) for k, v in train_split.items() if torch.is_tensor(v)})


# ### Train
# DASTGN reads the static graph only (its space-time weights are learned), so there
# is no dynamic_graph / states to pass.

model = DASTGN(num_nodes=adj_norm.shape[0],
               num_features=train_split['features'].shape[3],
               num_timesteps_input=lookback,
               num_timesteps_output=horizon,
               device=device).to(device)

model.fit(train_input=train_split['features'],
          train_target=train_split['targets'],
          train_states=None,
          train_graph=adj_norm,
          val_input=val_split['features'],
          val_target=val_split['targets'],
          val_states=None,
          val_graph=adj_norm,
          loss='mse',
          verbose=False,
          batch_size=batch_size,
          epochs=epochs,
          lr=1e-3)   # 1e-2 drives the output ReLU dead on this data


# ### Evaluate
# Metrics are in log1p units; expm1 puts them back into case counts.

out = model.predict(feature=test_split['features'], graph=adj_norm)
targets = test_split['targets'].cpu()

print(f"\npredictions {tuple(out.shape)} (samples, horizon, nodes)")
print(f"mean log1p(cases) in the test window : {targets.mean():.3f}")
print(f"MAE, log1p units                     : {metrics.get_MAE(out, targets).item():.3f}")
print(f"MAE, cases                           : "
      f"{metrics.get_MAE(torch.expm1(out), torch.expm1(targets)).item():.2f}")
