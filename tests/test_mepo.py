"""MepoGNN demo: a metapopulation GNN that predicts new cases from SIR states and a
time-varying mobility (OD) graph, trained through the low-level model.fit path.

Run it from the repo root with:

    python tests/test_mepo.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # load_toy_dataset() resolves './datasets' from the cwd

import torch

from epilearn.models.SpatialTemporal.MepoGNN import MepoGNN
from epilearn.data import Dataset
from epilearn.utils import metrics, utils


# ### Configs

device = torch.device('cpu')
torch.manual_seed(7)

lookback = 13   # input size
horizon = 5     # prediction size

epochs = 15     # kept small so the demo finishes in a few seconds
batch_size = 32


# ### Load the toy dataset

dataset = Dataset()                 # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
print(dataset)

features, _, _ = utils.normalize(dataset.x.clone())
# The toy graphs are raw OD matrices (values up to 5.4e6), so normalize them before
# feeding them to a graph model. normalize_adj() writes through to a stacked
# (T, N, N, 1) tensor it is handed, hence the clone().
adj_norm = utils.normalize_adj(dataset.graph.clone()).to(device)
dyn_adj_norm = utils.normalize_adj(dataset.dynamic_graph.clone()).to(device)

features = features.to(device)
# MepoGNN's SIR cell works in real population units, so the states and the target
# stay unnormalized: it predicts new cases per region per horizon step.
states = dataset.states.to(device)
target = dataset.y.to(device)


# ### Build the sliding windows
# 0.1.0: generate_dataset returns a DICT (features/targets/states/dynamic_graph/graph),
# and it does not fall back to dataset.graph / .states -- pass them explicitly.

def make_split(start, end):
    split = dataset.generate_dataset(X=features[start:end],
                                     Y=target[start:end],
                                     states=states[start:end],
                                     dynamic_adj=dyn_adj_norm[start:end],
                                     adj=adj_norm,
                                     lookback_window_size=lookback,
                                     horizon_size=horizon)
    # generate_dataset squeezes the trailing channel of the dynamic graph, but
    # MepoGNN's OD tensor is (samples, lookback, nodes, nodes, 1).
    split['dynamic_graph'] = split['dynamic_graph'].unsqueeze(-1)
    return split


train_split = make_split(0, 400)
val_split = make_split(400, 470)
test_split = make_split(470, 539)
print({k: tuple(v.shape) for k, v in train_split.items() if torch.is_tensor(v)})


# ### Train
# glm_type='Dynamic' learns how to aggregate the observed OD graph over the lookback
# window; glm_type='Adaptive' ignores it and learns a mobility matrix from scratch,
# seeded with adapt_graph.

model = MepoGNN(num_nodes=adj_norm.shape[0],
                num_features=train_split['features'].shape[3],
                num_timesteps_input=lookback,
                num_timesteps_output=horizon,
                glm_type='Dynamic',
                adapt_graph=adj_norm,
                blocks=2,
                layers=3,
                device=device).to(device)

model.fit(train_input=train_split['features'],
          train_target=train_split['targets'],
          train_states=train_split['states'],
          train_graph=adj_norm,
          train_dynamic_graph=train_split['dynamic_graph'],
          val_input=val_split['features'],
          val_target=val_split['targets'],
          val_states=val_split['states'],
          val_graph=adj_norm,
          val_dynamic_graph=val_split['dynamic_graph'],
          loss='mse',
          verbose=False,
          batch_size=batch_size,
          epochs=epochs,
          lr=1e-3)   # the SIR cell starts far from the data scale; 1e-2 diverges


# ### Evaluate

out = model.predict(feature=test_split['features'],
                    graph=adj_norm,
                    states=test_split['states'],
                    dynamic_graph=test_split['dynamic_graph'])

targets = test_split['targets'].cpu()
print(f"\npredictions {tuple(out.shape)} (samples, nodes, horizon)")
print(f"mean daily cases in the test window : {targets.mean():.2f}")
print(f"MepoGNN MAE                         : {metrics.get_MAE(out, targets).item():.2f}")
print(f"MepoGNN RMSE                        : {metrics.get_RMSE(out, targets).item():.2f}")
