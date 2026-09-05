"""STAN demo: a spatio-temporal attention network whose 'stan' loss trains a data-driven
head and a physics (SIR) head against the same target -- the daily change in each
region's infected and recovered compartments.

Run it from the repo root with:

    python tests/test_stan.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # load_toy_dataset() resolves './datasets' from the cwd

import torch

from epilearn.models.SpatialTemporal.STAN import STAN
from epilearn.data import Dataset
from epilearn.utils import metrics, utils


# ### Configs

device = torch.device('cpu')
torch.manual_seed(7)

lookback = 13   # input size
horizon = 3     # prediction size

epochs = 15     # kept small so the demo finishes in a few seconds
batch_size = 32


# ### Load the toy dataset

dataset = Dataset()                 # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
print(dataset)

# STAN is trained on daily changes in the SIR compartments, so difference the state
# trajectory. diff() drops one timestep, so the raw states have to be re-aligned.
daily = utils.diff(dataset.states)                  # (T-1, nodes, 3) = d(S, I, R)
features, _, _ = utils.normalize(daily.clone())
states = dataset.states[1:].to(device)              # raw counts: the SIR head needs them
features = features.to(device)

# The toy graph is a raw OD matrix (values up to 5.4e6): normalize before use.
adj_norm = utils.normalize_adj(dataset.graph.clone()).to(device)

# STAN's physics head needs a population size. Take the real one from the states.
population = float(dataset.states[0].sum(-1).mean())
print(f"mean region population: {population:,.0f}")


# ### Build the sliding windows
# 0.1.0: generate_dataset returns a DICT and does not fall back to dataset.states,
# so states= has to be passed explicitly.
# The target is the raw daily new infections and recoveries (channels 1 and 2),
# left unnormalized so the physics head, which predicts dI/dR in real counts,
# is scored on the same scale as the data-driven head.

daily = daily.to(device)


def make_split(start, end):
    split = dataset.generate_dataset(X=features[start:end],
                                     Y=daily[start:end, :, 1:],
                                     states=states[start:end],
                                     adj=adj_norm,
                                     lookback_window_size=lookback,
                                     horizon_size=horizon)
    # STAN returns (samples, horizon, nodes, 2); generate_dataset's default layout
    # puts horizon last, so transpose the targets to match the model.
    split['targets'] = split['targets'].transpose(1, 2)
    return split


train_split = make_split(0, 320)
val_split = make_split(320, 430)
test_split = make_split(430, daily.shape[0])
print({k: tuple(v.shape) for k, v in train_split.items() if torch.is_tensor(v)})


# ### Train

model = STAN(num_nodes=adj_norm.shape[0],
             num_features=train_split['features'].shape[3],
             num_timesteps_input=lookback,
             num_timesteps_output=horizon,
             population=population,
             gat_dim1=32,
             gat_dim2=32,
             gru_dim=32,
             num_heads=1,
             device=device).to(device)

model.fit(train_input=train_split['features'],
          train_target=train_split['targets'],
          train_states=train_split['states'],
          train_graph=adj_norm,
          val_input=val_split['features'],
          val_target=val_split['targets'],
          val_states=val_split['states'],
          val_graph=adj_norm,
          loss='stan',
          verbose=False,
          batch_size=batch_size,
          epochs=epochs,
          lr=1e-2)


# ### Evaluate
# STAN's forward returns two tensors, so call it directly instead of model.predict()
# (which reads a 2-tuple as (prediction, uncertainty_dict)).

with torch.no_grad():
    model.eval()
    pred, pred_physics = model(test_split['features'], adj_norm, test_split['states'])

targets = test_split['targets']
print(f"\npredictions {tuple(pred.shape)} (samples, horizon, nodes, [dI, dR])")
print(f"mean |daily change| in the test window : {targets.abs().mean():.2f}")
print(f"MAE, data-driven head                 : {metrics.get_MAE(pred, targets).item():.2f}")
print(f"MAE, physics (SIR) head               : {metrics.get_MAE(pred_physics, targets).item():.2f}")
