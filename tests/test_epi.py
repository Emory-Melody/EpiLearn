"""EpiGNN forecast demo: hand-built sliding windows (with SIR states) fed to model.fit.

Run from the repo root:
    python tests/test_epi.py
"""

import torch

from epilearn.data import Dataset
from epilearn.models.SpatialTemporal.EpiGNN import EpiGNN
from epilearn.utils import utils
from epilearn.utils.metrics import get_MAE, get_R2, get_RMSE

# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 13    # inputs size
horizon = 5      # predicts size

epochs = 15      # training epochs (kept small so the demo stays fast)
batch_size = 15  # training batch size
lr = 5e-3        # learning rate

# load toy dataset: x=(539, 47, 4), states=(539, 47, 3) SIR compartments
dataset = Dataset()
dataset.load_toy_dataset()

# preprocessing. The toy graph is an origin-destination matrix whose entries reach
# 5.4e6, so it must be normalized before it is fed to a graph model.
features, mean, std = utils.normalize(dataset.x)
features = features.to(device)
adj_norm = utils.normalize_adj(dataset.graph).to(device)

# prepare datasets: chronological 60 / 20 / 20 split
num_steps = features.shape[0]
split_line1 = int(num_steps * 0.6)
split_line2 = int(num_steps * 0.8)


def make_split(start, end):
    # In 0.1.0 generate_dataset returns a DICT with keys
    # features / targets / states / dynamic_graph / graph
    # (it used to return a 4-tuple). Nothing is inferred from the Dataset any
    # more: states= and adj= have to be passed explicitly or they come back None.
    return dataset.generate_dataset(X=features[start:end],
                                    Y=features[start:end, :, 0],
                                    states=dataset.states[start:end],
                                    adj=adj_norm,
                                    lookback_window_size=lookback,
                                    horizon_size=horizon)


train_split = make_split(0, split_line1)
val_split = make_split(split_line1, split_line2)
test_split = make_split(split_line2, num_steps)

print(f"train inputs {tuple(train_split['features'].shape)} "
      f"-> targets {tuple(train_split['targets'].shape)}, "
      f"states {tuple(train_split['states'].shape)}")

# initialize model
model = EpiGNN(num_nodes=adj_norm.shape[0],
               num_features=train_split['features'].shape[3],
               num_timesteps_input=lookback,
               num_timesteps_output=horizon,
               device=device).to(device=device)

# training
model.fit(train_input=train_split['features'],
          train_target=train_split['targets'],
          train_states=train_split['states'],
          train_graph=train_split['graph'],
          val_input=val_split['features'],
          val_target=val_split['targets'],
          val_states=val_split['states'],
          val_graph=val_split['graph'],
          loss='mse',
          epochs=epochs,
          batch_size=batch_size,
          lr=lr,
          verbose=True)

# evaluation on the held-out tail
pred = model.predict(feature=test_split['features'],
                     graph=test_split['graph'],
                     states=test_split['states'])
target = test_split['targets']

# Metrics are in normalized units (we normalized the features ourselves above),
# so rescale by the target channel's std to read them as cases per region.
print("\n--- EpiGNN, toy dataset, held-out tail ---")
print(f"test samples        : {target.shape[0]}")
print(f"MAE  (normalized)   : {get_MAE(pred, target):.4f}")
print(f"RMSE (normalized)   : {get_RMSE(pred, target):.4f}")
print(f"MAE  (cases/region) : {get_MAE(pred, target) * std[0]:.2f}")
print(f"R2                  : {get_R2(pred, target):.4f}")
