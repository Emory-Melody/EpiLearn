"""EpiColaGNN forecast demo: hand-built sliding windows fed straight to model.fit.

Run from the repo root:
    python tests/test_epi_cola.py
"""

import torch

from epilearn.data import Dataset
from epilearn.models.SpatialTemporal.EpiColaGNN import EpiColaGNN
from epilearn.utils import utils
from epilearn.utils.metrics import get_MAE, get_R2, get_RMSE

# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 13    # inputs size
horizon = 3      # predicts size

epochs = 8       # training epochs (kept small so the demo stays fast)
batch_size = 50  # training batch size
lr = 5e-3        # learning rate

# load toy dataset: x=(539, 47, 4), y = x[..., 0] (weekly cases per region)
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
    # (it used to return a 4-tuple). It also no longer falls back to
    # dataset.graph, so the adjacency has to be passed in via adj=.
    # EpiColaGNN.forward binarizes the adjacency it is given *in place*, so hand
    # each split its own copy and keep adj_norm itself intact.
    return dataset.generate_dataset(X=features[start:end],
                                    Y=features[start:end, :, 0],
                                    adj=adj_norm.clone(),
                                    lookback_window_size=lookback,
                                    horizon_size=horizon)


train_split = make_split(0, split_line1)
val_split = make_split(split_line1, split_line2)
test_split = make_split(split_line2, num_steps)

print(f"train inputs {tuple(train_split['features'].shape)} "
      f"-> targets {tuple(train_split['targets'].shape)}")

# prepare model
model = EpiColaGNN(num_nodes=adj_norm.shape[0],
                   num_features=train_split['features'].shape[3],
                   num_timesteps_input=lookback,
                   num_timesteps_output=horizon,
                   nhid=32,
                   device=device).to(device=device)

# training.
# NOTE: 0.0.x demos used loss='epi_cola' here, because EpiColaGNN.forward used to
# return a (prediction, epidemiological_prediction) pair and epi_cola_loss scored
# both heads. In 0.1.0 forward returns the prediction tensor only, so 'epi_cola'
# can no longer unpack it -- use 'mse'. epilearn.utils.metrics.epi_cola_loss is
# still there for two-headed models.
model.fit(train_input=train_split['features'],
          train_target=train_split['targets'],
          train_graph=train_split['graph'],
          val_input=val_split['features'],
          val_target=val_split['targets'],
          val_graph=val_split['graph'],
          loss='mse',
          epochs=epochs,
          batch_size=batch_size,
          lr=lr,
          verbose=True)

# evaluation on the held-out tail
pred = model.predict(feature=test_split['features'], graph=test_split['graph'])
target = test_split['targets']

# Metrics are in normalized units (we normalized the features ourselves above),
# so rescale by the target channel's std to read them as cases per region.
print("\n--- EpiColaGNN, toy dataset, held-out tail ---")
print(f"test samples        : {target.shape[0]}")
print(f"MAE  (normalized)   : {get_MAE(pred, target):.4f}")
print(f"RMSE (normalized)   : {get_RMSE(pred, target):.4f}")
print(f"MAE  (cases/region) : {get_MAE(pred, target) * std[0]:.2f}")
print(f"R2                  : {get_R2(pred, target):.4f}")
