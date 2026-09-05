"""Low-level forecasting demo: build the sliding windows yourself, then call model.fit.

This is the manual path that the Forecast task wraps -- normalize by hand,
turn a (time, node, feature) tensor into supervised windows with
``Dataset.generate_dataset``, and train an STGCN directly. Any other
``SpatialTemporal`` model takes the same constructor arguments, so swapping the
model class is the only edit needed.

Run from the repo root:

    python tests/test.py
"""

import torch

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.data import Dataset
from epilearn.utils import utils, metrics

device = torch.device('cpu')
torch.manual_seed(7)

# settings
lookback = 13    # input window length
horizon = 3      # steps predicted
epochs = 10
batch_size = 50

# load toy dataset: x=(539, 47, 4), graph=(47, 47)
dataset = Dataset()
dataset.load_toy_dataset()

# preprocessing: z-score the features, degree-normalize the mobility graph
# (the toy graph is a raw OD matrix with entries up to 5.4e6)
features, mean, std = utils.normalize(dataset.x)
adj_norm = utils.normalize_adj(dataset.graph)
features = features.to(device)
adj_norm = adj_norm.to(device)
targets = features[:, :, 0]    # channel 0 = case counts, in normalized units

# chronological split: 60% train, 20% val, 20% test
split1 = int(features.shape[0] * 0.6)
split2 = int(features.shape[0] * 0.8)


def make_windows(start, end):
    """Slice a time range and cut it into (lookback -> horizon) samples.

    In 0.1.0 generate_dataset returns a DICT with keys
    features / targets / states / dynamic_graph / graph -- it is no longer a
    4-tuple. It also does not fall back to dataset.graph, so pass adj= yourself.
    """
    return dataset.generate_dataset(X=features[start:end],
                                    Y=targets[start:end],
                                    adj=adj_norm,
                                    lookback_window_size=lookback,
                                    horizon_size=horizon)


train_split = make_windows(0, split1)
val_split = make_windows(split1, split2)
test_split = make_windows(split2, features.shape[0])

print(f"train features {tuple(train_split['features'].shape)} "
      f"targets {tuple(train_split['targets'].shape)}")
print(f"test  features {tuple(test_split['features'].shape)} "
      f"targets {tuple(test_split['targets'].shape)}")

# prepare model
model = STGCN(num_nodes=adj_norm.shape[0],
              num_features=train_split['features'].shape[3],
              num_timesteps_input=lookback,
              num_timesteps_output=horizon,
              device=device).to(device=device)

# training: the graph goes in as train_graph= / val_graph=
model.fit(train_input=train_split['features'],
          train_target=train_split['targets'],
          train_graph=train_split['graph'],
          val_input=val_split['features'],
          val_target=val_split['targets'],
          val_graph=val_split['graph'],
          loss='mse',
          epochs=epochs,
          batch_size=batch_size,
          patience=epochs,
          verbose=True)

# evaluation
preds = model.predict(feature=test_split['features'], graph=test_split['graph'])
mae = metrics.get_MAE(preds, test_split['targets']).item()
rmse = metrics.get_RMSE(preds, test_split['targets']).item()

print(f"\ntest samples: {preds.shape[0]}, predictions {tuple(preds.shape)}")
print(f"MAE  {mae:.4f} (normalized units) | {mae * std[0].item():.1f} cases")
print(f"RMSE {rmse:.4f} (normalized units) | {rmse * std[0].item():.1f} cases")
