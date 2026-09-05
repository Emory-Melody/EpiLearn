#!/usr/bin/env python
# coding: utf-8
"""
Temporal models without a Task (EpiLearn 0.1.0).

The Forecast / Detection / Nowcast tasks wrap a lot of bookkeeping. This example
does the same work by hand, which is what you want when you are debugging a model
or plugging EpiLearn's model zoo into your own training loop:

    set_transforms(..., apply_now=True) -> get_process_history()
        -> generate_dataset() -> model.fit() -> model.predict() -> denormalize

Note that ``Dataset.get_transformed()`` and ``Compose.feat_mean`` / ``.feat_std``
were removed in 0.1.0; the replacements are ``apply_transforms()`` and the
``get_process_history()`` dict.

Run it with::

    python examples/temporal.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
# load_toy_dataset() resolves './datasets' relative to the working directory.
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch
import matplotlib.pyplot as plt

from epilearn.models.Temporal.LSTM import LSTMModel
from epilearn.models.Temporal.GRU import GRUModel
from epilearn.models.Temporal.Dlinear import DlinearModel
# 0.0.x path epilearn.models.Temporal.ARIMA is now Temporal.StatsModel
from epilearn.models.Temporal.StatsModel import VARMAXModel, ARIMAModel, SeasonalNaiveModel

from epilearn.data import Dataset
from epilearn.utils import metrics, transforms


# ### Configs

device = torch.device('cpu')
torch.manual_seed(7)

lookback = 12   # input size
horizon = 3     # prediction size

epochs = 20
batch_size = 50

region = 0      # temporal models see one region at a time


# ### Load and transform the dataset

dataset = Dataset()              # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()

transformation = transforms.Compose({
    'features': [transforms.normalize_feat()],
    'target': [transforms.normalize_target()],
    'graph': [transforms.normalize_adj()],
    'states': [],
})

# apply_now=True normalizes dataset.x / .y / .graph / .dynamic_graph in place.
dataset.set_transforms(transformation, apply_now=True)

# The normalization statistics live in the process-history dict (0.0.x read them
# off transformation.feat_mean / .feat_std, which no longer exist).
stats = dataset.get_process_history()
print("process history keys:", sorted(stats.keys()))
target_mean = float(stats['target_mean'])
target_std = float(stats['target_std'])

features = dataset.x.to(device)
target = dataset.y.to(device)
states = dataset.states.to(device)


# ### Split the data by time

train_rate = 0.6
val_rate = 0.2

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))


def make_split(start, end):
    # generate_dataset returns a DICT: features / targets / states / dynamic_graph / graph
    # (0.0.x unpacked a 4-tuple here). A temporal model ignores the graphs, so we
    # do not bother passing dynamic_adj.
    return dataset.generate_dataset(X=features[start:end],
                                    Y=target[start:end],
                                    states=states[start:end],
                                    adj=dataset.graph,
                                    lookback_window_size=lookback,
                                    horizon_size=horizon)


train_split = make_split(0, split_line1)
val_split = make_split(split_line1, split_line2)
test_split = make_split(split_line2, features.shape[0])
print({k: tuple(v.shape) for k, v in train_split.items() if v is not None})

# Keep a single region: features (samples, lookback, channels), target (samples, horizon)
train_input = train_split['features'][:, :, region, :]
train_target = train_split['targets'][:, region, :]

val_input = val_split['features'][:, :, region, :]
val_target = val_split['targets'][:, region, :]

test_input = test_split['features'][:, :, region, :]
test_target = test_split['targets'][:, region, :]

print(f"train_input {tuple(train_input.shape)} (samples, timesteps, features)")


# ### Prepare the model
# Any Temporal model works here; XGBModel was dropped in 0.1.0, and the modern
# alternatives are the scikit-learn wrappers in epilearn.models.Temporal.ScikitModel.

model = GRUModel(num_features=train_input.shape[2],
                 num_timesteps_input=lookback,
                 num_timesteps_output=horizon,
                 nhid=32).to(device)

# model = LSTMModel(num_features=train_input.shape[2],
#                   num_timesteps_input=lookback,
#                   num_timesteps_output=horizon).to(device)

# model = DlinearModel(num_features=train_input.shape[2],
#                      num_timesteps_input=lookback,
#                      num_timesteps_output=horizon).to(device)

# model = VARMAXModel(num_features=train_input.shape[2],
#                     num_timesteps_input=lookback,
#                     num_timesteps_output=horizon)


# ### Train

model.fit(train_input=train_input,
          train_target=train_target,
          val_input=val_input,
          val_target=val_target,
          verbose=True,
          batch_size=batch_size,
          epochs=epochs)


# ### Evaluate
# Metrics are NOT auto-denormalized in 0.1.0, so undo normalize_target() by hand.

out = model.predict(feature=test_input)
preds = out.detach().cpu() * target_std + target_mean
targets = test_target.detach().cpu() * target_std + target_mean
print(f"GRU MAE: {metrics.get_MAE(preds, targets).item():.4f}")

# A statistical baseline for reference: fit() is a no-op, predict() fits per window.
baseline = SeasonalNaiveModel(num_features=test_input.shape[2],
                              num_timesteps_input=lookback,
                              num_timesteps_output=horizon,
                              season_length=lookback)
baseline_out = baseline.predict(feature=test_input)
baseline_preds = baseline_out.detach().cpu() * target_std + target_mean
print(f"SeasonalNaive MAE: {metrics.get_MAE(baseline_preds, targets).item():.4f}")


# ### Visualize the fit on the training window

out = model.predict(feature=train_input).detach().cpu()

num_samples = 40    # number of samples to display
time_points = horizon

plt.figure(figsize=(15, 5))
for t in range(time_points):
    plt.subplot(1, time_points, t + 1)

    predictions = out[:num_samples, t]
    truths = train_target[:num_samples, t]

    plt.plot(range(num_samples), predictions.numpy(), 'r-', label='Prediction')
    plt.plot(range(num_samples), truths.numpy(), 'b--', label='Ground Truth')
    plt.title(f"Time Point {t + 1}")
    plt.xlabel("Sample Index")
    plt.ylabel("Value (normalized)")
    plt.legend()

plt.tight_layout()
plt.show()
