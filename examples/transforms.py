#!/usr/bin/env python
# coding: utf-8
"""
Transforms (EpiLearn 0.1.0).

The one thing to remember when upgrading: ``Compose.__call__`` now returns a
TUPLE, ``(data, process_history)``. Code written as ``data = transformation(data)``
silently ends up holding the tuple.

``process_history`` replaces the old ``Compose.feat_mean`` / ``Compose.feat_std``
attributes. On a Dataset you reach it through ``Dataset.get_process_history()``.

Run it with::

    python examples/transforms.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
# load_toy_dataset() resolves './datasets' relative to the working directory.
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch

from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.models.Temporal.GRU import GRUModel
from epilearn.tasks.forecast import Forecast


torch.manual_seed(7)

lookback = 12
horizon = 3

dataset = Dataset()              # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
dataset.dynamic_graph = None
print(dataset)


# ## 1. Compose returns (data, process_history)

transformation = transforms.Compose({
    'features': [transforms.normalize_feat()],
    'target': [transforms.normalize_target()],
    'graph': [transforms.normalize_adj()],
})

data, process_history = transformation({
    'features': dataset.x.clone(),
    'target': dataset.y.clone(),
    'graph': dataset.graph.clone(),
})
print("\ntransformed keys :", sorted(data.keys()))
print("process history  :", sorted(process_history.keys()))
print(f"feat_mean={float(process_history['feat_mean']):.4f} "
      f"feat_std={float(process_history['feat_std']):.4f}")
print(f"target_mean={float(process_history['target_mean']):.4f} "
      f"target_std={float(process_history['target_std']):.4f}")


# ## 2. The same thing through the Dataset API
# set_transforms(..., apply_now=True) normalizes the dataset in place, and the
# statistics land in get_process_history().

dataset.set_transforms(transformation, apply_now=True)
stats = dataset.get_process_history()
target_mean = float(stats['target_mean'])
target_std = float(stats['target_std'])
print(f"\nnormalized target: mean={dataset.y.mean():.4f}, std={dataset.y.std():.4f}")

# normalize_feat / normalize_target use one global mean and std (not per node),
# so undoing them is a single affine map. This is what you need at evaluation
# time, because 0.1.0 does not auto-denormalize metrics.
reference = Dataset()
reference.load_toy_dataset()
restored = dataset.y * target_std + target_mean
print(f"round-trip max abs error: {(restored - reference.y).abs().max():.6f}")


# ## 3. The signal-processing transforms want windowed data
# normalize_* and normalize_adj work on the raw (timesteps, nodes, channels)
# layout. add_time_embedding / convert_to_frequency / learnable_time_embedding /
# seasonality_and_trend_decompose instead expect time on axis 2, i.e. the
# (samples, nodes, timesteps[, channels]) layout you get after windowing.

split = dataset.generate_dataset(X=dataset.x, Y=dataset.y,
                                 adj=dataset.graph,
                                 lookback_window_size=lookback,
                                 horizon_size=horizon)
# generate_dataset gives (samples, timesteps, nodes, channels); swap to put time on axis 2
windowed = split['features'][:8].transpose(1, 2).contiguous()
print(f"\nwindowed features: {tuple(windowed.shape)} (samples, nodes, timesteps, channels)")

time_embedded = transforms.add_time_embedding(embedding_dim=4)(windowed, device='cpu')
print(f"add_time_embedding              -> {tuple(time_embedded.shape)}")

learned = transforms.learnable_time_embedding(timesteps=lookback, embedding_dim=4)(windowed, device='cpu')
print(f"learnable_time_embedding        -> {tuple(learned.shape)}")

spectrum = transforms.convert_to_frequency(ftype='fft')(windowed, device='cpu')
print(f"convert_to_frequency('fft')     -> {tuple(spectrum.shape)}")

# decomposition needs a single channel and returns a LIST [seasonality, trend],
# so it cannot be chained inside a Compose.
one_channel = windowed[..., 0].contiguous()
seasonality, trend = transforms.seasonality_and_trend_decompose()(one_channel, device='cpu')
print(f"seasonality_and_trend_decompose -> {tuple(seasonality.shape)} + {tuple(trend.shape)}")


# ## 4. A Compose inside a real pipeline
# Registering transforms with set_transforms (without apply_now) makes
# rolling_train refit the normalization on every training fold, so the
# validation and test windows never leak their statistics into training.

region = 0
region_ds = Dataset(x=reference.x[:, region, :],
                    y=reference.y[:, region:region + 1])
region_ds.set_transforms(transforms.Compose({
    'features': [transforms.normalize_feat()],
    'target': [transforms.normalize_target()],
}))

task = Forecast(prototype=GRUModel, lookback=lookback, horizon=horizon, device='cpu')
result = task.rolling_train(dataset=region_ds,
                            train_size=300,
                            val_size=60,
                            test_size=60,
                            train_loss='mse',
                            epochs=10,
                            batch_size=32,
                            model_args={'nhid': 32})

print("\nper-fold scores:")
for fold in result['fold_results']:
    print(f"  fold {fold['fold']}: mae={fold['mae']:.4f} "
          f"(train {fold['train_timestamps']}, test {fold['test_timestamps']})")

# Each fold refits the statistics on its own (growing) training window:
for end in (300, 360, 420):
    probe = transforms.Compose({'target': [transforms.normalize_target()]})
    _, hist = probe({'target': reference.y[:end, region:region + 1].clone()})
    print(f"  train[0:{end}] -> target_mean={float(hist['target_mean']):8.3f}, "
          f"target_std={float(hist['target_std']):8.3f}")

# Caveat: every fold stores a reference to the SAME process-history dict, so
# fold_results[i]['process_history'] all read back the statistics of the LAST
# training window. Inverse-transform the last fold, or copy the stats yourself.
final_stats = result['fold_results'][-1]['process_history']
print(f"\nstats used for inverse transform: target_mean={float(final_stats['target_mean']):.3f}, "
      f"target_std={float(final_stats['target_std']):.3f}")
