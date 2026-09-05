#!/usr/bin/env python
# coding: utf-8
"""
Nowcasting pipeline (new in EpiLearn 0.1.0).

Nowcasting corrects for reporting delay. Today's case count is incomplete: cases
that happened today keep arriving for days or weeks. The data structure for this
is a *reporting triangle* ``triangle[t, d]`` = how many cases with event date ``t``
had been reported by delay ``d``. The task is to predict, for the most recent
``horizon`` days, the count each will eventually be revised up to.

``NowcastTask`` (aliased as ``Nowcast``) builds the sliding windows for you:

    task.create_dataset(triangle, final_counts, delays=...)  -> Dataset
    task.rolling_train(dataset, train_size=, val_size=, test_size=, ...)
    task.compute_naive_baseline(dataset)                     -> "trust the latest report"

Run it with::

    python examples/nowcast_task.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import numpy as np
import torch
import matplotlib.pyplot as plt

from epilearn.models.Temporal import GRUModel, LSTMModel, MLPModel
from epilearn.models.Temporal.StatsModel import RKINowcastModel, NobBSModel
from epilearn.tasks.nowcast import NowcastTask


torch.manual_seed(7)


# ### Configs

lookback = 14   # days of reporting history used as input
horizon = 7     # nowcast the last 7 days, whose reports are still incomplete
min_delay = 1
max_delay = 9


# ### Build a synthetic reporting triangle
# Day t's cases trickle in over delays 1..9, following an exponential
# reporting-delay distribution. triangle[t, d] is the cumulative count reported
# for day t by delay d; final[t] is the value it eventually converges to.

n_days = 260
rng = np.random.default_rng(0)
delays = np.arange(min_delay, max_delay + 1)
final = np.round(100 + 60 * np.sin(np.arange(n_days) / 18) + rng.normal(0, 4, n_days))
share = np.diff(1 - np.exp(-np.r_[0, delays] / 2.5))
share /= share.sum()
triangle = np.cumsum([rng.multinomial(int(count), share) for count in final], axis=1)

print(f"triangle       : {triangle.shape} (days, delays)")
print(f"final counts   : {final.shape}")
print(f"day 100 reported so far: {triangle[100].tolist()} -> final {int(final[100])}")

plt.figure(figsize=(11, 4))
plt.plot(final, 'k-', label='final (fully reported)')
for d in (0, 2, 5):
    plt.plot(triangle[:, d], alpha=0.7, label=f'reported by delay {delays[d]}')
plt.xlabel("event day")
plt.ylabel("cases")
plt.title("Reporting triangle: recent days are always incomplete")
plt.legend()
plt.tight_layout()
plt.show()


# ### Initialize the task and build the dataset

task = NowcastTask(prototype=GRUModel,
                   lookback=lookback,
                   horizon=horizon,
                   min_delay=min_delay,
                   max_delay=max_delay,
                   device='cpu')
dataset = task.create_dataset(triangle, final, delays=delays)
print(f"\nnowcast dataset: {dataset}")
print(f"  x: (samples, lookback, regions, delays) = {tuple(dataset.x.shape)}")
print(f"  y: (samples, regions, horizon)          = {tuple(dataset.y.shape)}")
print("  unobserved entries in the features are marked -1")


# ### Train with rolling evaluation

result = task.rolling_train(dataset,
                            train_size=140,
                            val_size=40,
                            test_size=40,
                            epochs=60,
                            batch_size=32,
                            lr=1e-2)

naive = task.compute_naive_baseline(dataset)
print(f"\nnowcast MAE      : {result['aggregate_metrics']['mae_mean']:.4f}")
print(f"latest-report MAE: {naive['naive_mae']:.4f}   <- what you get by trusting the latest report")
print(f"90% conformal coverage: {result['aggregate_metrics']['coverage_mean']:.1%}")


# ### Compare a few models
# Any Temporal model works as prototype; RKINowcastModel and NobBSModel are
# nowcasting-specific statistical baselines whose fit() is essentially free.

print("\nmodel comparison (MAE, lower is better):")
for name, prototype in [('GRU', GRUModel),
                        ('LSTM', LSTMModel),
                        ('MLP', MLPModel),
                        ('RKI', RKINowcastModel),
                        ('NobBS', NobBSModel)]:
    other = NowcastTask(prototype=prototype,
                        lookback=lookback,
                        horizon=horizon,
                        min_delay=min_delay,
                        max_delay=max_delay,
                        device='cpu')
    other_ds = other.create_dataset(triangle, final, delays=delays)
    other_result = other.rolling_train(other_ds,
                                       train_size=140,
                                       val_size=40,
                                       test_size=40,
                                       epochs=30,
                                       batch_size=32,
                                       lr=1e-2)
    print(f"  {name:6s}: {other_result['aggregate_metrics']['mae_mean']:8.4f}")
print(f"  {'naive':6s}: {naive['naive_mae']:8.4f}")


# ### Look at one fold's nowcasts

# The nowcast task collapses the single-region axis, so predictions and targets
# come back as (samples, horizon).
predictions = np.concatenate([p.numpy() for p in result['all_predictions']])
targets = np.concatenate([t.numpy() for t in result['all_targets']])
print(f"\nstacked predictions {predictions.shape} (samples, horizon)")

plt.figure(figsize=(11, 4))
plt.plot(targets[:, -1], 'k-', label='final count')
plt.plot(predictions[:, -1], 'r--', label='nowcast')
plt.xlabel("test sample")
plt.ylabel("cases")
plt.title(f"Nowcast of the most recent day (horizon step {horizon})")
plt.legend()
plt.tight_layout()
plt.show()
