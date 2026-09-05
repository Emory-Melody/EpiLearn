"""Nowcasting demo: correct a reporting delay with NowcastTask, and check the
nowcast against the naive "trust the latest report" baseline plus its per-fold
conformal prediction intervals.

Run from the repository root:

    python tests/nowcast.py

Self-contained: the reporting triangle is synthetic, so nothing is downloaded.
"""

import numpy as np
import torch

from epilearn.models.Temporal import GRUModel
from epilearn.regime import reporting_completeness
from epilearn.tasks.nowcast import NowcastTask

np.random.seed(7)
torch.manual_seed(7)
# This GRU is tiny; on a many-core box the default thread pool costs far more in
# synchronisation than it saves.
torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# 1. Build a synthetic reporting triangle.
#
#    Row t of the triangle is the *cumulative* count for day t as it looked
#    1, 2, ... 9 days later. Recent days are therefore still incomplete, which
#    is exactly the problem nowcasting solves.
# ---------------------------------------------------------------------------
n_days = 420
delays = np.arange(1, 10)          # reports trickle in over delays 1..9
mean_delay = 2.5                   # exponential reporting delay, in days

rng = np.random.default_rng(0)
final = np.round(100 + 60 * np.sin(np.arange(n_days) / 18) + rng.normal(0, 4, n_days))
share = np.diff(1 - np.exp(-np.r_[0, delays] / mean_delay))
share /= share.sum()
triangle = np.cumsum([rng.multinomial(int(c), share) for c in final], axis=1)

print(f"triangle shape      : {triangle.shape} (days x delays)")
for d in (0, 2, 5):
    completeness = np.nanmean(reporting_completeness(triangle, delay=d))
    print(f"  reported by delay {delays[d]}: {completeness:.1%} of the final count")

# ---------------------------------------------------------------------------
# 2. Nowcast the last 7 days, whose reports are still incomplete.
# ---------------------------------------------------------------------------
lookback = 14      # days of reporting history fed to the model
horizon = 7        # days to nowcast

task = NowcastTask(prototype=GRUModel,
                   lookback=lookback,
                   horizon=horizon,
                   min_delay=int(delays[0]),
                   max_delay=int(delays[-1]),
                   device='cpu')
dataset = task.create_dataset(triangle, final, delays=delays)
print(f"\nnowcast samples     : {dataset.n_timesteps}")
print(f"features            : {tuple(dataset.x.shape)} "
      f"(samples, lookback, regions, delays; -1 = not yet reported)")
print(f"targets             : {tuple(dataset.y.shape)} (samples, regions, horizon)")

# ---------------------------------------------------------------------------
# 3. Rolling-window training. Every fold also calibrates a conformal interval
#    on its own validation window.
# ---------------------------------------------------------------------------
result = task.rolling_train(dataset,
                            train_size=150,
                            val_size=40,
                            test_size=40,
                            step_size=40,
                            max_folds=3,
                            epochs=60,
                            batch_size=32,
                            lr=1e-2,
                            conformal_alpha=0.1,
                            verbose=False)

# ---------------------------------------------------------------------------
# 4. Results: the model vs. simply believing the latest report.
# ---------------------------------------------------------------------------
agg = result['aggregate_metrics']
naive = task.compute_naive_baseline(dataset)

print("\n" + "=" * 64)
print("NOWCAST vs NAIVE LATEST-REPORT BASELINE")
print("=" * 64)
print(f"nowcast MAE         : {agg['mae_mean']:.2f} +/- {agg['mae_std']:.2f} cases "
      f"({agg['n_folds']} folds)")
print(f"nowcast RMSE        : {agg['rmse_mean']:.2f} +/- {agg['rmse_std']:.2f} cases")
print(f"latest-report MAE   : {naive['naive_mae']:.2f} cases "
      f"({naive['n_samples']} day-slots)")
print(f"error reduction     : "
      f"{100 * (1 - agg['mae_mean'] / naive['naive_mae']):+.1f}%")

# Per-fold conformal intervals: one calibrated quantile per fold, plus the
# empirical coverage and mean width of [pred - q, pred + q] on that fold's test set.
print("\n" + "=" * 64)
print("PER-FOLD CONFORMAL INTERVALS (90% target)")
print("=" * 64)
print(f"{'fold':>4} {'MAE':>9} {'quantile':>10} {'coverage':>9} {'width':>9}")
for fold, ci in zip(result['fold_results'], result['conformal_intervals']):
    print(f"{fold['fold']:>4} {fold['mae']:>9.2f} {ci['quantile']:>10.2f} "
          f"{fold['coverage']:>8.1%} {fold['interval_width']:>9.2f}")
print(f"{'all':>4} {agg['mae_mean']:>9.2f} {agg['conformal_quantile_mean']:>10.2f} "
      f"{agg['coverage_mean']:>8.1%} {agg['interval_width_mean']:>9.2f}")

lower, upper = result['conformal_intervals'][-1]['lower'], result['conformal_intervals'][-1]['upper']
print(f"\nlast fold, first nowcast day: [{lower[0, 0]:.1f}, {upper[0, 0]:.1f}] "
      f"vs truth {result['all_targets'][-1][0, 0]:.1f}")
