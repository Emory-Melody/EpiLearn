"""Ensembling demo: label each rolling fold with an epidemic regime, then show
that combining five weak forecasters beats every one of them individually.

Run from the repository root:

    python tests/ensemble.py

Pure numpy -- no torch training, runs in about a second. The five "base models"
are real (if simple) forecasting rules applied to a simulated multi-wave
epidemic, so their errors genuinely differ and genuinely cancel.
"""

import numpy as np
import pandas as pd

from epilearn import strategies
from epilearn.ensemble import compute_nrmse
from epilearn.regime import REGIME_LABELS, assign_regimes, compute_regime_features

# ---------------------------------------------------------------------------
# 1. A simulated epidemic: four waves of different height and width, a weekly
#    reporting cycle, and multiplicative noise.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(3)
n_days, lookback, horizon = 900, 21, 4

t = np.arange(n_days)
waves = (320 * np.exp(-((t - 110) / 26.) ** 2)
         + 210 * np.exp(-((t - 280) / 45.) ** 2)
         + 470 * np.exp(-((t - 640) / 22.) ** 2)
         + 280 * np.exp(-((t - 820) / 34.) ** 2))
series = ((25 + waves)
          * (1 + 0.12 * np.sin(2 * np.pi * t / 7))
          * (1 + 0.05 * rng.standard_normal(n_days)))
series = np.clip(series, 0.0, None)

# Sliding windows: lookback days of history -> the next `horizon` days.
starts = np.arange(lookback, n_days - horizon)
history = np.stack([series[i - lookback:i] for i in starts])   # (n, lookback)
targets = np.stack([series[i:i + horizon] for i in starts])    # (n, horizon)


# ---------------------------------------------------------------------------
# 2. Five base forecasters with deliberately different failure modes.
#    Predictions have the shape every epilearn strategy expects:
#    (n_models, n_samples, horizon).
# ---------------------------------------------------------------------------
def _slope(h, window=7):
    """Least-squares slope of the last `window` observations."""
    x = np.arange(window) - (window - 1) / 2
    centred = h[:, -window:] - h[:, -window:].mean(axis=1, keepdims=True)
    return (centred * x).sum(axis=1) / (x ** 2).sum()


def persistence(h):
    """Repeat the latest value; always a step behind a moving epidemic."""
    return np.repeat(h[:, -1:], horizon, axis=1)


def linear_trend(h, damping=1.0):
    """Extrapolate the local slope; overshoots hard at turning points."""
    steps = np.cumsum(damping ** np.arange(1, horizon + 1))
    return h[:, -1:] + _slope(h)[:, None] * steps[None, :]


base_models = {
    'Persistence': persistence,
    'LinearTrend': linear_trend,
    'DampedTrend': lambda h: linear_trend(h, damping=0.6),
    'MeanReverter': lambda h: np.repeat(
        0.6 * h[:, -1:] + 0.4 * h.mean(axis=1, keepdims=True), horizon, axis=1),
    'SeasonalNaive': lambda h: h[:, -7:][:, np.arange(horizon) % 7],
}
model_names = list(base_models)
predictions = np.stack([f(history) for f in base_models.values()])
print(f"predictions : {predictions.shape} (models, samples, horizon)")
print(f"targets     : {targets.shape}")

# ---------------------------------------------------------------------------
# 3. Cut the samples into rolling folds and label each fold's regime.
#    `assign_regimes` picks the level/volatility quantiles from all the folds,
#    then labels each one with one of the seven REGIME_LABELS.
# ---------------------------------------------------------------------------
n_folds, n_train_folds = 9, 6
edges = np.linspace(0, len(starts), n_folds + 1).astype(int)
folds = [{'predictions': predictions[:, a:b], 'targets': targets[a:b]}
         for a, b in zip(edges[:-1], edges[1:])]

features = pd.DataFrame([compute_regime_features(f['targets']) for f in folds])
regimes = list(assign_regimes(features)['regime'])

print("\n" + "=" * 72)
print("ROLLING FOLDS AND THEIR REGIMES")
print("=" * 72)
print(f"{'fold':>4} {'split':>6} {'samples':>8} {'level':>9} {'growth':>8} {'regime':>18}")
for i, fold in enumerate(folds):
    split = 'train' if i < n_train_folds else 'TEST'
    print(f"{i:>4} {split:>6} {len(fold['targets']):>8} "
          f"{features['level'][i]:>9.1f} {features['growth_rate'][i]:>8.2f} "
          f"{regimes[i]:>18}")
print(f"\n{len(set(regimes))} of the {len(REGIME_LABELS)} regime labels occur here: "
      f"{sorted(set(regimes))}")

train_folds, test_folds = folds[:n_train_folds], folds[n_train_folds:]
train_regimes = {i: regimes[i] for i in range(n_train_folds)}
train_all = {
    'predictions': np.concatenate([f['predictions'] for f in train_folds], axis=1),
    'targets': np.concatenate([f['targets'] for f in train_folds], axis=0),
}

# ---------------------------------------------------------------------------
# 4. Score every base model, then every ensemble strategy, on the test folds.
#    All numbers are NRMSE = RMSE / std(targets), averaged over the test folds.
# ---------------------------------------------------------------------------
member_scores = {
    name: np.mean([compute_nrmse(f['predictions'][m], f['targets'])
                   for f in test_folds])
    for m, name in enumerate(model_names)
}

candidates = [
    strategies.MeanEnsemble(),
    strategies.MedianEnsemble(),
    strategies.TrimmedMeanEnsemble(trim_frac=0.2),
    strategies.InverseNRMSEEnsemble(),
    strategies.ConstrainedStackingEnsemble(),
    strategies.StackingEnsemble(alpha=1.0),
    strategies.RegimeStackingEnsemble(alpha=1.0),
]

ensemble_scores, fitted = {}, {}
for strategy in candidates:
    # Regime stacking needs the fold structure to learn one weight set per regime.
    fit_kwargs = {}
    if isinstance(strategy, strategies.RegimeStackingEnsemble):
        fit_kwargs = dict(fold_predictions=[f['predictions'] for f in train_folds],
                          fold_targets=[f['targets'] for f in train_folds],
                          fold_regimes=train_regimes)

    ensemble = strategies.EnsembleModel(strategy).rolling_train(train_all, **fit_kwargs)
    scores = []
    for i, fold in enumerate(test_folds, start=n_train_folds):
        predict_kwargs = ({'regime': regimes[i]}
                          if isinstance(strategy, strategies.RegimeStackingEnsemble)
                          else {})
        scores.append(ensemble.evaluate(fold, **predict_kwargs)['nrmse'])
    ensemble_scores[ensemble.name] = np.mean(scores)
    fitted[ensemble.name] = ensemble

print("\n" + "=" * 72)
print("TEST-FOLD NRMSE (lower is better)")
print("=" * 72)
for name, score in sorted(member_scores.items(), key=lambda kv: kv[1]):
    print(f"  member    {name:<20} {score:.4f}")
print()
for name, score in sorted(ensemble_scores.items(), key=lambda kv: kv[1]):
    print(f"  ensemble  {name:<20} {score:.4f}")

best_member = min(member_scores, key=member_scores.get)
worst_ensemble = max(ensemble_scores, key=ensemble_scores.get)
best_ensemble = min(ensemble_scores, key=ensemble_scores.get)
print(f"\nbest single model : {best_member} at {member_scores[best_member]:.4f}")
print(f"worst ensemble    : {worst_ensemble} at {ensemble_scores[worst_ensemble]:.4f} "
      f"({100 * (1 - ensemble_scores[worst_ensemble] / member_scores[best_member]):+.1f}% "
      f"vs the best single model)")
print(f"best ensemble     : {best_ensemble} at {ensemble_scores[best_ensemble]:.4f} "
      f"({100 * (1 - ensemble_scores[best_ensemble] / member_scores[best_member]):+.1f}%)")
assert max(ensemble_scores.values()) < min(member_scores.values()), \
    "every ensemble is expected to beat every single member here"
print("-> every ensemble strategy beats every individual member.")

# Learned weights are the interpretable part: which members the ensemble trusts.
weights = fitted['ConstrainedStack'].get_weights().mean(axis=0)   # (horizon, m) -> (m,)
print("\nConstrainedStack weights (convex, averaged over the horizon):")
for name, w in sorted(zip(model_names, weights), key=lambda kv: -kv[1]):
    print(f"  {name:<20} {w:.3f}")
