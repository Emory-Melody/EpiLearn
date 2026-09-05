# Migrating from EpiLearn 0.0.x to 0.1.0

0.1.0 is a large update, but only two things break existing scripts. Everything
else in the public API — transforms, model import paths, built-in dataset names,
`utils`, `visualize`, and every `Spatial` / `SpatialTemporal` model — is unchanged.

## 1. `UniversalDataset` is now `Dataset`

```python
from epilearn.data import Dataset    # was: UniversalDataset
```

The constructor is unchanged: `Dataset`'s parameters are a strict superset of the
old `UniversalDataset`'s, in the same order, so every existing call still works.
`load_toy_dataset()` and the built-in names (`JHU_covid`, `Measles`, `Tycho_v1`,
`Covid_<Country>`) behave exactly as before.

`UniversalDataset` is kept as a deprecated alias, so old code keeps importing.

Two old import paths are likewise aliased and still work, but should be updated:

| old | new |
|---|---|
| `epilearn.models.Temporal.SIR` | `epilearn.models.Temporal.Compartmental` |
| `epilearn.models.Temporal.ARIMA` | `epilearn.models.Temporal.StatsModel` |

## 2. `train_model` no longer splits the data for you

This is the real break, and there is no alias for it. Previously one call did
everything: split the dataset, train, and evaluate.

```python
# 0.0.x -- no longer works
result = task.train_model(dataset=dataset, loss='mse', epochs=50,
                          batch_size=5, train_rate=0.6, val_rate=0.2,
                          permute_dataset=True)
evaluation = task.evaluate_model()
```

In 0.1.0, evaluation is rolling-origin by default, because a single random split
leaks information in time series. Use `rolling_train`, which takes the dataset
directly and trains, evaluates, and calibrates conformal intervals per fold:

```python
# 0.1.0
result = task.rolling_train(dataset=dataset,
                            train_size=400, val_size=50, test_size=50,
                            train_loss='mse', epochs=50, batch_size=5)
print(result['aggregate_metrics'])
```

Keyword-by-keyword:

| 0.0.x | 0.1.0 |
|---|---|
| `dataset=` | first argument of `rolling_train` |
| `loss=` | `train_loss=` and `val_loss=` |
| `train_rate=` / `val_rate=` | `train_size=` / `val_size=` / `test_size=` (counts, not rates) |
| `permute_dataset=` | removed — models now handle the layout internally |
| `config=`, `region_idx=` | removed |
| `epochs=`, `batch_size=`, `lr=`, `patience=`, `verbose=`, `device=`, `model_args=` | unchanged |
| — | new: `weight_decay=`, `pretrained=`, `conformal_alpha=`, `use_optuna=`, `n_trials=`, `max_folds=`, `expanding=` |

`train_model` still exists as the lower-level, single-split trainer. It now
requires `train_split` / `val_split` / `test_split` dicts built by
`Dataset.generate_dataset(...)`, and raises `RuntimeError` if any is missing.
Use it when you want to control the split yourself (see the Detection example in
the README).

`evaluate_model()` also needs a split now: it takes `dataset=<split dict>`, not a
`Dataset`. After `rolling_train` you rarely need it — the metrics are already in
`result['aggregate_metrics']` and `result['fold_results']`.

## 3. Smaller changes worth knowing

These do not raise errors, so they are easy to miss:

- **`transforms.Compose.__call__` now returns `(data, process_history)`**, not
  just `data`. Code like `data = transformation(data)` silently gets a tuple.
- **`Dataset.generate_dataset` returns a dict**, not a tuple, with keys
  `features`, `targets`, `states`, `graph`, `dynamic_graph`. Its `permute` flag
  also flipped meaning: the new default `permute=False` matches the old
  `permute=True`.
- **Metrics are not auto-denormalized.** If you use
  `transforms.normalize_target()`, the reported MSE/MAE/RMSE and conformal
  interval widths are in normalized units. Pass `inverse_normalize=True` to
  `evaluate_model` to get original units back.
- **Two different `coverage` numbers.** Per-fold `coverage` from `rolling_train`
  is marginal (element-wise), while
  `epilearn.utils.uncertainty.static_conformal()['coverage']` is joint (all
  horizon steps covered simultaneously). They are not comparable — on the same
  data, joint 0.65 vs marginal 0.94.
- **Renamed attributes:** `dataset.features` → `dataset.feature_names`,
  `dataset.timestamp` → `dataset.timestamps`; `index` / `anual_population` /
  `coordinates` are now under `dataset.metadata`. `Dataset.save()` requires a
  `path`.
- **`Dataset.from_csv` renamed its keywords:** `feature_csv` → `file_path`,
  `node_id_col` → `region_col`, `time_col` → `timestamp_col`,
  `edge_csv` → `graph_file`.
- **`Forecast.plot_forecasts` is now `Forecast.plot_preds`.**
- **`Dataset.get_transformed()`, `ganerate_splits()`, `get_splits()` were
  removed.** Use `set_transforms()` / `apply_transforms()` and
  `rolling_splits()` or `get_slice()`.
- **`utils.simulation.Time_geo` was removed.** The new simulation helpers are
  `simulate_temporal_epidemic`, `simulate_spatiotemporal_individual` and
  `simulate_spatiotemporal_regions`.
- **`torch_geometric` is now required**, not optional — several
  `SpatialTemporal` models import it at package import time.
- **New hard dependencies:** `optuna`, `psutil`, `pandas`, `PyYAML`.

## 4. What's new

- **65 models**, up from 26: 9 foundation models (Chronos, Moirai, Moment,
  TimesFM), 9 scikit-learn regressors, 6 modern deep time-series models
  (PatchTST, iTransformer, TSMixer, FreTS, DLinear), epidemic-specific deep
  models (EINN, EpiDeep, CALINet), statistical and nowcasting baselines
  (SeasonalNaive, RKI, NobBS), and compartmental wrappers.
  Foundation backends are optional, one extra each: `pip install epilearn[chronos]`
  (also `[moirai]`, `[moment]`, `[timesfm]`). Their upstream pins conflict, so
  install one per environment; all four need Python >= 3.10.
- **Two new tasks**: `NowcastTask` (reporting-delay correction) and
  `ScenarioTask` (counterfactual intervention scenarios, scored with PEHE/ATE).
- **Rolling-window evaluation with conformal prediction intervals** for every
  task, plus four conformal strategies in `epilearn.utils.uncertainty`
  (split, ACI, locally-weighted, locally-weighted ACI).
- **Per-fold Optuna tuning** built into `rolling_train` (`use_optuna=True`).
- **A config-driven benchmark**: `python -m epilearn.benchmark --config x.yaml`,
  or the `epilearn-benchmark` command. See [benchmark.md](./benchmark.md).
- **Ensembling**: `epilearn.strategies` provides 13 numpy-only aggregation
  strategies and `epilearn.ensemble` provides trainable stacking ensembles.
- **`epilearn.regime`** classifies a series into one of seven epidemic regimes.
