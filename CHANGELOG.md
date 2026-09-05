# Changelog

## 0.1.0

A major update. Upgrading from 0.0.x? See [MIGRATION.md](./MIGRATION.md) — two APIs
changed, everything else is source-compatible.

### Added

- **Nowcasting** (`epilearn.tasks.nowcast.NowcastTask`): corrects for reporting
  delay from a reporting triangle, with a latest-report naive baseline for
  comparison.
- **Scenario modeling** (`epilearn.tasks.scenario_modeling.ScenarioTask`):
  counterfactual intervention scenarios scored with PEHE / ATE error. Needs no
  user data — it synthesises from `SEIRVIModel`.
- **Rolling-window evaluation** (`BaseTask.rolling_train`) for every task type:
  walk-forward folds, expanding or sliding, with per-fold split-conformal
  prediction intervals.
- **Per-fold Optuna tuning** inside `rolling_train` (`use_optuna=True`,
  `n_trials=`, `optuna_model_args=`, `optimizer_params=`), including tuning the
  lookback window.
- **Config-driven benchmark**: `python -m epilearn.benchmark --config x.yaml`, or
  the new `epilearn-benchmark` console script. Writes per-model metrics,
  conformal coverage, Optuna trials and raw predictions. See
  [benchmark.md](./benchmark.md).
- **Uncertainty quantification** (`epilearn.utils.uncertainty`): split conformal,
  adaptive conformal inference (ACI), locally-weighted conformal, locally-weighted
  ACI, Winkler score, and coverage/width metrics.
- **39 new models**, taking the zoo from 26 to 65 exported classes:
  - Foundation models: Chronos, Chronos-Bolt, Moirai (base/large), Moment
    (small/base), TimesFM. Optional, one extra per backend —
    `pip install epilearn[chronos]`, `[moirai]`, `[moment]`, `[timesfm]`.
  - Deep time series: PatchTST, iTransformer, TSMixer, FreTS, MLP.
  - Epidemic-specific: EINN, EpiDeep, CALINet.
  - Statistical / nowcasting baselines: SeasonalNaive, RKI, NobBS.
  - scikit-learn regressors: LinearRegression, Ridge, Lasso, ElasticNet,
    RandomForest, GradientBoosting, SVR, KNN, DecisionTree.
  - Compartmental wrappers: SIRModel, SISModel, SEIRModel.
  - Spatiotemporal: STGCN_c, DSTGCN.
- **Ensembling**: `epilearn.strategies` (13 numpy-only aggregation strategies) and
  `epilearn.ensemble` (trainable stacking ensembles).
- **Epidemic regime classification**: `epilearn.regime` labels a series as one of
  seven regimes (`REGIME_LABELS`).
- **Compartmental simulators**: `epilearn.utils.compartmental_models` with
  SIRModel / SEIRModel / SIRSModel / SEIRVIModel, supporting time-varying
  parameters via a callable `parameter_schedule`.
- **Data layer**: `Dataset.from_csv` for long-format CSVs, a loader registry
  (CSV / tensor / numpy), timestamp and region indexing, and
  `Dataset.rolling_splits` / `get_slice`.
- New simulation helpers: `simulate_temporal_epidemic`,
  `simulate_spatiotemporal_individual`, `simulate_spatiotemporal_regions`.
- `pyproject.toml` packaging, replacing `setup.py`, with a `plot` extra and one
  extra per foundation backend (`chronos`, `moirai`, `moment`, `timesfm`).

### Changed

- **`UniversalDataset` is now `Dataset`.** The old name remains as a deprecated
  alias; the constructor signature is a strict superset of the old one.
- **`train_model` no longer splits the data.** It is now the low-level
  single-split trainer taking `train_split` / `val_split` / `test_split` dicts;
  `rolling_train` is the main entry point. The kwargs `dataset=`, `loss=`,
  `train_rate=`, `val_rate=`, `permute_dataset=`, `config=` and `region_idx=` are
  gone; `loss=` split into `train_loss=` / `val_loss=`.
- `evaluate_model` takes a split dict via `dataset=`, and returns more metrics
  (mape, r2, median_ae, max_error, residual stats) plus conformal intervals.
- `transforms.Compose.__call__` returns `(data, process_history)`.
- `Dataset.generate_dataset` returns a dict instead of a tuple, and its `permute`
  flag inverted meaning.
- Metrics are no longer auto-denormalized; pass `inverse_normalize=True` to
  `evaluate_model` for original units.
- `torch_geometric` is now a required dependency. `optuna`, `psutil`, `pandas`
  and `PyYAML` are new required dependencies. `streamlit`, `pyvis` and `xgboost`
  were dropped — the library never imported them.
- `Forecast.plot_forecasts` → `Forecast.plot_preds` (with an optional plotly
  backend).
- Renamed attributes: `Dataset.features` → `feature_names`,
  `Dataset.timestamp` → `timestamps`; `index` / `anual_population` /
  `coordinates` moved under `Dataset.metadata`.
- `Dataset.from_csv` keyword names: `feature_csv` → `file_path`,
  `node_id_col` → `region_col`, `time_col` → `timestamp_col`,
  `edge_csv` → `graph_file`.
- Install from source is now `pip install .` (`setup.py` was replaced by
  `pyproject.toml`).

### Removed

- The Streamlit web interface (`interface/`) and its bundled JS assets. It was a
  standalone app that never imported `epilearn`, and the hosted deployment was no
  longer reachable.
- The two Colab tutorial notebooks. The documentation and the `examples/` and
  `tests/` folders are now the single source of runnable tutorials.
- `Dataset(name='Covid_Japan')`. The `'Japan'` entry in the COVID archive was a
  copy of the 47-region toy dataset under different key names, not Japanese data.
  Unsupported country names now raise `ValueError` listing the real ones.
- The bundled nowcasting reporting triangle
  (`epilearn/data/nowcast_ready_data.npz`). It is CMU Delphi Epidata and is not
  redistributed; regenerate it from the original source with
  `python datasets/build_nowcast_triangle.py`.
- `epilearn.tasks.projection` and `epilearn.tasks.surveillance` (both were empty
  files) — superseded by `nowcast` and `scenario_modeling`.
- `Dataset.get_transformed()`, `ganerate_splits()`, `get_splits()` and
  `BaseTask.cv_train`.
- `epilearn.utils.simulation.Time_geo`.
- `epilearn.models.Temporal.XGB` and `epilearn.models.SpatialTemporal.HierST`
  (both were entirely commented out and never functional).
- The unused `epilearn.data.base.Dataset` stub, which shadowed the real
  `Dataset` name.

### Fixed

- **Installation failed under `uv` and other strict resolvers.** 0.0.x pinned
  `matplotlib==3.9.1`, which upstream later yanked, making every `epilearn>=0.0.19`
  install unsatisfiable. All nine exact `==` pins are now lower bounds, so the
  package resolves against current matplotlib/numpy/networkx. Verified with
  `uv pip install`: 0.1.0 resolves and runs on matplotlib 3.11, numpy 2.4 and
  torch 2.14.
- `Detection.evaluate_model` read `dataset['target']` while splits carry
  `'targets'`, raising `KeyError`.
- `Dataset.from_csv` squeezed the target axis for single-column targets, so a
  single time series (no `region_col`) crashed in `generate_dataset` with
  `IndexError`.
- `configs/quick_test_config.yaml` listed only an optional foundation model, so
  the smoke test "succeeded" while running zero models.
- `Detection.__init__` mis-passed positional arguments to `BaseTask`, so
  `device` was silently always `cpu`.
