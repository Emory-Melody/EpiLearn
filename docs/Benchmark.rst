Benchmark
==========================

Comparing models fairly means giving each one the same data, the same rolling
windows, the same hyperparameter budget and the same uncertainty calibration.
EpiLearn 0.1.0 ships that protocol as a config-driven runner, so a comparison is
a YAML file instead of a script. Every model listed in the file is trained with
``rolling_train`` -- rolling-origin evaluation, per-fold Optuna tuning and
split-conformal prediction intervals -- and its metrics, tuning history and raw
predictions are written to CSV/NPZ files you can analyse later.

.. code-block:: bash

    # as a module
    python -m epilearn.benchmark --config configs/benchmark_config.yaml

    # as a console script (installed with the package)
    epilearn-benchmark --config configs/benchmark_config.yaml

    # short flags; -o overrides output.save_path from the config
    epilearn-benchmark -c configs/quick_test_config.yaml -o ./results/smoke_test/

``--config`` / ``-c`` is required and ``--output`` / ``-o`` is optional. Paths
inside the config are resolved relative to the current working directory, so run
the command from the repository root (or use absolute paths).

Ready-made configs live in ``configs/``: ``quick_test_config.yaml`` (a two-model
smoke test), ``benchmark_config.yaml`` (the full forecasting sweep),
``nowcast_benchmark_config.yaml``, ``scenario_benchmark_config.yaml`` and
``epi_models_config.yaml`` (EINN / EpiDeep / CALI-Net).


A minimal config
--------------------------

.. code-block:: yaml

    task: forecast

    dataset:
      feature_path: "./datasets/toy_features.csv"
      timestamp_col: "time"
      region_col: "node"
      feature_cols: ["f1", "f2", "f3", "y"]
      target_cols: ["y"]

    evaluation:
      lookback: 14
      horizon: 7
      train_size: 100
      val_size: 30
      test_size: 30
      max_folds: 1
      use_optuna: true
      n_trials: 5

    parallel:
      enable: false

    output:
      save_path: "./benchmark_results/"

    models:
      - name: GRUModel
        requires_graph: false
        optuna_model_args:
          nhids: [32, 64, 128]
          lookback: [7, 14, 21]
        optimizer_params:
          lr: [0.0001, 0.001]
          epochs: [50]

``feature_cols`` and ``target_cols`` are not optional in practice: the CSV loader
does not guess them, and leaving them out fails with
``'NoneType' object is not subscriptable``.

Mind the cost -- this config trains the model six times (five Optuna trials plus
the final fit), a couple of minutes on CPU. ``configs/quick_test_config.yaml``
runs a Ridge baseline in about a second and is the config to use when you only
want to check that the pipeline works.


The ``task`` block
--------------------------

A single scalar, which also decides what the ``dataset`` block means and which
metrics are reported:

* ``forecast`` (default) -- reads a long-format CSV, reports MSE / MAE / RMSE
* ``nowcast`` -- reads a reporting-triangle ``.npz``, reports MSE / MAE / RMSE
* ``scenario`` -- simulates SEIRVI data, reports PEHE / ATE error


The ``dataset`` block
--------------------------

For ``task: forecast``, the data is one long-format CSV (one row per region per
timestamp) plus an optional edge list:

* ``feature_path`` -- CSV with the features and targets. Required.
* ``graph_path`` -- edge-list CSV (``source,target``). Only read for models with
  ``requires_graph: true``.
* ``graph_weight_col`` -- optional edge-weight column in ``graph_path``.
* ``timestamp_col`` -- time column, default ``"time"``.
* ``region_col`` -- region/node column, default ``"node"``.
* ``feature_cols`` -- input columns, in order. **Set this.**
* ``target_cols`` -- target column(s). **Set this.**
* ``transforms`` -- ``normalize_features`` / ``normalize_target`` /
  ``normalize_graph``, booleans, each default ``true``.

All models receive the same transforms so their metrics are on the same scale.
Note that metrics are reported in *normalized* units when
``normalize_target: true`` -- the saved ``.npz`` keeps the per-fold mean and
standard deviation so you can invert it (see `Reading the results back`_).

For ``task: nowcast`` the block instead points at a reporting triangle:

.. code-block:: yaml

    dataset:
      triangle_path: "./data/my_reporting_triangle.npz"
      min_delay: 3      # first delay column to use (default 3)
      max_delay: 30     # last delay column; null = all delays

The ``.npz`` must contain the arrays ``triangle`` ``(n_days, n_delays)``,
``final_counts`` ``(n_days,)``, ``delays`` and ``time_values``; unobserved cells
are marked with ``-1``. No triangle ships with the repository, because the one the
shipped config names comes from the CMU Delphi Epidata API and is not
redistributed. Either regenerate it from the original source::

    python datasets/build_nowcast_triangle.py -o epilearn/data/nowcast_ready_data.npz

or point ``triangle_path`` at your own file. To try nowcasting with no external
data at all, ``tests/nowcast.py`` and the example in
:doc:`tutorials/task_building` build a synthetic triangle in a few lines.
Spatiotemporal models are skipped for this task. The runner also scores the "just
trust the latest report" baseline, but that number stays in memory and never
reaches the CSVs; call ``NowcastTask.compute_naive_baseline(dataset)['naive_mae']``
yourself if you want to report it.

For ``task: scenario`` the data is simulated, so the block describes the
simulator:

.. code-block:: yaml

    dataset:
      scenario:
        n_scenarios: 4            # number of intervention scenarios
        n_samples: 200            # timesteps to simulate
        population: 1000000
        process_noise: 0.01
        seed: 42
        baseline_scenario_idx: 0  # which scenario is the no-intervention control
        target_compartment: 'I'
        compartmental_model:      # SEIRVI parameters
          beta: 0.3
          gamma: 0.1
          sigma: 0.2
          vaccine_efficacy: 0.8
          isolation_efficacy: 0.9
      transforms:
        normalize_features: true
        normalize_target: true


The ``evaluation`` block
--------------------------

These settings are shared by every model, which is what makes the comparison
fair. The training window always expands (fold *k* trains on everything before
its validation window), and folds never overlap in test time.

Keys, with their defaults:

* ``lookback: 14`` -- input window length; an upper bound when ``lookback`` is also
  tuned.
* ``horizon: 7`` -- number of steps predicted.
* ``train_size: 100`` -- timesteps in the *first* training window; it expands on
  later folds.
* ``val_size: 30`` -- validation window, used for early stopping, Optuna and
  conformal calibration.
* ``test_size: 35`` -- test window.
* ``step_size: 30`` -- how far the origin advances per fold.
* ``max_folds: 3`` -- cap on the number of folds; ``null`` runs as many as the data
  allows.
* ``use_optuna: true`` -- tune per fold. When ``false``, ``optuna_model_args`` is
  ignored.
* ``n_trials: 5`` -- Optuna trials per fold; a model entry may override it.
* ``patience: 15`` -- early-stopping patience, in epochs.
* ``conformal_alpha: 0.1`` -- miscoverage level; ``0.1`` calibrates 90% intervals.
* ``seed: 42`` -- global seed. Parallel workers derive a deterministic per-model
  seed from it, so a parallel run is reproducible.
* ``verbose: false`` -- per-epoch training logs.


The ``parallel`` and ``output`` blocks
----------------------------------------

.. code-block:: yaml

    parallel:
      enable: true          # false = one model at a time in this process
      gpu_ids: null         # null = auto-detect every visible GPU; or [0, 1]
      workers_per_gpu: 2    # processes per GPU

    output:
      save_path: "./benchmark_results/forecast/"

Models are the unit of parallelism: each one runs in its own spawned process with
``CUDA`` pinned to its assigned device, and there is a one-hour timeout per
model. With no GPU available the runner falls back to sequential CPU execution
regardless of ``enable``. ``--output``/``-o`` on the command line overrides
``save_path``.


The ``models`` block
--------------------------

A list; each entry configures one benchmark row.

* ``name`` -- row label, and the model class unless ``class_name`` is given.
* ``class_name`` -- the class to instantiate. Use it to benchmark one class twice
  under different labels (e.g. ``name: ARIMA_RKI``, ``class_name: ARIMAModel``).
* ``requires_graph`` -- ``true`` for spatiotemporal models; controls whether
  ``graph_path`` is loaded and whether the model is scored jointly over all regions.
* ``model_args`` -- fixed constructor arguments, not tuned.
* ``optuna_model_args`` -- search space for constructor arguments (see below).
* ``optimizer_params`` -- search space for training arguments: ``lr``, ``epochs``,
  ``batch_size``, ``weight_decay``, ``train_loss``, ``val_loss``.
* ``n_trials`` -- per-model override of ``evaluation.n_trials``.


Hyperparameter search ranges
------------------------------

``optuna_model_args`` and ``optimizer_params`` use the same compact encoding,
where the *length* of the list decides how the value is interpreted:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - written as
     - interpreted as
   * - ``alpha: 1.0``
     - a fixed value (a bare scalar is never tuned)
   * - ``epochs: [150]``
     - a fixed value, recorded in the Optuna history
   * - ``dropout: [0.1, 0.3]``
     - a **range** ``[min, max]``: floats become ``suggest_float``, two ints
       become ``suggest_int``
   * - ``lr: [0.0001, 0.001]``
     - a range sampled log-uniformly (``lr``, or any range spanning more than
       two orders of magnitude)
   * - ``nhids: [32, 64, 128]``
     - **categorical** choices -- three or more entries are never a range
   * - ``mode: ["add", "mul"]``
     - categorical (two entries, but not numeric)

The consequence worth remembering: ``[32, 64]`` means *any integer from 32 to
64*, while ``[32, 64, 128]`` means *one of these three*. Write a third value (or
a one-element list per value) when you mean discrete choices.

.. code-block:: yaml

    models:
      - name: STGCN
        requires_graph: true
        optuna_model_args:
          nhids: [16, 32, 64]       # categorical
          lookback: [8, 16, 32]     # window length, tuned per fold
        optimizer_params:
          lr: [0.0001, 0.001]       # log-uniform range
          batch_size: [32]          # fixed
          epochs: [150]

``lookback`` is special. Placed in ``optuna_model_args``, it tunes the input
window itself: the runner builds the splits with the largest candidate so that
every trial and every model sees the same target timestamps, then re-slices the
window for the chosen value. ``evaluation.lookback`` is the fallback when a model
does not tune it.


Skipped models
--------------------------

The runner survives a partial environment and partial failures: results are
appended as each model finishes, and a model that cannot run becomes a row with a
``status`` other than ``success`` instead of aborting the benchmark.

* ``skipped_dependency`` -- a foundation model whose backend is not installed.
  The nine foundation entries (Chronos, Moirai, Moment, TimesFM) need
  the matching extra (e.g. ``pip install epilearn[chronos]``); the check happens before any training,
  and the ``error`` column carries the exact pip hint. Running
  ``configs/quick_test_config.yaml`` in a default install therefore reports
  ``Success: 1/2``, with ``ChronosModel`` skipped for a missing ``chronos``.
* ``error`` -- the model raised. The traceback is kept in the ``error`` column.

In ``task: nowcast`` runs, which are single-region by construction, spatiotemporal
models are handled differently again: they are dropped from the job list with a
warning and never appear in the results at all.


Output files
--------------------------

Every run is stamped with ``<ts>`` = ``YYYYmmdd_HHMMSS`` and writes into
``output.save_path``:

.. code-block:: text

    benchmark_results/
      benchmark_summary_<ts>.csv      one row per model: aggregate metrics
      benchmark_detailed_<ts>.csv     one row per model per fold
      optuna_trials_<ts>.csv          one row per trial per region
      models_<ts>/
        <Model>_summary.csv           that model's row from the summary
        <Model>_detailed.csv          that model's per-fold rows
        <Model>_optuna.csv            that model's tuning history
        <Model>_predictions.npz       raw predictions and calibration data

``benchmark_summary_<ts>.csv`` has one row per model with ``status``,
``evaluation_paradigm`` (``node-independent``, ``graph-joint`` or ``zero-shot``),
mean and standard deviation across folds for ``mse``/``mae``/``rmse`` (or
``pehe``/``ate_error`` for scenarios), the conformal ``coverage_mean`` and
``interval_width_mean``, ``runtime_seconds`` and ``error``. A model that was
skipped or that failed only gets ``<Model>_summary.csv``, with the reason in
``error``.

``benchmark_detailed_<ts>.csv`` has one row per fold, with that fold's metrics,
its conformal coverage and interval width, and the ``best_params`` that fold
chose -- this is the file to read when you want to know whether a model's average
hides one catastrophic fold, and it is where the tuned hyperparameters live (the
``best_params`` column of the summary file is a placeholder and stays ``{}``).

``<Model>_predictions.npz`` keeps everything needed to recompute metrics without
retraining, per fold ``i``:

* ``fold_i_predictions``, ``fold_i_targets`` -- matched arrays; for
  forecasting/nowcasting ``(n_windows * n_regions, horizon)``, i.e. the region axis
  is folded into the first one
* ``fold_i_inputs`` -- the lookback window behind each prediction,
  ``(n_windows * n_regions, lookback, n_features)``
* ``fold_i_target_mean``/``_std``, ``fold_i_feat_mean``/``_std`` -- normalization
  stats, for inverting ``normalize_target`` / ``normalize_features``
* ``fold_i_val_residuals``, ``fold_i_conformal_quantile`` -- calibration residuals
  and the conformal quantile, for recalibrating intervals post hoc (see
  ``epilearn.utils.uncertainty``)
* ``n_folds``, ``conformal_alpha`` -- scalars describing the run


Reading the results back
------------------------------

Two helpers in ``epilearn.benchmark`` read those artefacts, so a new metric does
not require a new benchmark run. Both default to the most recent ``models_<ts>``
directory when ``timestamp`` is omitted.

.. code-block:: python

    from epilearn.benchmark import load_predictions, evaluate_from_saved

    # raw arrays for one model
    saved = load_predictions('./benchmark_results/quick_test/', 'RidgeModel')
    print(saved['n_folds'], saved['folds'][0]['predictions'].shape)

    # re-score every model in the run with different metrics
    print(evaluate_from_saved('./benchmark_results/quick_test/',
                              metrics=['mae', 'nrmse']))

``evaluate_from_saved`` accepts the built-in names ``mse``, ``mae``, ``rmse``,
``mape``, ``r2``, ``nrmse`` and the epidemic-specific ``outbreak_recall``,
``alert_sensitivity``, ``peak_underestimate``, ``rising_phase_mae``,
``trend_accuracy``, as well as your own ``f(pred, target)`` or
``f(pred, target, inputs)`` callables. Pass ``denormalize=True`` to score in the
original units using the saved normalization stats.


Model names
--------------------------

``name`` (or ``class_name``) must be one of the classes the runner knows about.

**Temporal and single-series models** (``requires_graph: false``) --
deep: ``GRUModel``, ``LSTMModel``, ``CNNModel``, ``MLPModel``, ``DlinearModel``,
``PatchTSTModel``, ``iTransformerModel``, ``TSMixerModel``, ``FreTSModel``;
epidemic-specific: ``EINNModel``, ``EpiDeepModel``, ``CALINetModel``,
``SIRModel``, ``SEIRModel``;
scikit-learn: ``LinearRegressionModel``, ``RidgeModel``, ``LassoModel``,
``ElasticNetModel``, ``RandomForestModel``, ``GradientBoostingModel``,
``SVRModel``, ``KNNModel``, ``DecisionTreeModel``;
statistical and nowcasting baselines: ``ARIMAModel``, ``VARMAXModel``,
``SeasonalNaiveModel``, ``RKINowcastModel``, ``NobBSModel``.

**Foundation models** (zero-shot; each needs its own extra, e.g. ``epilearn[chronos]``) --
``ChronosModel``, ``ChronosBoltModel``, ``MoiraiModel``, ``MoiraiBaseModel``,
``MoiraiLargeModel``, ``MomentModel``, ``MomentSmallModel``, ``MomentBaseModel``,
``TimesFMModel``.

**Spatiotemporal models** (``requires_graph: true``) -- ``STGCN``, ``DSTGCN``,
``DCRNN``, ``GraphWaveNet``, ``EpiGNN``, ``ColaGNN``, ``MepoGNN``, ``ATMGNN``.

A name outside these sets raises ``ValueError: Unknown model: <name>``. Adding a
model of your own means implementing the class, exporting it from its family's
``__init__.py``, and adding its name to the matching registry set at the top of
``epilearn/benchmark.py``.
