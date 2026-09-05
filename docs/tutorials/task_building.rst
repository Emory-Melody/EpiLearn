Pipeline for Epidemic Modeling
===================================


In this section, we will walk you through the building of epidemic modeling tasks using **Epilearn**, from dataset construction to model evaluation.
See :doc:`../Installation` and :doc:`../Quickstart` first.

.. note::
   **Coming from 0.0.x?** Training now goes through
   :ref:`rolling_train <rolling-train>`; ``train_model`` is the
   :ref:`lower-level single-split trainer <manual-splits>`. Every rename and
   behaviour change is listed in `MIGRATION.md
   <https://github.com/Emory-Melody/EpiLearn/blob/main/MIGRATION.md>`_ — this
   page does not repeat them.

1. Dataset construction
-------------------------

Everything starts from a :class:`~epilearn.data.dataset.Dataset`, which holds the series,
the optional graph and the metadata. There are four ways to build one:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Source
     - Call
   * - your own arrays
     - ``Dataset(x=..., y=..., graph=..., dynamic_graph=..., states=...,
       timestamps=..., regions=..., feature_names=...)``
   * - a built-in dataset
     - ``Dataset(name='Measles', root='./tmp/')`` — also ``'JHU_covid'``,
       ``'Tycho_v1'``, ``'Covid_<Country>'`` (see `Datasets
       <https://github.com/Emory-Melody/EpiLearn/tree/main/datasets>`_)
   * - a long-format CSV
     - ``Dataset.from_csv('cases.csv', timestamp_col='date', region_col='state',
       feature_cols=[...], target_cols=[...], graph_file='edges.csv')``
   * - a simulator
     - every function in :doc:`simulation` returns tensors that drop straight into
       ``Dataset(x=..., y=...)``

Only ``x`` (``[T, N, F]`` or ``[T, F]``) and ``y`` (``[T, N]`` or ``[T, 1]``) are
required. ``timestamps`` and ``regions`` default to ``range(T)`` / ``range(N)``
and are what the rolling-window splitter reports back as fold boundaries, so
passing real dates makes the fold log readable. :doc:`customization` covers the
attribute layout, ``from_tensor`` / ``from_numpy`` and ``save`` / ``load``.

The rest of this tutorial uses the small bundled toy dataset:

.. code-block:: python

    from epilearn.data import Dataset

    dataset = Dataset()
    dataset.load_toy_dataset()
    print(dataset)
    # Dataset(Spatiotemporal, x=(539, 47, 4), y=(539, 47), graph=(47, 47))


2. Transformations
----------------------

Epilearn provides numerous transformations — normalization, seasonal decomposition, conversion to the frequency domain, and more — composed in the same style as Pytorch:

.. code-block:: python

    from epilearn.utils import transforms

    transformation = transforms.Compose({
                                        "features": [transforms.normalize_feat()],
                                        "target": [transforms.normalize_target()],
                                        "graph": [transforms.normalize_adj()]})
    dataset.set_transforms(transformation)

``set_transforms`` only *registers* the pipeline: ``rolling_train`` then re-fits
the normalization constants on each fold's **training window** and applies them to
that fold's validation and test windows, which keeps the evaluation leak-free. Pass
``apply_now=True`` to transform the whole dataset immediately (for example before
building splits by hand), then read the fitted constants back:

.. code-block:: python

    dataset.set_transforms(transformation, apply_now=True)
    print(sorted(dataset.get_process_history()))
    # ['feat_mean', 'feat_std', 'target_mean', 'target_std']

.. warning::
   Because ``normalize_target()`` is in the pipeline, every metric and interval
   width below is in **normalized** units — nothing is denormalized
   automatically. :doc:`customization` shows how to invert it, and how to write
   your own transformation.


3. Choosing a model
----------------------

Models come in three families, distinguished by the input they take: Spatial
(``[Batch, Nodes, Channels]``), Temporal (``[Batch, Window, Channels]``) and
Spatial-Temporal (``[Batch, Window, Nodes, Channels]``). All 65 of them are listed
in :doc:`../API/models`; each is a class, imported from its family:

.. code-block:: python

    from epilearn.models.Temporal import GRUModel, LSTMModel, DlinearModel, ARIMAModel
    from epilearn.models.Spatial.GCN import GCN
    from epilearn.models.SpatialTemporal.STGCN import STGCN

The task fills in the shape-dependent constructor arguments for you —
``num_features``, ``num_timesteps_input``, ``num_timesteps_output``, ``device``,
plus ``num_nodes`` when a graph is present — from the data. Anything else
(hidden width, dropout, ...) is passed through ``model_args``.

.. important::
   A **temporal** model must not receive a graph. The task decides which
   constructor signature to use from ``split['graph']``, so if your ``Dataset``
   carries a graph, a temporal model is handed 4-D ``[B, L, N, F]`` input and
   fails with ``GRU: Expected input to be 2D or 3D, got 4D``. Build a graph-free
   view of the same series first:

   .. code-block:: python

       toy = Dataset(); toy.load_toy_dataset()
       dataset = Dataset(x=toy.x, y=toy.y,            # no graph=, no dynamic_graph=
                         timestamps=toy.timestamps, regions=toy.regions)

   Nodes are then folded into the batch dimension automatically.

To build a customized model, inherit from the ``BaseModel`` of the matching
family and define ``forward`` and ``initialize``; :doc:`customization` walks
through a complete example, including the shapes each family's ``forward``
receives.


4. Task Initialization
-------------------------------------------

Epilearn supports four tasks, all importable from ``epilearn.tasks``. They share
one model interface, so switching task means swapping the task class, not the
model: ``Forecast`` predicts the next ``horizon`` steps, ``Detection`` a per-node
class label, ``NowcastTask`` the final value of days whose reports are still
incomplete, and ``ScenarioTask`` one trajectory per intervention scenario.

A task is built from a model prototype plus the window geometry:

.. code-block:: python

    from epilearn.tasks.forecast import Forecast

    task = Forecast(prototype=STGCN,
                    dataset=None,      # optional; rolling_train takes the dataset
                    lookback=12,       # input window length
                    horizon=3,         # number of steps to predict
                    device='cpu')      # 'cuda' to train on GPU

``prototype`` is the model *class*, not an instance — the task re-instantiates it
for every fold. Pass ``model=`` instead if you already hold a built model.


.. _rolling-train:

5. Training and evaluation with ``rolling_train``
-------------------------------------------------

``rolling_train`` is the main entry point. For each fold it trains on the
training window, predicts on the validation window to calibrate a conformal
quantile, then predicts on the test window and applies that quantile — so you
get metrics *and* calibrated intervals in one call.

.. code-block:: python

    import torch
    from epilearn.data import Dataset
    from epilearn.utils import transforms
    from epilearn.tasks.forecast import Forecast
    from epilearn.models.SpatialTemporal.STGCN import STGCN

    torch.manual_seed(0)
    dataset = Dataset()
    dataset.load_toy_dataset()
    dataset.set_transforms(transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target":   [transforms.normalize_target()],
        "graph":    [transforms.normalize_adj()]}))

    task = Forecast(prototype=STGCN, lookback=12, horizon=3, device='cpu')
    result = task.rolling_train(dataset=dataset,
                                train_size=350,        # initial training window (timesteps)
                                val_size=60,           # calibration window
                                test_size=60,          # test window per fold
                                step_size=60,          # how far the origin moves each fold
                                expanding=True,        # training window grows; False = slides
                                train_loss='mse', epochs=20, batch_size=32,
                                conformal_alpha=0.1)   # 90% target coverage
    print(result['aggregate_metrics'])

Every fold is printed as it runs, then the aggregate (lines omitted where marked;
the last digits drift with BLAS threading)::

    --- Fold 1 ---   [...]
      Test MSE: 0.1858, MAE: 0.2202, RMSE: 0.4310
      Conformal quantile: 1.2406, Coverage: 96.9%, Interval Width: 2.4812
    --- Fold 2 ---   [...]
      Test MSE: 15.2154, MAE: 1.4149, RMSE: 3.9007
      Conformal quantile: 0.2815, Coverage: 47.9%, Interval Width: 0.5630

    MSE: 7.7006 ± 7.5148
    Coverage: 72.4% ± 24.5%

Fold 2 is eighty times worse than fold 1 with half the coverage — exactly the kind
of instability a single random split hides, and the reason rolling evaluation is
the default. Window sizes are **counts of timesteps**, not fractions, and
``val_size`` should not be 0: without a calibration window the conformal quantile
falls back to training residuals and comes out optimistic.

The result holds ``aggregate_metrics`` (``<metric>_mean`` / ``_std``),
``all_predictions`` / ``all_targets``, ``conformal_intervals``, ``runtime`` and
``fold_results`` — one dict per fold with that fold's metrics, ``coverage``,
``conformal_quantile``, ``prediction_lower`` / ``prediction_upper``,
``val_residuals``, ``test_split``, ``process_history`` and ``best_params`` (when
Optuna ran). :doc:`../API/tasks` documents every keyword, including
``report_metrics`` and ``max_folds``.


.. _manual-splits:

6. Controlling the split yourself
-------------------------------------------

When you want one specific split — a fixed hold-out period, a leaderboard
protocol, or just a fast debugging loop — build the sliding windows with
:meth:`~epilearn.data.dataset.Dataset.generate_dataset` and call the lower-level
``train_model`` / ``evaluate_model`` pair.

.. code-block:: python

    from epilearn.data import Dataset
    from epilearn.utils import transforms
    from epilearn.tasks.forecast import Forecast
    from epilearn.models.SpatialTemporal.STGCN import STGCN

    lookback, horizon = 12, 3

    dataset = Dataset()
    dataset.load_toy_dataset()
    # apply_now=True: we are splitting by hand, so transform up front
    dataset.set_transforms(transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target":   [transforms.normalize_target()],
        "graph":    [transforms.normalize_adj()]}), apply_now=True)
    history = dataset.get_process_history()

    def make_split(start, end):
        return dataset.generate_dataset(X=dataset.x[start:end], Y=dataset.y[start:end],
                                        adj=dataset.graph,          # <-- see warning
                                        lookback_window_size=lookback,
                                        horizon_size=horizon)

    train_split, val_split, test_split = make_split(0, 350), make_split(350, 430), make_split(430, 539)
    print(sorted(train_split), train_split['features'].shape, train_split['targets'].shape)
    # ['dynamic_graph', 'features', 'graph', 'states', 'targets']
    # torch.Size([336, 12, 47, 4]) torch.Size([336, 47, 3])

    task = Forecast(prototype=STGCN, dataset=dataset,
                    lookback=lookback, horizon=horizon, device='cpu')
    task.train_model(train_split=train_split, val_split=val_split, test_split=test_split,
                     train_loss='mse', val_loss='mse', epochs=20, batch_size=32, lr=1e-3)

    evaluation = task.evaluate_model(dataset=test_split, process_history=history,
                                     inverse_normalize=True)   # report in original units
    print(evaluation['mse'], evaluation['mae'], evaluation['rmse'])
    # 99593.5 83.6 315.6  <- original units (inverse_normalize=True); no seed, so it varies

``evaluate_model`` takes a split **dict** as ``dataset=`` — a bare
``task.evaluate_model()`` raises ``TypeError``. ``Forecast.evaluate_model``
returns ``mse``, ``mae``, ``rmse``, ``mape``, ``r2``, ``median_ae``, ``max_error``,
``residual_mean``, ``residual_std``, plus ``predictions``, ``targets`` and
``residuals``; feed that dict to ``task.plot_preds(evaluation, region_idx=0,
horizon_idx=-1, save_path='forecast.png')`` to plot one region and horizon step.

.. warning::
   ``generate_dataset`` does **not** read ``self.graph`` — you must pass
   ``adj=dataset.graph`` explicitly. Omit it and ``split['graph']`` is ``None``,
   at which point the task builds the *temporal* constructor signature and a
   graph model dies on a missing ``num_nodes``.

   ``train_model`` requires all three splits (``RuntimeError`` otherwise), and it
   catches training exceptions, prints them, and returns ``None`` — so check the
   return value before indexing it.


7. The other three tasks
-------------------------------------------

7.1 Detection
~~~~~~~~~~~~~~~~~~~~~~~~

Detection classifies each node, so ``horizon`` carries the **number of classes**
rather than a number of future steps. Build the splits with ``horizon_size=1``
and train through ``train_model`` with a cross-entropy loss.

.. code-block:: python

    import torch
    from epilearn.data import Dataset
    from epilearn.utils import transforms
    from epilearn.tasks.detection import Detection
    from epilearn.models.Spatial.GCN import GCN

    torch.manual_seed(0)
    dataset = Dataset()
    dataset.load_toy_dataset()
    dataset.y = (dataset.y > dataset.y.median()).long()      # per-node class labels
    dataset.set_transforms(transforms.Compose({"features": [], "graph": []}), apply_now=True)

    def make_split(start, end):
        return dataset.generate_dataset(X=dataset.x[start:end], Y=dataset.y[start:end],
                                        adj=dataset.graph,
                                        lookback_window_size=1, horizon_size=1)

    train_split, val_split, test_split = make_split(0, 300), make_split(300, 400), make_split(400, 539)

    task = Detection(prototype=GCN, dataset=dataset,
                     lookback=1, horizon=2, device='cpu')    # horizon == number of classes
    task.train_model(train_split=train_split, val_split=val_split, test_split=test_split,
                     train_loss='ce', val_loss='ce', epochs=50, batch_size=5)

    evaluation = task.evaluate_model(dataset=test_split)
    print(evaluation['accuracy'], evaluation['macro_f1'])
    # 0.8910 0.8123   (drifts by ~0.001 between runs even with the seed fixed)

``Detection.evaluate_model`` prints a full report and returns per-class
precision/recall/F1, mean confidence, and bootstrap confidence intervals
(``compute_bootstrap_ci=False`` turns the resampling off).

.. note::
   Use ``train_model`` — not ``rolling_train`` — for Detection. ``rolling_train``
   builds its windows with ``horizon_size=self.horizon``, which for Detection is
   the class count, so the target tensor comes out the wrong shape and every
   fold fails with a batch-size mismatch inside the cross-entropy loss.

7.2 Nowcasting
~~~~~~~~~~~~~~~~~~~~~~~~

Nowcasting corrects for reporting delay: recent counts are still incomplete, and
the task learns how much each day will be revised upward. ``NowcastTask``
consumes a *reporting triangle* — ``triangle[day, d]`` is the count known for
``day`` after delay ``d`` — and turns it into a ``Dataset`` where unobserved
cells are marked ``-1``.

.. code-block:: python

    import numpy as np
    import torch
    from epilearn.models.Temporal import GRUModel
    from epilearn.tasks.nowcast import NowcastTask

    torch.manual_seed(0)

    # build a reporting triangle: day t's cases trickle in over delays 1..9
    rng = np.random.default_rng(0)
    delays = np.arange(1, 10)
    final = np.round(100 + 60 * np.sin(np.arange(260) / 18) + rng.normal(0, 4, 260))
    share = np.diff(1 - np.exp(-np.r_[0, delays] / 2.5)); share /= share.sum()
    triangle = np.cumsum([rng.multinomial(int(c), share) for c in final], axis=1)

    task = NowcastTask(prototype=GRUModel,
                       lookback=14,          # days of history per sample
                       horizon=7,            # days still incomplete, to be nowcast
                       min_delay=1, max_delay=9, device='cpu')
    dataset = task.create_dataset(triangle, final, delays=delays)
    print(dataset.x.shape, dataset.y.shape)
    # torch.Size([239, 14, 1, 9]) torch.Size([239, 1, 7])

    result = task.rolling_train(dataset, train_size=140, val_size=40, test_size=40,
                                epochs=60, batch_size=32, lr=1e-2)
    print("nowcast MAE      :", result['aggregate_metrics']['mae_mean'])
    print("latest-report MAE:", task.compute_naive_baseline(dataset)['naive_mae'])
    # nowcast MAE      : 6.2384209632873535
    # latest-report MAE: 26.414824

``compute_naive_baseline`` scores "just trust the most recent report", the
baseline any nowcast has to beat. If your triangle is on disk,
``NowcastTask.load_triangle(path)`` reads an ``.npz`` with ``triangle``,
``final_counts``, ``delays`` and ``time_values``.

7.3 Scenario modeling
~~~~~~~~~~~~~~~~~~~~~~~~

``ScenarioTask`` asks a counterfactual question: given shared history, how do
trajectories differ under different intervention policies? Each sample holds
``n_scenarios`` trajectories, one of which is the baseline, and the model is
scored on the *effect* it predicts rather than on the level.

.. code-block:: python

    import torch
    from epilearn.models.Temporal import GRUModel
    from epilearn.tasks.scenario_modeling import ScenarioTask

    torch.manual_seed(0)
    task = ScenarioTask(prototype=GRUModel, lookback=30, horizon=14, n_scenarios=4,
                        target_compartment='I',    # measure the effect on infectious
                        baseline_scenario_idx=0, device='cpu')

    # simulate from the built-in SEIR-VI model (vaccination + isolation policies)
    dataset = task.generate_dataset(n_samples=400, population=1e6, seed=0)
    print(dataset.x.shape, dataset.y.shape, task.comp_model.compartments)
    # torch.Size([400, 30, 4, 10]) torch.Size([400, 14, 4, 6]) ('S','E','I','R','V','Q')

    result = task.rolling_train(dataset, train_size=200, val_size=60, test_size=60,
                                epochs=40, batch_size=32, lr=1e-2,
                                report_metrics=['pehe', 'ate_error'])
    print(result['aggregate_metrics'])
    # {'n_folds': 2, 'pehe_mean': 28841.5, 'ate_error_mean': 3484.9, ...}

    evaluation = task.evaluate_model(dataset=result['fold_results'][-1]['test_split'])
    print(evaluation['pehe'], evaluation['ate_error'], evaluation['tau_true'].shape)
    # 25861.9 3053.5 torch.Size([60, 3, 14])

The metrics are **PEHE** (``sqrt(E[(tau_pred - tau_true)^2])``) and **ATE error**
(``|E[tau_pred] - E[tau_true]|``), where ``tau`` is the difference between a
scenario and the baseline scenario. You must pass
``report_metrics=['pehe', 'ate_error']``: ``ScenarioTask`` overrides the per-fold
metric computation, so ``'mse'`` and friends are ignored and
``aggregate_metrics`` comes back with only the conformal numbers.
``plot_scenario_comparison(evaluation, sample_idx=0)`` draws one sample's
scenarios side by side, and ``compartmental_model=`` swaps in your own
:class:`~epilearn.utils.compartmental_models.SEIRVIModel`.


8. Hyperparameter tuning
-------------------------------------------

Set ``use_optuna=True`` and ``rolling_train`` tunes **per fold** — each fold
searches on its own training/validation windows, so the tuning respects the same
temporal ordering as the evaluation.

.. code-block:: python

    result = task.rolling_train(dataset=dataset,
                               train_size=300, val_size=60, test_size=60, step_size=120,
                               epochs=20, batch_size=32,
                               use_optuna=True, n_trials=5,
                               optuna_model_args={'nhids': [32, 64, 128],
                                                  'dropout': [0.0, 0.5]},
                               optimizer_params={'lr': [1e-4, 1e-2],
                                                 'batch_size': [16, 32, 64]})
    for fold in result['fold_results']:
        print(fold['fold'], fold['best_params'])

One observed run (the sampler is not seeded, so the values move between runs)::

    1 {'lr': 0.000817, 'batch_size': 16, 'model_nhids': 128, 'model_dropout': 0.0853}
    2 {'lr': 0.001593, 'batch_size': 64, 'model_nhids': 32,  'model_dropout': 0.0917}

The range encoding is positional, and it is easy to get wrong: a **2-element**
list is a continuous range ``[min, max]``, **3 or more** elements are categorical
choices, and a scalar is fixed. So ``'dropout': [0.0, 0.5]`` samples anywhere in
``[0, 0.5]`` while ``'nhids': [32, 64, 128]`` picks one of those three. Model
arguments come back under ``fold['best_params']`` with a ``model_`` prefix.
``optuna_model_args`` may also contain ``'lookback'``, in which case the splits
are re-generated with an alignment offset so every candidate lookback predicts
the same time points.

.. note::
   Unrecognized keys are swallowed by the model's ``**kwargs`` and silently do
   nothing. Check the constructor in :doc:`../API/models` first — ``GRUModel``
   takes ``nhids``, ``dropout`` and ``use_norm``, not ``hidden_dim``.


9. Prediction intervals
-------------------------------------------

Every ``rolling_train`` fold ships a split-conformal interval calibrated on that
fold's validation window; ``conformal_alpha=0.1`` targets 90% coverage.

.. code-block:: python

    for fold, ci in zip(result['fold_results'], result['conformal_intervals']):
        print(fold['fold'], ci['quantile'], fold['coverage'], fold['interval_width'])

For adaptive, locally-weighted or sequentially-updated intervals, feed
``fold['val_residuals']`` into ``epilearn.utils.uncertainty`` — see :doc:`utils`.

.. warning::
   Do not pass ``conformal_quantile=`` to ``Forecast.evaluate_model``: the helper it
   dispatched to was removed, so the call raises ``NotImplementedError`` pointing you
   at ``rolling_train`` and ``epilearn.utils.uncertainty``. Leave it at ``None``.


10. Comparing many models
-------------------------------------------

To run a whole model zoo under one protocol, describe the run in YAML instead of
writing a script:

.. code-block:: bash

    python -m epilearn.benchmark --config configs/quick_test_config.yaml

Per-model metrics, conformal coverage, Optuna trials and raw predictions are
written to ``benchmark_results/<task>/models_<timestamp>/``. The ``evaluation``
block mirrors the ``rolling_train`` keywords above; see :doc:`../Benchmark` for the
full config schema, the hyperparameter-range encoding, the output files and the
list of supported model names.
