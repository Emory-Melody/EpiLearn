Tasks
===================================

A *task* pairs a model prototype with an evaluation protocol: it slices the data
into windows, initializes the model from the shapes it finds, trains it, scores it
and calibrates prediction intervals. The model is a parameter of the task, so
moving a model between tasks means swapping the task class, not the model.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Task
     - Question it answers
     - Default metrics
   * - ``Forecast``
     - What do the next ``horizon`` steps look like?
     - MSE / MAE / RMSE
   * - ``NowcastTask``
     - The last few days are still being reported -- what will the final counts be?
     - MSE / MAE / RMSE
   * - ``ScenarioTask``
     - What would have happened under a different intervention?
     - PEHE / ATE error
   * - ``Detection``
     - Which nodes are the outbreak sources?
     - accuracy / macro-F1

All four are importable from ``epilearn.tasks``; the two new tasks also have
short aliases:

.. code-block:: python

    from epilearn.tasks import Forecast, Detection, NowcastTask, ScenarioTask
    from epilearn.tasks import Nowcast, Scenario   # aliases of the two above

This page is the reference: keywords, return shapes, and the trap each task carries.
The runnable scripts live elsewhere -- :doc:`../Quickstart` for a complete
forecasting pipeline, :doc:`../tutorials/task_building` for a step-by-step
walkthrough of all four tasks with the output each one prints.


Constructing a task
-----------------------------------

.. code-block:: python

    task = Forecast(prototype=STGCN, lookback=12, horizon=3, device='cpu')

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``prototype`` / ``model``
     - The model *class*, re-instantiated per fold with ``num_features``,
       ``num_timesteps_input``, ``num_timesteps_output``, ``device`` and (when a graph
       is present) ``num_nodes`` filled in from the data; anything else goes through
       ``model_args``. ``model=`` passes an already-built model instead, which is not
       re-initialized between folds.
   * - ``lookback`` / ``horizon`` / ``ahead``
     - Input window length, steps to predict, and the gap between the two (``0`` = the
       next step; ``ahead`` is on ``Forecast`` / ``Detection`` only). On ``Detection``,
       ``horizon`` is the **class count**, not a step count.
   * - ``dataset`` / ``device``
     - ``dataset`` is optional on ``Forecast`` / ``Detection`` and absent from
       ``NowcastTask`` / ``ScenarioTask``, which take it in ``rolling_train`` instead;
       ``device`` is ``'cpu'`` or ``'cuda'``. ``NowcastTask`` additionally takes
       ``min_delay`` / ``max_delay``, and ``ScenarioTask`` ``n_scenarios``,
       ``baseline_scenario_idx``, ``target_compartment`` and ``compartmental_model``.


Rolling-window training
-----------------------------------

``rolling_train`` is the primary entry point for every task, and it replaces the old
``train_model(dataset=..., train_rate=..., val_rate=...)`` call, which no longer
splits the data for you (see ``MIGRATION.md``). One call walks a rolling origin across
the series, training a fresh model per fold, calibrating a conformal quantile on that
fold's validation window and scoring its test window:

.. code-block:: text

    fold 1: |--- train ---|- val -|- test -|
    fold 2: |------ train ------|- val -|- test -|      (expanding=True)
    fold 3: |-------- train --------|- val -|- test -|

.. code-block:: python

    result = task.rolling_train(dataset=dataset, train_size=400, val_size=50,
                                test_size=50, epochs=50, batch_size=5)
    print(result['aggregate_metrics'])

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Argument
     - Meaning
   * - ``train_size`` / ``val_size`` / ``test_size``
     - Timesteps in the first training window, in the window held out for early
       stopping *and* conformal calibration, and scored per fold. Keep
       ``val_size > 0``: with ``0`` the quantile falls back to training residuals
       and the intervals come out optimistically narrow.
   * - ``step_size`` / ``expanding`` / ``max_folds``
     - How far the origin advances per fold (default ``test_size``, i.e.
       non-overlapping test windows); whether the training window grows
       (``True``, default) or slides at fixed width; and a cap on the number of
       folds (``None`` = every fold the series allows).
   * - ``conformal_alpha``
     - Miscoverage target, ``0.1`` (default) = 90% intervals.
   * - ``use_optuna`` / ``n_trials``
     - Tune hyperparameters per fold with Optuna. Search ranges come from
       ``optuna_model_args`` and ``optimizer_params``; put ``'lookback'`` in
       ``optuna_model_args`` to tune the input window too.
   * - ``report_metrics``
     - Metric names to compute, default ``['mse', 'mae', 'rmse']``. Accepts
       ``'mape'``, ``'r2'``, ``'acc'`` or your own callables. ``ScenarioTask``
       needs ``['pehe', 'ate_error']``.

The return value is a dictionary with seven entries: ``fold_results`` (per fold:
its metrics, ``conformal_quantile``, ``coverage``, ``interval_width``,
``val_residuals``, ``test_split``, ``process_history``, ``best_params``),
``aggregate_metrics`` (``n_folds`` plus ``<metric>_mean`` / ``<metric>_std``
across folds), ``conformal_alpha``, ``all_predictions``, ``all_targets``,
``conformal_intervals`` and ``runtime``.

.. warning::

   Seven entries only on success. A single failing fold is caught and reported, but
   if **every** fold fails the return value degrades to ``{'fold_results': [],
   'aggregate_metrics': None}``, so ``aggregate_metrics['mse_mean']`` raises
   ``TypeError`` and ``result['all_predictions']`` raises ``KeyError``.

.. note::

   Split-conformal intervals come for free -- there is no separate calibration
   step -- but the two coverage numbers EpiLearn reports are not the same
   quantity: the per-fold ``coverage`` here is **marginal** (fraction of
   individual ``(sample, region, horizon-step)`` entries inside their interval)
   while ``epilearn.utils.uncertainty.static_conformal()['coverage']`` is
   **joint** (fraction of samples whose *whole* horizon is inside the band), and
   joint coverage is always the smaller number.

.. note::

   Metrics are computed on whatever scale the model trained on: with
   ``transforms.normalize_target()`` attached they are in normalized units, and
   only ``evaluate_model(inverse_normalize=True)`` reports original units.

.. automethod:: epilearn.tasks.base.BaseTask.rolling_train


Forecast
-----------------------------------

Predicts the next ``horizon`` steps from the previous ``lookback`` steps; see
:doc:`../Quickstart` for the full script.

.. autoclass:: epilearn.tasks.forecast.Forecast
    :members:

``dataset.transforms`` is fitted on each fold's training window only and applied
from there to that fold's validation and test windows, so the statistics never leak
forward in time. Each fold returns its fitted constants as ``process_history``,
which is what ``evaluate_model`` needs to undo them:

.. code-block:: python

    last_fold = result['fold_results'][-1]
    evaluation = task.evaluate_model(dataset=last_fold['test_split'],        # split dict
                                     process_history=last_fold['process_history'],
                                     inverse_normalize=True)
    task.plot_preds(evaluation, region_idx=0, horizon_idx=-1, save_path='f.png')

``evaluate_model`` returns ``mse``, ``mae``, ``rmse``, ``mape``, ``r2``,
``median_ae``, ``max_error``, ``residual_mean``, ``residual_std`` and the raw
``predictions`` / ``targets`` / ``residuals`` tensors. Its ``dataset=`` is a split
**dict** (``features``, ``targets``, ``graph``, ``dynamic_graph``, ``states``) from
``Dataset.generate_dataset`` or ``fold_results[i]['test_split']``; hand it a
``Dataset``, or nothing at all, and it fails on the lookup instead.

.. warning::

   Do not pass ``conformal_quantile=`` to ``evaluate_model``: the helper it
   dispatched to was removed and the call now raises ``NotImplementedError``. Read
   the calibrated value from ``fold_results[i]['conformal_quantile']``, or apply
   the strategies in :doc:`utils` to ``fold_results[i]['val_residuals']``.


Detection
-----------------------------------

Classifies nodes -- typically outbreak source detection -- from a spatial
snapshot, so it pairs with the ``Spatial`` models.

.. warning::

   ``horizon`` here is the **number of classes**, not a number of future steps: it
   reaches the model as ``num_classes``, and the target window length is set
   separately by ``horizon_size=1`` in ``generate_dataset``. That also rules out
   ``rolling_train``, which builds windows with ``horizon_size=self.horizon``: the
   targets come out the wrong shape and every fold dies in the cross-entropy loss
   with ``ValueError: Expected input batch_size (...) to match target batch_size
   (...)``. Use the single-split ``train_model`` instead.

.. autoclass:: epilearn.tasks.detection.Detection
    :members:

.. code-block:: python

    task = Detection(prototype=GCN, dataset=dataset, lookback=1, horizon=2)  # 2 classes
    task.train_model(train_split=train_split, val_split=val_split,
                     test_split=test_split, train_loss='ce', val_loss='ce')
    evaluation = task.evaluate_model(dataset=test_split)   # a split dict, not a Dataset

Build those splits with ``generate_dataset(..., horizon_size=1)``, passing ``adj=``
or the graph reaching the model is silently ``None``
(:doc:`../tutorials/task_building` has the script); ``train_model`` needs all three
splits, and it catches training exceptions, prints them and returns ``None``.
``evaluate_model`` returns ``accuracy``, ``macro_precision``, ``macro_recall``,
``macro_f1``, per-class ``precision_per_class`` / ``recall_per_class`` /
``f1_per_class``, ``mean_confidence``, ``predictions`` / ``targets`` /
``probabilities``, and a bootstrap confidence interval per metric (turn the
resampling off with ``compute_bootstrap_ci=False``).


NowcastTask
-----------------------------------

Nowcasting corrects for reporting delay: recent counts are still incomplete, and the
task learns how much each day will be revised upward. The input is a **reporting
triangle** -- row ``t`` holds the counts known about day ``t`` after delays
``d = min_delay ... max_delay``, not-yet-observable entries marked ``-1``.
``lookback`` is how many past days of the triangle the model sees, ``horizon`` how
many recent incomplete days it nowcasts per sample, and
``n_delays = max_delay - min_delay + 1`` becomes the model's feature dimension.

.. autoclass:: epilearn.tasks.nowcast.NowcastTask
    :members: load_triangle, create_dataset, compute_naive_baseline

``create_dataset`` shapes features as ``(n_samples, lookback, 1, n_delays)`` and
targets as ``(n_samples, 1, horizon)``, ready for ``rolling_train``;
``load_triangle`` expects a ``.npz`` holding ``triangle``, ``final_counts``,
``delays`` and ``time_values``. ``compute_naive_baseline`` returns
``{'naive_mae': ..., 'n_samples': ...}`` for the "just trust the latest report"
baseline -- the number a nowcast has to beat -- computed over *every* sample in the
dataset rather than the rolling test windows, so read it as a reference level and
not a fold-matched comparison.

.. note::

   Neither ``NowcastTask`` nor ``BaseTask`` defines ``evaluate_model``, so
   ``task.evaluate_model(...)`` raises ``AttributeError: 'NowcastTask' object has no
   attribute 'evaluate_model'``. Score nowcasts from the ``rolling_train`` return
   value: ``aggregate_metrics``, ``fold_results[i]``, or ``all_predictions`` /
   ``all_targets`` for raw values.


ScenarioTask
-----------------------------------

Scenario modeling answers counterfactual questions: several intervention policies
share the same observed history, then diverge over the projection horizon. Each
sample holds ``n_scenarios`` futures, one of which (``baseline_scenario_idx``,
default ``0``) is the no-intervention control, and the task is scored on the
*difference* from that control, measured on ``target_compartment`` (``'I'`` by
default): **PEHE** (``sqrt(mean((tau_pred - tau_true)**2))``, error on the
per-sample treatment effect ``tau = Y_intervention - Y_baseline``) and **ATE
error** (``|mean(tau_pred) - mean(tau_true)|``, error on its average).

.. autoclass:: epilearn.tasks.scenario_modeling.ScenarioTask
    :members: generate_dataset, evaluate_model, plot_scenario_comparison

.. code-block:: python

    task = ScenarioTask(prototype=GRUModel, lookback=20, horizon=10,
                        n_scenarios=4, target_compartment='I', device='cpu')
    dataset = task.generate_dataset(n_samples=150, population=1e6, seed=42)
    result = task.rolling_train(dataset, train_size=60, val_size=45, test_size=45,
                                report_metrics=['pehe', 'ate_error'])

``ScenarioTask`` needs no data of your own: ``generate_dataset`` simulates
trajectories with the built-in ``SEIRVIModel``
(``epilearn.utils.compartmental_models``, replaceable via ``compartmental_model=``),
sampling vaccination and isolation rates/delays per scenario. Features come out as
``(n_samples, lookback, n_scenarios, N + 4)`` -- ``N`` compartments plus four
intervention descriptors -- and targets as ``(n_samples, horizon, n_scenarios, N)``.
Those "timesteps" are independent simulated samples, so ``train_size`` /
``val_size`` / ``test_size`` count samples rather than days, and both metrics carry
the units of the simulation (people out of ``population``). Passing a single split
dict to ``evaluate_model`` returns ``pehe``, ``ate_error``, ``mse``,
``predictions`` / ``targets`` and the treatment-effect tensors ``tau_pred`` /
``tau_true`` of shape ``(n_samples, n_scenarios - 1, horizon)``, which
``plot_scenario_comparison`` draws.

.. important::

   PEHE and ATE error are the only metrics ``ScenarioTask`` computes, and they are
   not in the default ``report_metrics``. Pass
   ``report_metrics=['pehe', 'ate_error']`` to ``rolling_train``; otherwise it
   goes looking for ``'mse'``, finds nothing, and ``aggregate_metrics`` comes back
   holding only the conformal numbers. The per-fold ``pehe`` / ``ate_error`` are
   in ``fold_results`` either way -- it is the aggregation and the printed summary
   that go missing.
