Quickstart for EpiLearn
==========================

Every task in EpiLearn is built the same way: load a :class:`~epilearn.data.dataset.Dataset`, attach the transformations, initialize a task with a model prototype, then train. The example below is complete and runnable -- it uses the built-in toy dataset, so you can paste it straight into a terminal.

.. note::

   In 0.1.0 ``UniversalDataset`` was renamed to ``Dataset`` (the old name still imports, as a deprecated alias), and ``train_model(dataset=..., train_rate=..., val_rate=...)`` was replaced by ``rolling_train``, which splits the series into rolling folds for you. If you are upgrading from 0.0.x, see ``MIGRATION.md`` in the repository.


Your first forecast
--------------------

``rolling_train`` walks a rolling origin over the series and, for every fold, trains the model, scores it on the held-out test block, and calibrates a conformal prediction interval on the validation block.

.. code-block:: python

    from epilearn.models.SpatialTemporal.STGCN import STGCN
    from epilearn.data import Dataset
    from epilearn.utils import transforms
    from epilearn.tasks.forecast import Forecast

    lookback = 12   # inputs size
    horizon = 3     # predicts size

    # load the toy dataset: 539 days x 47 regions, 4 features and a static graph
    dataset = Dataset()
    dataset.load_toy_dataset()

    # define and attach the transformations
    transformation = transforms.Compose({
                    "features": [transforms.normalize_feat()],
                    "target": [transforms.normalize_target()],
                    "graph": [transforms.normalize_adj()]})
    dataset.transforms = transformation

    # initialize the task
    task = Forecast(prototype=STGCN,
                    dataset=None,
                    lookback=lookback,
                    horizon=horizon,
                    device='cpu')

    # training: each rolling fold is trained, scored, and conformalized
    result = task.rolling_train(dataset=dataset,
                                train_size=400,
                                val_size=50,
                                test_size=50,
                                train_loss='mse',
                                epochs=50,
                                batch_size=5)

    # evaluation: rolling_train already scored every fold
    print(result['aggregate_metrics'])

    # inspect the most recent fold and plot it
    last_fold = result['fold_results'][-1]
    eval_results = task.evaluate_model(dataset=last_fold['test_split'])
    task.plot_preds(eval_results, region_idx=0, horizon_idx=-1, save_path='forecast.png')

``rolling_train`` reports each fold as it goes, then the aggregate:

.. code-block:: text

    ============================================================
    Rolling Evaluation Complete (2 folds)
    ============================================================
    MSE: 8.6531 ± 0.3623
    MAE: 1.0727 ± 0.3734
    RMSE: 2.9410 ± 0.0616

    Conformal (90% target):
      Quantile: 1.0973 ± 0.6956
      Coverage: 75.5% ± 0.8%
      Interval Width: 2.1947 ± 1.3913

    Total time: 204.81s, Memory: +192.14 MB

The same summary is returned as a dictionary (``result['aggregate_metrics']``: ``n_folds`` plus ``<metric>_mean``/``_std`` for ``mse``, ``mae``, ``rmse``, ``conformal_quantile``, ``coverage`` and ``interval_width``), so you can compare models programmatically. The full return value has seven entries: ``fold_results`` (per-fold metrics, conformal results and the test split itself), ``aggregate_metrics``, ``conformal_alpha``, ``all_predictions``, ``all_targets``, ``conformal_intervals`` and ``runtime``. Those seven are the success shape only: a fold that raises is caught and skipped, and if *every* fold fails ``rolling_train`` prints ``WARNING: No successful folds!`` and returns just ``{'fold_results': [], 'aggregate_metrics': None}`` -- ``result['aggregate_metrics']['mse_mean']`` then raises ``TypeError`` and ``result['all_predictions']`` raises ``KeyError``, so check ``result['aggregate_metrics'] is not None`` before indexing.

``plot_preds`` (called ``plot_forecasts`` before 0.1.0) draws one region and one horizon step of an evaluation result. For an interactive figure, pass ``backend='plotly', interactive=True`` and an ``.html`` save path instead; see :doc:`Installation` for the extra it needs.

.. note::

   Metrics are **not** denormalized automatically. Because ``normalize_target()`` is
   attached above, the MSE/MAE/RMSE and interval widths printed here are in
   normalized units. To get original units, hand the fold's normalization statistics
   back to ``evaluate_model``::

       eval_results = task.evaluate_model(dataset=last_fold['test_split'],
                                          process_history=last_fold['process_history'],
                                          inverse_normalize=True)

   The snippet above does not fix a random seed, so your numbers will differ
   slightly from the output shown.


Next steps
-----------

* :doc:`tutorials/task_building` builds this pipeline step by step and covers the other three tasks: detection (``train_model`` with splits you build yourself), nowcasting (reporting-delay correction) and scenario modeling.
* :doc:`Benchmark` runs many models under one rolling-window protocol from a YAML config, instead of a script.
* :doc:`API/dataset` documents loading your own data -- CSV, tensor/numpy, transforms and rolling splits.
* :doc:`API/tasks` documents every keyword of ``rolling_train``, ``train_model`` and ``evaluate_model``, including per-fold Optuna tuning (``use_optuna=True``).
