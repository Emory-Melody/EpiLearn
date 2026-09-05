Utilities
===================================
In this section, we introduce supplementary tools for analyzing epidemic data.
``epilearn.utils`` holds six modules — ``utils`` (tensor/graph helpers, smoothing
and decomposition blocks, ``significance_test``), ``metrics``, ``transforms``,
``uncertainty``, ``compartmental_models`` and ``simulation`` — all imported
eagerly, so a plain ``import epilearn`` is enough.

This page is task-oriented: it shows what each module is *for*. Every signature,
argument and return key is listed in :doc:`../API/utils`, and the simulators have
their own page, :doc:`simulation`.


1. Smoothing and decomposition
-------------------------------------

Epidemic series are noisy and strongly seasonal, so most deep models start by
splitting a window into a trend and a residual. Those blocks are plain
``nn.Module``\ s you can use directly. They live in ``epilearn.utils.utils`` (not
in ``transforms``, even though the old ``transforms.moving_avg`` path still
resolves), expect ``[batch, time, channels]`` — the layout a temporal model
receives — and ``series_decomp`` returns ``(residual, trend)``:

.. code-block:: python

    import torch
    from epilearn.data import Dataset
    from epilearn.utils import transforms
    from epilearn.utils.utils import moving_avg, series_decomp, series_decomp_multi

    toy = Dataset()
    toy.load_toy_dataset()
    series = toy.y[:, :4].T.unsqueeze(-1)        # 4 regions as a batch: [4, 539, 1]

    trend = moving_avg(kernel_size=25, stride=1)(series)
    residual, trend = series_decomp(kernel_size=25)(series)
    print(trend.shape, (residual + trend - series).abs().max().item())
    # torch.Size([4, 539, 1]) 0.0

    residual, trend = series_decomp_multi(kernel_size=[13, 25, 51])(series)
    print(residual.shape, trend.shape)           # both torch.Size([4, 539, 1])

    # this one takes [batch, nodes, time] and returns a LIST [seasonality, trend]
    decompose = transforms.seasonality_and_trend_decompose(decompose_type='dynamic',
                                                           kernel_size=[4, 8, 12])
    seasonality, trend = decompose(torch.rand(16, 10, 36))
    print(seasonality.shape, trend.shape)        # both torch.Size([16, 36, 10])

``moving_avg`` pads both ends, so the output keeps the input length and the
decomposition is exactly additive. ``series_decomp_multi`` mixes several kernel
widths with a learned softmax weighting, so it has parameters and must be trained
inside a model. ``transforms.seasonality_and_trend_decompose`` returns a *list*, so
it too belongs inside a ``forward`` rather than in a dataset-level ``Compose``,
whose entries must return something that can go back into ``dataset.x``; its
``decompose_type='static'`` swaps the Fourier seasonality model for a plain
``series_decomp``, while ``'dynamic'`` keeps the top-3 frequencies and so needs a
long enough window.


2. Losses and metrics
-------------------------------------

``get_loss`` resolves the string you pass as ``train_loss`` / ``val_loss``
(``'mse'``, ``'mae'``, ``'ce'``, ``'stan'``, ``'epi_cola'``) and returns any
callable unchanged, so a custom loss just gets handed straight to
``rolling_train``. ``get_metric`` and ``compute_metrics`` do the same for the
``report_metrics`` list, where a custom metric is any callable
``f(preds, targets) -> scalar`` and is keyed by its ``__name__``:

.. code-block:: python

    import torch
    from epilearn.utils.metrics import compute_metrics, METRIC_REGISTRY

    print(sorted(METRIC_REGISTRY))
    # ['acc', 'accuracy', 'mae', 'mape', 'mse', 'nrmse', 'r2', 'rmse']

    torch.manual_seed(0)
    preds = torch.rand(64, 3)
    targets = preds + 0.1 * torch.randn(64, 3)
    print({k: round(v, 4) for k, v in
           compute_metrics(preds, targets, ['mse', 'mae', 'rmse', 'mape', 'r2']).items()})
    # {'mse': 0.0091, 'mae': 0.0763, 'rmse': 0.0954, 'mape': 30.6287, 'r2': 0.8948}

    def smape(p, t):
        return (2 * (p - t).abs() / (p.abs() + t.abs() + 1e-8)).mean()

    print({k: round(v, 4) for k, v in compute_metrics(preds, targets, ['mae', smape]).items()})
    # {'mae': 0.0763, 'smape': 0.3097}

Passing ``report_metrics=['mae', smape]`` to ``rolling_train`` then puts
``smape_mean`` / ``smape_std`` in the aggregate.

``significance_test`` then says whether the gap between two models is real. It
consumes the bootstrap samples ``Detection.evaluate_model`` produces by default
(``bootstrap_maes`` / ``_mses`` / ``_rmses`` for regression,
``bootstrap_accuracys`` / ``_precisions`` / ``_recalls`` / ``_f1s`` for
classification) and returns both means with confidence intervals, the difference
and its interval, a p-value, ``is_significant`` and ``better_model``:

.. code-block:: python

    from epilearn.utils.utils import significance_test

    out = significance_test(eval_a, eval_b, metric='mae', alpha=0.05)
    print(out['difference'], out['p_value'], out['better_model'])

Finally, ``epilearn.utils.epidemic_metrics`` adds outbreak-aware scores
(``compute_outbreak_recall``, ``compute_alert_sensitivity``,
``compute_peak_underestimate_rate``, ``compute_rising_phase_mae``,
``compute_trend_accuracy``) for when average error hides the failures that matter
epidemiologically. It is the one ``utils`` submodule not imported eagerly, so
import it by name: ``from epilearn.utils import epidemic_metrics``.


3. Uncertainty quantification
-------------------------------------

*New in 0.1.0.* ``rolling_train`` already returns one split-conformal interval per
fold, which is often enough. ``epilearn.utils.uncertainty`` is for when a single
constant width is not: ``compute_aci`` adapts the width over time,
``locally_weighted_conformal`` scales it with the predicted level, and
``locally_weighted_aci`` does both (``static_conformal`` reproduces what
``rolling_train`` does). All four share the signature
``strategy(val_residuals, predictions, targets, target_alpha=0.1)``, return the
same dict (keys in :doc:`../API/utils`), and work on NumPy arrays in **original
units**. The recipe is to hold out a calibration window, score it, and hand its
absolute residuals to the strategy:

.. code-block:: python

    import numpy as np
    import torch
    from epilearn.data import Dataset
    from epilearn.utils import transforms
    from epilearn.tasks.forecast import Forecast
    from epilearn.models.SpatialTemporal.STGCN import STGCN
    from epilearn.utils.uncertainty import (static_conformal, compute_aci,
                                            locally_weighted_conformal,
                                            locally_weighted_aci,
                                            compute_uncertainty_metrics)

    torch.manual_seed(0)                      # so the numbers below reproduce
    lookback, horizon = 12, 3
    toy = Dataset(); toy.load_toy_dataset()
    dataset = Dataset(x=toy.x[:, :10], y=toy.y[:, :10], graph=toy.graph[:10, :10])
    dataset.set_transforms(transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target":   [transforms.normalize_target()],
        "graph":    [transforms.normalize_adj()]}), apply_now=True)
    history = dataset.get_process_history()

    def make_split(a, b):
        return dataset.generate_dataset(X=dataset.x[a:b], Y=dataset.y[a:b], adj=dataset.graph,
                                        lookback_window_size=lookback, horizon_size=horizon)

    train_split, val_split, test_split = make_split(0, 350), make_split(350, 430), make_split(430, 539)
    task = Forecast(prototype=STGCN, dataset=dataset,
                    lookback=lookback, horizon=horizon, device='cpu')
    task.train_model(train_split=train_split, val_split=val_split,
                     test_split=test_split, epochs=20, batch_size=32)

    # inverse_normalize=True: conformal works in original units
    cal = task.evaluate_model(dataset=val_split,  process_history=history, inverse_normalize=True)
    tst = task.evaluate_model(dataset=test_split, process_history=history, inverse_normalize=True)
    cal_pred, cal_targ = cal['predictions'].numpy(), cal['targets'].numpy()
    pred, targ = tst['predictions'].numpy(), tst['targets'].numpy()
    val_residuals = np.abs(cal_pred - cal_targ).ravel()

    for name, fn, kw in [("static", static_conformal, {}),
                         ("aci",    compute_aci, {}),
                         ("lw",     locally_weighted_conformal, {'val_predictions': cal_pred}),
                         ("lw_aci", locally_weighted_aci,       {'val_predictions': cal_pred})]:
        out = fn(val_residuals, pred, targ, target_alpha=0.1, **kw)
        marginal = compute_uncertainty_metrics(pred, targ, out['lower'], out['upper'],
                                              target_alpha=0.1)
        print(f"{name:7s} joint={out['coverage']:.3f} marginal={marginal['coverage']:.3f} "
              f"width={out['avg_width']:7.2f} winkler={out['winkler_score']:7.2f} "
              f"corr={marginal['width_abs_corr_r']:.3f}")

Output (the last digits move a little between runs)::

    static  joint=0.274 marginal=0.772 width=  72.49 winkler= 276.10 corr=nan
    aci     joint=0.737 marginal=0.980 width= 304.32 winkler= 313.90 corr=0.854
    lw      joint=0.232 marginal=0.899 width= 121.17 winkler= 194.80 corr=0.892
    lw_aci  joint=0.758 marginal=0.984 width= 318.02 winkler= 324.71 corr=0.907

Read that table with the Winkler score, not the coverage: ``lw`` wins (195 vs 276)
because it spends its width where the model is actually uncertain. ``corr``, the
correlation between an interval's half-width and that sample's absolute error, says
the same thing — ``nan`` for ``static_conformal``, whose width is constant by
construction, 0.89 for the locally weighted version.

The two coverage columns are different quantities: what these strategies return is
**joint** coverage (every horizon step of a sample inside its interval), while
``compute_uncertainty_metrics`` — and the per-fold ``coverage`` from
``rolling_train`` — is **marginal** (element-wise). Hence 0.274 vs 0.772 on the
``static`` row. ``compute_uncertainty_metrics`` also reports ``width_abs_corr_r`` /
``_rho``, ``mse_by_uncertainty_q`` (monotone 177 → 5769 here: the check that widths
are informative rather than merely wide) and ``per_horizon_coverage``, which is
filled only for 2-D ``(n_samples, horizon)`` predictions.

.. important::
   Pass ``val_predictions=`` to the two locally-weighted strategies. Without it
   they estimate the data scale from the calibration *residuals*, which
   over-covers badly — on the run above, dropping ``val_predictions`` inflates
   ``lw`` from width 121 to width 565 at joint coverage 0.968.

To re-calibrate a finished ``rolling_train`` without retraining, use
``fold['val_residuals']``: it is enough for ``static_conformal`` and
``compute_aci``. The locally-weighted strategies additionally need calibration
*predictions*, which is why the example above scores the validation split
explicitly.


4. Compartmental models
-------------------------------------

*New in 0.1.0.* ``epilearn.utils.compartmental_models`` provides deterministic ODE
models that need no data at all — useful as a baseline, as a data generator, or
to reason about an intervention before fitting anything: ``SIRModel``,
``SEIRModel`` (``sigma`` = 1/latent period), ``SIRSModel`` (``omega`` wanes
immunity R → S) and ``SEIRVIModel`` (adds vaccinated ``V`` and isolated ``Q``).
:doc:`../API/utils` lists their constructors.

Every model exposes ``compartments``, ``parameters``, ``step(...)`` for a single
integration step, and ``simulate(initial_state, steps, dt=1.0, method='rk4')``
returning ``{'time', 'trajectory', 'compartments'}`` with ``trajectory`` shaped
``(steps + 1, n_compartments)``. ``initial_state`` is in **counts** and its length
must equal ``len(model.compartments)``:

.. code-block:: python

    from epilearn.utils.compartmental_models import SIRModel

    sir = SIRModel(beta=0.4, gamma=0.1)
    print(sir.compartments, sir.parameters)
    # ('S', 'I', 'R') {'beta': 0.4, 'gamma': 0.1, 'mu': 0.0, 'birth_rate': 0.0}
    print("R0 =", sir.parameters['beta'] / sir.parameters['gamma'])     # R0 = 4.0

    run = sir.simulate([9990.0, 10.0, 0.0], steps=150)   # counts, not fractions
    I = sir.compartments.index('I')
    print(f"peak {run['trajectory'][:, I].max():.0f} on day {run['trajectory'][:, I].argmax()}")
    # peak 4036 on day 27
    print(f"final size {run['trajectory'][-1, 2] / run['trajectory'][0].sum() * 100:.1f}%")
    # final size 98.0%

    # a latent compartment delays the peak without changing R0 ...
    from epilearn.utils.compartmental_models import SEIRModel, SIRSModel

    seir = SEIRModel(beta=0.4, gamma=0.1, sigma=0.2)      # 5-day latent period
    r = seir.simulate([9990.0, 0.0, 10.0, 0.0], steps=150)
    print(int(r['trajectory'][:, seir.compartments.index('I')].argmax()))   # 57, vs 27 for SIR

    # ... and waning immunity turns the single wave into an endemic equilibrium
    sirs = SIRSModel(beta=0.4, gamma=0.1, omega=0.02)
    r = sirs.simulate([9990.0, 10.0, 0.0], steps=600)
    print(f"{r['trajectory'][-1, 1]:.0f} still infectious at t=600")        # 1250 (SIR: 0)

Interventions: rates vs inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two override channels, and picking the wrong one is the usual mistake.
``parameter_schedule`` overrides **rate parameters** (``beta``, ``gamma``, ...) —
use it for non-pharmaceutical interventions, seasonality or variant changes.
``input_schedule`` supplies **external inputs**: ``force_of_infection`` for any
model, plus ``vaccination_rate`` / ``isolation_rate`` for ``SEIRVIModel``. Both
accept a callable ``f(step_idx, t, state) -> dict | None``, a ``{step_idx: dict}``
mapping, or a sequence indexed by step, where ``None`` keeps the base values. So a
lockdown is a ``parameter_schedule`` (worked example in :doc:`simulation`), while
vaccination and isolation are inputs:

.. code-block:: python

    from epilearn.utils.compartmental_models import SEIRVIModel

    vi = SEIRVIModel(beta=0.6, gamma=0.2, sigma=0.3, vaccine_efficacy=0.9)
    print(vi.compartments)             # ('S', 'E', 'I', 'R', 'V', 'Q')
    init = [9990.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    I, V, Q = (vi.compartments.index(c) for c in 'IVQ')

    campaign = lambda step_idx, t, state: {'vaccination_rate': 0.02} if t >= 20 else None
    isolate  = lambda step_idx, t, state: {'isolation_rate': 0.15}   if t >= 20 else None

    print(f"{vi.simulate(init, steps=150)['trajectory'][:, I].max():.0f}")                    # 1764
    vax = vi.simulate(init, steps=150, input_schedule=campaign)
    print(f"{vax['trajectory'][:, I].max():.0f}, {vax['trajectory'][-1, V]:.0f} vaccinated")  # 1202, 2978
    iso = vi.simulate(init, steps=150, input_schedule=isolate)
    print(f"{iso['trajectory'][:, I].max():.0f}, peak Q {iso['trajectory'][:, Q].max():.0f}") # 609, 430

.. note::
   Each step is clamped at 0 by ``project_state``, so a too-large ``dt`` saturates
   silently instead of erroring; prefer ``method='euler'`` with a small ``dt`` for
   stiff or strongly forced systems, and the default ``'rk4'`` otherwise.

These models are the engine behind :doc:`simulation` (which adds noise, contact
graphs and mobility) and behind ``ScenarioTask``, whose intervention scenarios are
generated from a ``SEIRVIModel`` — see :doc:`task_building`.
