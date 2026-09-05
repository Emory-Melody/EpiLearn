Utils
===================================

``epilearn.utils`` groups the framework's helper code into six modules:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Contents
   * - ``epilearn.utils.utils``
     - Tensor/graph helpers, plus the ``moving_avg`` / ``series_decomp`` blocks used by
       the deep models.
   * - ``epilearn.utils.metrics``
     - Loss functions and error metrics (MSE/MAE/RMSE/ACC).
   * - ``epilearn.utils.transforms``
     - Composable preprocessing applied to a :class:`~epilearn.data.dataset.Dataset`.
   * - ``epilearn.utils.uncertainty``
     - Conformal prediction strategies and interval-quality metrics. **New in 0.1.0.**
   * - ``epilearn.utils.compartmental_models``
     - SIR / SEIR / SIRS / SEIR-VI ODE models with time-varying parameters.
       **New in 0.1.0.**
   * - ``epilearn.utils.simulation``
     - Graph generators and the temporal / individual / regional epidemic simulators.

Every module is imported eagerly, so ``from epilearn.utils import uncertainty`` (or
``from epilearn import utils; utils.simulation...``) works after a plain
``import epilearn``.

Utility_Functions
===================================
Accuracy
----------
.. autofunction:: epilearn.utils.utils.accuracy

Normalize
----------
.. autofunction:: epilearn.utils.utils.normalize

Normalize_Adj
-------------
.. autofunction:: epilearn.utils.utils.normalize_adj

Diff
----------
.. autofunction:: epilearn.utils.utils.diff

Degree_Matrix
-------------
.. autofunction:: epilearn.utils.utils.Degree_Matrix

Static_Full
-----------
.. autofunction:: epilearn.utils.utils.Static_full

Kronecker
----------
.. autofunction:: epilearn.utils.utils.kronecker

Edge_to_Adj
-----------
.. autofunction:: epilearn.utils.utils.edge_to_adj

Moving_Avg
----------
.. note::
   ``moving_avg``, ``series_decomp`` and ``series_decomp_multi`` live in
   ``epilearn.utils.utils``; ``epilearn.utils.transforms.moving_avg`` still resolves
   (``transforms`` does ``from .utils import *``) but is not the canonical import.

.. autoclass:: epilearn.utils.utils.moving_avg
    :members:

Series_Decomp
-------------
.. autoclass:: epilearn.utils.utils.series_decomp
    :members:

Series_Decomp_Multi
-------------------
.. autoclass:: epilearn.utils.utils.series_decomp_multi
    :members:

Metrics
===================================
MSE_loss
----------
.. autofunction:: epilearn.utils.metrics.get_loss

Stan_loss
----------
.. autofunction:: epilearn.utils.metrics.stan_loss

Epi_cola_loss
-------------
.. autofunction:: epilearn.utils.metrics.epi_cola_loss

Cross_entropy_loss
------------------
.. autofunction:: epilearn.utils.metrics.cross_entropy_loss

MAE
----------
.. autofunction:: epilearn.utils.metrics.get_MAE

RMSE
----------
.. autofunction:: epilearn.utils.metrics.get_RMSE

ACC
----------
.. autofunction:: epilearn.utils.metrics.get_ACC

Uncertainty
===================================

*New in 0.1.0.* ``epilearn.utils.uncertainty`` turns point forecasts into calibrated
prediction intervals. Four conformal strategies are provided, all sharing one
signature and one return schema::

    strategy(val_residuals, predictions, targets, target_alpha=0.1)

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Strategy
     - Interval width
   * - ``static_conformal``
     - Constant (split conformal).
   * - ``compute_aci``
     - Varies over time; ``alpha_t`` adapts sequentially (Gibbs & Candès, 2021).
   * - ``locally_weighted_conformal``
     - Per-sample, scales with prediction magnitude.
   * - ``locally_weighted_aci``
     - Per-sample **and** time-adaptive.

Note the argument order: ``val_residuals`` (1-D absolute residuals from the
calibration split) comes **first**, and the miscoverage keyword is ``target_alpha``,
not ``alpha``. All four share one return schema — ``lower``, ``upper``, ``coverage``,
``avg_width``, ``winkler_score`` plus the ``alpha_trace`` / ``quantile_trace`` /
``coverage_per_sample`` diagnostics — or ``None`` when ``val_residuals`` is empty.

.. note::
   The returned ``coverage`` is **joint** — a test sample counts as covered only if
   *every* horizon step is inside its interval — while ``task.rolling_train`` and
   :func:`~epilearn.utils.uncertainty.compute_uncertainty_metrics` report
   **marginal** (element-wise) coverage. The two are not comparable.

Inputs are NumPy arrays in **original (denormalized) scale**, and the locally-weighted
pair wants ``val_predictions=`` (calibration-set predictions) or falls back to a
simplified, over-covering procedure. :doc:`../tutorials/utils` works both points
through with numbers and compares the four strategies on a trained model.

Static_Conformal
----------------
.. autofunction:: epilearn.utils.uncertainty.static_conformal

Compute_ACI
-----------
.. autofunction:: epilearn.utils.uncertainty.compute_aci

Locally_Weighted_Conformal
--------------------------
.. autofunction:: epilearn.utils.uncertainty.locally_weighted_conformal

Locally_Weighted_ACI
--------------------
.. autofunction:: epilearn.utils.uncertainty.locally_weighted_aci

Winkler_Score
-------------
.. autofunction:: epilearn.utils.uncertainty.winkler_score

Compute_Uncertainty_Metrics
---------------------------
.. autofunction:: epilearn.utils.uncertainty.compute_uncertainty_metrics

Difficulty_From_Predictions
---------------------------
.. autofunction:: epilearn.utils.uncertainty.difficulty_from_predictions

Difficulty_Reference_Stats
--------------------------
.. autofunction:: epilearn.utils.uncertainty.difficulty_reference_stats

Evaluate_ACI_From_Saved
-----------------------
.. autofunction:: epilearn.utils.uncertainty.evaluate_aci_from_saved

Compartmental_Models
===================================

*New in 0.1.0.* ``epilearn.utils.compartmental_models`` provides deterministic ODE
models that need no data at all: ``SIRModel`` / ``SIRSModel`` over ``('S', 'I', 'R')``
(``SIRSModel`` adds ``omega``, waning immunity ``R`` → ``S``), ``SEIRModel`` over
``('S', 'E', 'I', 'R')`` (``sigma`` = 1/latent period) and ``SEIRVIModel`` over
``('S', 'E', 'I', 'R', 'V', 'Q')`` (``V`` vaccinated, ``Q`` isolated). Constructors
are in the class signatures below; every model exposes

* ``compartments`` — the compartment names, in state-vector order;
* ``parameters`` — the rate dictionary;
* ``step(state, t, dt, method='rk4'|'euler', external_inputs=None,
  parameter_overrides=None)`` — one integration step;
* ``simulate(initial_state, steps, dt=1.0, method='rk4', parameter_schedule=None,
  input_schedule=None)`` → ``{'time', 'trajectory', 'compartments'}`` where
  ``trajectory`` has shape ``(steps + 1, n_compartments)``.

``initial_state`` is in **counts, not fractions**, and its length must equal
``len(model.compartments)`` or ``validate_state`` raises ``ValueError``;
``project_state`` clamps every compartment at 0 after each step.

``parameter_schedule`` overrides *rate parameters* (``beta``, ``gamma``, ...) while
``input_schedule`` supplies *external inputs* (``force_of_infection``, and for
``SEIRVIModel`` also ``vaccination_rate`` / ``isolation_rate``). Both accept a
callable ``f(step_idx, t, state) -> dict | None``, a ``{step_idx: dict}`` mapping,
or a sequence indexed by step; ``None`` leaves the base values in place.
:doc:`../tutorials/utils` explains which channel an intervention belongs to (with
worked lockdown and vaccination examples, and when to prefer ``method='euler'``);
:doc:`../tutorials/simulation` drives these models at population, individual and
regional level.

CompartmentalModel
------------------
.. autoclass:: epilearn.utils.compartmental_models.CompartmentalModel
    :members:

SIRModel
----------
.. autoclass:: epilearn.utils.compartmental_models.SIRModel
    :members:

SEIRModel
----------
.. autoclass:: epilearn.utils.compartmental_models.SEIRModel
    :members:

SIRSModel
----------
.. autoclass:: epilearn.utils.compartmental_models.SIRSModel
    :members:

SEIRVIModel
-----------
.. autoclass:: epilearn.utils.compartmental_models.SEIRVIModel
    :members:

Simulation
===================================

.. deprecated:: 0.1.0
   ``epilearn.utils.simulation.Time_geo`` **has been removed** with no replacement.
   Use :func:`~epilearn.utils.simulation.simulate_spatiotemporal_individual` for
   individual-level dynamics on a contact graph, or
   :func:`~epilearn.utils.simulation.simulate_spatiotemporal_regions` for
   metapopulation dynamics with mobility-driven flows.

0.1.0 exposes three simulators, all driven by a
:class:`~epilearn.utils.compartmental_models.CompartmentalModel` and all returning a
dict of tensors whose keys are listed per function below:

* ``simulate_temporal_epidemic`` — one population; ``trajectory``
  ``(steps+1, n_compartments)``. It is ``model.simulate()`` plus optional
  ``process_noise`` (scalar or per-compartment, with ``seed``), added after each
  step and re-projected to be non-negative.
* ``simulate_spatiotemporal_individual`` — individuals on a contact graph;
  ``trajectory`` ``(steps+1, n_nodes)`` of compartment indices.
* ``simulate_spatiotemporal_regions`` — regions with mobility-driven flows;
  ``trajectory`` ``(steps+1, n_regions, n_compartments)``, signed net flows in
  ``dynamic_graph`` (``directed_flow`` is the non-negative form) and per-region
  :math:`R_t = (\beta_t / \gamma)\,(S / N)` in ``effective_reproduction_number``.

Both spatiotemporal simulators take their graph as a static adjacency matrix /
NetworkX graph, a ``(steps+1, N, N)`` tensor or a callable
``f(t, step_idx) -> adjacency``, and both return a ``node_features`` tensor already
shaped for a ``SpatialTemporal`` model; initial conditions come from
``create_initial_conditions_individual`` / ``create_initial_conditions_region``, with
``create_regional_forcing_params`` for per-region seasonal forcing.

:doc:`../tutorials/simulation` is where these are taught: it runs all three end to
end, feeds the output into a :class:`~epilearn.data.dataset.Dataset`, explains the
``Gravity_model`` connection-strength convention and which flow tensor to hand a
model, and documents the two traps in the helpers above
(``initial_compartment_fractions`` is renormalized to sum to 1;
``n_initial_infected`` must stay below ``n_regions``).

Simulate_Temporal_Epidemic
--------------------------
.. autofunction:: epilearn.utils.simulation.simulate_temporal_epidemic

Simulate_Spatiotemporal_Individual
----------------------------------
.. autofunction:: epilearn.utils.simulation.simulate_spatiotemporal_individual

Simulate_Spatiotemporal_Regions
-------------------------------
.. autofunction:: epilearn.utils.simulation.simulate_spatiotemporal_regions

Create_Initial_Conditions_Individual
------------------------------------
.. autofunction:: epilearn.utils.simulation.create_initial_conditions_individual

Create_Initial_Conditions_Region
--------------------------------
.. autofunction:: epilearn.utils.simulation.create_initial_conditions_region

Create_Regional_Forcing_Params
------------------------------
.. autofunction:: epilearn.utils.simulation.create_regional_forcing_params

Gravity_Model
-------------
.. autoclass:: epilearn.utils.simulation.Gravity_model
    :members:

Get_Random_Graph
----------------
.. autofunction:: epilearn.utils.simulation.get_random_graph

Get_Graph_From_Features
-----------------------
.. autofunction:: epilearn.utils.simulation.get_graph_from_features

Transformation
===================================

.. note::
   ``Compose.__call__`` returns a **tuple** ``(data, process_history)`` in 0.1.0, so
   ``data = transformation(data)`` silently binds the tuple. ``process_history``
   holds the fitted normalization constants — ``feat_mean`` / ``feat_std`` and
   ``target_mean`` / ``target_std`` — replacing the old ``Compose.feat_mean`` /
   ``Compose.feat_std`` attributes; it is also readable as
   ``transformation.process_history`` after the call.

.. code-block:: python

    from epilearn.data import Dataset
    from epilearn.utils import transforms

    dataset = Dataset(); dataset.load_toy_dataset()
    transformation = transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target": [transforms.normalize_target()],
        "graph": [transforms.normalize_adj()]})
    data, process_history = transformation({"features": dataset.x,
                                           "target": dataset.y,
                                           "graph": dataset.graph})
    print(sorted(process_history))
    # ['feat_mean', 'feat_std', 'target_mean', 'target_std']

Usually the ``Compose`` is handed to the dataset instead
(``dataset.set_transforms(transformation, apply_now=True)``) and applied per fold by
the task. Keep ``target_mean`` / ``target_std``: without them metrics and interval
widths stay in normalized units (:doc:`../tutorials/customization`).

Compose
----------
.. autoclass:: epilearn.utils.transforms.Compose
    :members:

Normalize_Feat
--------------
.. autoclass:: epilearn.utils.transforms.normalize_feat
    :members:

Normalize_Target
----------------
.. autoclass:: epilearn.utils.transforms.normalize_target
    :members:

Normalize_Adj
-------------
.. autoclass:: epilearn.utils.transforms.normalize_adj
    :members:

Convert_To_Frequency
--------------------
.. autoclass:: epilearn.utils.transforms.convert_to_frequency
    :members:

Add_Time_Embedding
------------------
.. autoclass:: epilearn.utils.transforms.add_time_embedding
    :members:

Learnable_Time_Embedding
------------------------
.. autoclass:: epilearn.utils.transforms.learnable_time_embedding
    :members:

Seasonality_And_Trend_Decompose
-------------------------------
.. autoclass:: epilearn.utils.transforms.seasonality_and_trend_decompose
    :members:

Calculate_DTW_Matrix
--------------------
.. autoclass:: epilearn.utils.transforms.calculate_dtw_matrix
    :members:
