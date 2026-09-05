Models
===================================

EpiLearn 0.1.0 ships **65 models**, from linear regression to pretrained time-series
foundation models. They are grouped into three families by the kind of input they
consume:

* ``epilearn.models.Temporal`` -- one series at a time (no graph).
* ``epilearn.models.Spatial`` -- one graph snapshot at a time (used by ``Detection``).
* ``epilearn.models.SpatialTemporal`` -- a series *and* a graph.

One interface for every model
-----------------------------------

Whatever the family, a model is constructed from the same keywords::

    num_features, num_timesteps_input, num_timesteps_output, device

plus ``num_nodes`` for the ``SpatialTemporal`` family (the ``Spatial`` family takes
``num_features``, ``num_classes``, ``device``). Everything model-specific --- hidden
size, dropout, number of layers, a Hugging Face checkpoint name --- is an extra
keyword with a default.

Because of that, you rarely construct a model yourself. You hand the **class** to a
task as ``prototype=``, and the task reads the data shape and builds the model for
you. Swapping architectures is then a one-line change, even across families:

.. code-block:: python

    import torch
    from epilearn.data import Dataset
    from epilearn.models.Temporal import GRUModel, RidgeModel     # deep net, scikit-learn
    from epilearn.models.SpatialTemporal import STGCN             # graph model
    from epilearn.tasks.forecast import Forecast
    from epilearn.utils import transforms

    lookback, horizon = 12, 3
    toy = Dataset()
    toy.load_toy_dataset()

    for prototype, needs_graph in [(GRUModel, False), (RidgeModel, False), (STGCN, True)]:
        torch.manual_seed(42)
        # graph models see the adjacency; purely temporal models do not
        data = Dataset(x=toy.x.clone(), y=toy.y.clone(),
                       graph=toy.graph if needs_graph else None,
                       timestamps=toy.timestamps)
        data.set_transforms(transforms.Compose(
            {"features": [transforms.normalize_feat()],
             "target": [transforms.normalize_target()]}))
        task = Forecast(prototype=prototype, lookback=lookback, horizon=horizon, device='cpu')
        result = task.rolling_train(data, train_size=300, val_size=60, test_size=60,
                                    max_folds=1, train_loss='mse', epochs=10, batch_size=64)
        print(f"{prototype.__name__:12s} RMSE={result['aggregate_metrics']['rmse_mean']:.4f} "
              f"coverage={result['aggregate_metrics']['coverage_mean']:.2f}")

Three families, one loop body::

    GRUModel     RMSE=0.6171 coverage=0.73
    RidgeModel   RMSE=0.5807 coverage=0.72
    STGCN        RMSE=1.2556 coverage=0.73

(One fold and 10 epochs: a smoke test of the interface, not a benchmark. For real
comparisons use the :doc:`benchmark <../Benchmark>`, which runs the same protocol
over many models with more folds and ``use_optuna=True``.)

The other tasks work the same way: ``NowcastTask`` and ``ScenarioTask`` take a
``prototype=`` too (``NowcastTask`` is single-region, so give it a ``Temporal``
model), and ``Detection`` takes a ``Spatial`` model.

Models that are *not* drop-in prototypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A handful of classes predate the shared interface and do not accept the keyword set
a task builds. Passing them as ``prototype=`` raises ``TypeError``:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - Use instead / how to use
   * - ``Temporal.Compartmental.SIR`` / ``SIS`` / ``SEIR``,
       ``NetworkSIR`` / ``NetworkSIS`` / ``NetworkSEIR``
     - Forward-simulation modules with no ``fit()``. For fitting an SIR/SIS/SEIR
       curve inside a task, use ``SIRModel`` / ``SISModel`` / ``SEIRModel`` from
       ``Temporal.CompartmentalModel``.
   * - ``SpatialTemporal.DMP``, ``SpatialTemporal.NetSIR``
     - Mechanistic simulators (``num_nodes``, ``horizon``, rate parameters). Call
       them directly; they have no ``fit()``.
   * - ``Spatial.GAT``, ``Spatial.SAGE``, ``Spatial.GIN``
     - They reject the ``num_nodes`` keyword that ``Detection`` passes. Use
       ``Spatial.GCN`` as a ``prototype=``, or construct GAT/SAGE/GIN yourself and
       call ``.fit()`` (see the ``Spatial`` code example below).

Two further classes are exported but return ``{"mean": ..., "std": ...}`` instead of a
single tensor, so they do not satisfy the task contract:
``Temporal.GRU_u.GRUModel`` (exported as ``GRU_u_Model``) and
``SpatialTemporal.DSTGCN_u.DSTGCN`` (exported as ``DSTGCN_u``). They are
heteroscedastic variants of ``GRUModel`` and ``DSTGCN`` that also predict a
per-step standard deviation; use them directly, or use ``rolling_train``'s conformal
intervals for calibrated uncertainty instead.


Temporal Models
===================================

Deep time-series models
-----------------------------------

Modern sequence architectures. All of them are pure PyTorch and train from scratch.

.. autoclass:: epilearn.models.Temporal.GRU.GRUModel
    :members:

.. autoclass:: epilearn.models.Temporal.LSTM.LSTMModel
    :members:

.. autoclass:: epilearn.models.Temporal.CNN.CNNModel
    :members:

.. autoclass:: epilearn.models.Temporal.MLP.MLPModel
    :members:

.. autoclass:: epilearn.models.Temporal.Dlinear.DlinearModel
    :members:

.. autoclass:: epilearn.models.Temporal.PatchTST.PatchTSTModel
    :members:

.. autoclass:: epilearn.models.Temporal.iTransformer.iTransformerModel
    :members:

.. autoclass:: epilearn.models.Temporal.TSMixer.TSMixerModel
    :members:

.. autoclass:: epilearn.models.Temporal.FreTS.FreTSModel
    :members:


Epidemic-specific deep models
-----------------------------------

Deep models whose architecture encodes epidemiological structure. ``EINNModel``
(Rodriguez et al., AAAI 2023) is a GRU encoder/decoder over SEIR compartments with a
learnable SEIR ODE as a soft regularizer; ``EpiDeepModel`` (Adhikari et al., KDD 2019)
clusters historical seasons with a dual autoencoder and attends over them;
``CALINetModel`` (Kamarthi et al., AAAI 2022) distils a pretrained ``EpiDeepModel``
into a lightweight target module.

.. autoclass:: epilearn.models.Temporal.EINN.EINNModel
    :members:

.. autoclass:: epilearn.models.Temporal.EpiDeep.EpiDeepModel
    :members:

.. autoclass:: epilearn.models.Temporal.CALINet.CALINetModel
    :members:


Foundation models
-----------------------------------

Pretrained time-series foundation models, used zero-shot by default. Each class is a
thin wrapper that tokenises the lookback window and calls the pretrained backend. The
``Base`` / ``Large`` / ``Small`` classes subclass the plain one and only pin a
different checkpoint (e.g. ``MoiraiBaseModel`` is ``MoiraiModel`` with
``model_name='Salesforce/moirai-1.0-R-base'``); ``ChronosBoltModel`` is the
encoder-only, faster Chronos variant.

.. note::

   The backends are **optional dependencies**::

       pip install epilearn[chronos]     # also [moirai], [moment], [timesfm]

   They are imported lazily, inside the wrapper's ``_load_model()``. Constructing the
   model therefore always works; the first ``forward()`` raises ``ImportError`` with
   the exact pip command if the backend is missing, e.g.
   ``chronos-forecasting is required for ChronosModel. Install it with: pip install
   chronos-forecasting``. The benchmark checks this up front
   (``epilearn.benchmark.check_foundation_deps``) and reports such models as skipped
   rather than failing the run.

   Backend per family: ``Chronos*`` needs ``chronos-forecasting``, ``Moirai*`` needs
   ``uni2ts``, ``Moment*`` needs ``momentfm``, ``TimesFM`` needs ``timesfm``.

.. autoclass:: epilearn.models.Temporal.Chronos.ChronosModel
    :members:

.. autoclass:: epilearn.models.Temporal.Chronos.ChronosBoltModel
    :members:

.. autoclass:: epilearn.models.Temporal.Moirai.MoiraiModel
    :members:

.. autoclass:: epilearn.models.Temporal.Moirai.MoiraiBaseModel
    :members:

.. autoclass:: epilearn.models.Temporal.Moirai.MoiraiLargeModel
    :members:

.. autoclass:: epilearn.models.Temporal.Moment.MomentModel
    :members:

.. autoclass:: epilearn.models.Temporal.Moment.MomentSmallModel
    :members:

.. autoclass:: epilearn.models.Temporal.Moment.MomentBaseModel
    :members:

.. autoclass:: epilearn.models.Temporal.TimesFM.TimesFMModel
    :members:


Statistical models and baselines
-----------------------------------

Classical statistics and the reference baselines every serious comparison needs.
``SeasonalNaiveModel`` repeats the value from the same point in the previous cycle --
hard to beat on strongly seasonal data.

``RKINowcastModel`` and ``NobBSModel`` are **nowcasting** baselines and expect a
reporting triangle, so use them with ``NowcastTask``; inside ``Forecast`` they raise
``NotImplementedError: Subclasses must implement _forecast_single_series``.
``RKINowcastModel`` (An der Heiden & Hamouda, 2020) learns a completion CDF
:math:`F(d)` during ``fit`` and divides each partial count by it; ``NobBSModel``
(McGough et al., 2020) estimates the delay distribution per sample and smooths the
log-incidence curve, so it needs no training data at all.

.. autoclass:: epilearn.models.Temporal.StatsModel.ARIMAModel
    :members:

.. autoclass:: epilearn.models.Temporal.StatsModel.VARMAXModel
    :members:

.. autoclass:: epilearn.models.Temporal.StatsModel.SeasonalNaiveModel
    :members:

.. autoclass:: epilearn.models.Temporal.StatsModel.RKINowcastModel
    :members:

.. autoclass:: epilearn.models.Temporal.StatsModel.NobBSModel
    :members:

.. note::

   ``epilearn.models.Temporal.ARIMA`` still imports ``VARMAXModel`` and
   ``ARIMAModel`` for backwards compatibility, but the module moved to
   ``epilearn.models.Temporal.StatsModel`` in 0.1.0. Prefer the new path.


scikit-learn regressors
-----------------------------------

Each wrapper flattens the lookback window into a feature vector and fits one
scikit-learn regressor per horizon step. They need no GPU and are the cheapest
sanity check available.

.. autoclass:: epilearn.models.Temporal.ScikitModel.LinearRegressionModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.RidgeModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.LassoModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.ElasticNetModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.RandomForestModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.GradientBoostingModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.SVRModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.KNNModel
    :members:

.. autoclass:: epilearn.models.Temporal.ScikitModel.DecisionTreeModel
    :members:


Compartmental models
-----------------------------------

These follow the same per-sample fitting interface as the statistical models: ``fit()``
is a no-op, and for every sample ``predict()`` integrates the ODE with
``scipy.integrate.odeint``, fits the rate parameters to that lookback window with
L-BFGS-B, and simulates ``horizon`` steps forward. They are ordinary ``prototype=``
models.

.. autoclass:: epilearn.models.Temporal.CompartmentalModel.SIRModel
    :members:

.. autoclass:: epilearn.models.Temporal.CompartmentalModel.SISModel
    :members:

.. autoclass:: epilearn.models.Temporal.CompartmentalModel.SEIRModel
    :members:

For richer mechanistic simulation (SEIRS, vaccination, waning immunity, time-varying
parameter schedules), see ``epilearn.utils.compartmental_models``.

Compartmental simulation modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``epilearn.models.Temporal.Compartmental`` holds the original ``nn.Module``
compartmental blocks. They hold their rates in ``nn.Linear`` weights and roll an
initial compartment vector forward one step at a time. They have **no** ``fit()`` and
do not accept ``num_features`` etc., so they cannot be passed to a task -- call them
directly:

.. code-block:: python

    import torch
    from epilearn.models.Temporal.Compartmental import SIR

    model = SIR(horizon=5, infection_rate=0.3, recovery_rate=0.1, population=1000)
    trajectory = model(torch.tensor([990., 10., 0.]))   # initial S, I, R
    print(trajectory.shape)                             # torch.Size([5, 3])

.. autoclass:: epilearn.models.Temporal.Compartmental.SIR
    :members:

.. autoclass:: epilearn.models.Temporal.Compartmental.SIS
    :members:

.. autoclass:: epilearn.models.Temporal.Compartmental.SEIR
    :members:

.. autoclass:: epilearn.models.Temporal.Compartmental.NetworkSIR
    :members:

.. autoclass:: epilearn.models.Temporal.Compartmental.NetworkSIS
    :members:

.. autoclass:: epilearn.models.Temporal.Compartmental.NetworkSEIR
    :members:

.. note::

   ``epilearn.models.Temporal.SIR`` still imports ``SIR``, ``SIS`` and ``SEIR`` for
   backwards compatibility, but the module moved to
   ``epilearn.models.Temporal.Compartmental`` in 0.1.0. Prefer the new path.


Code Example
-------------

Every model can also be driven directly, without a task, through ``.fit()``:

.. code-block:: python

    import torch
    from epilearn.models.Temporal.GRU import GRUModel

    num_features = 1
    lookback = 16 # inputs size
    horizon = 3 # predicts size

    features = torch.round(torch.rand((10, lookback, num_features)))
    node_target = torch.round(torch.rand((10, horizon)))

    model=GRUModel(num_features=num_features, num_timesteps_input=lookback, num_timesteps_output=horizon, device='cpu')
    model.fit(
            train_input=features,
            train_target=node_target,
            val_input=None,
            val_target=None,
            val_graph=None,
            epochs=20,
            loss='mse'
            )


Spatial Models
===================================

Graph neural networks that classify nodes from a single snapshot; the ``Detection``
task uses them for outbreak detection. They take ``num_features``, ``num_classes``
and ``device`` instead of a lookback/horizon pair.

.. autoclass:: epilearn.models.Spatial.GCN.GCN
    :members:

.. autoclass:: epilearn.models.Spatial.GAT.GAT
    :members:

.. autoclass:: epilearn.models.Spatial.GIN.GIN
    :members:

.. autoclass:: epilearn.models.Spatial.SAGE.SAGE
    :members:

.. note::

   Only ``GCN`` accepts the extra ``num_nodes`` keyword that ``Detection`` passes when
   it builds the model. ``GAT``, ``SAGE`` and ``GIN`` currently have to be constructed
   and fitted directly, as below.

Code Example
-------------

.. code-block:: python

    import torch
    from epilearn.models.Spatial import GCN

    num_features = 4
    num_classes = 2
    lookback = 1 # inputs size
    horizon = 2 # predicts size

    graph = torch.round(torch.rand((47,47)))
    features = torch.round(torch.rand((10,47,1,4)))
    node_target = torch.round(torch.rand((10,47)))

    model=GCN(num_features=num_features, num_classes=horizon, device='cpu')
    model.fit(
            train_input=features,
            train_target=node_target,
            train_graph=graph,
            val_input=None,
            val_target=None,
            val_graph=None,
            epochs=20,
            loss='ce'
            )


Spatial-Temporal Models
===================================

Models that consume a window of features *and* a graph, and predict a horizon for
every node jointly. All of them take
``num_nodes, num_features, num_timesteps_input, num_timesteps_output, device``, and
their ``forward`` accepts ``(X, adj, states, dynamic_adj)`` so a model can use node
states and a time-varying graph if the dataset provides them.

Spatio-temporal graph networks
-----------------------------------

General-purpose architectures that combine temporal convolution or recurrence with
graph propagation: ``STGCN`` (spatio-temporal graph convolution), ``DSTGCN``
(``STGCN`` plus a ``GraphLearningLayer`` that learns the adjacency instead of taking
it as given), ``DCRNN`` (diffusion convolutional recurrent network),
``GraphWaveNet`` (dilated temporal convolution with an optional adaptive adjacency)
and ``ATMGNN`` (attention-based temporal multiresolution GNN).

.. autoclass:: epilearn.models.SpatialTemporal.STGCN.STGCN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.DSTGCN.DSTGCN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.DCRNN.DCRNN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.GraphWaveNet.GraphWaveNet
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.ATMGNN.ATMGNN
    :members:


Epidemic-specific graph models
-----------------------------------

Architectures published for multi-region epidemic forecasting: ``ColaGNN``
(cross-location attention graph network), its epidemiological variant
``EpiColaGNN``, ``EpiGNN`` (epidemiological graph network) and ``CNNRNN_Res``
(convolution + recurrence with residual connections).

.. autoclass:: epilearn.models.SpatialTemporal.ColaGNN.ColaGNN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.EpiColaGNN.EpiColaGNN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.EpiGNN.EpiGNN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.CNNRNN_Res.CNNRNN_Res
    :members:


Network simulators
-----------------------------------

Mechanistic spreading models on a network. Like the ``Temporal`` compartmental
blocks, these have no ``fit()`` and are not task prototypes -- they are simulators.

.. autoclass:: epilearn.models.SpatialTemporal.NetworkSIR.NetSIR
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.DMP.DMP
    :members:


Models with known limitations
-----------------------------------

Four exported spatial-temporal classes need care before they run unchanged through
``Forecast.rolling_train`` on a dataset built by ``Dataset.generate_dataset``. They
are documented here so the failure mode is not a surprise; the alternative is listed
for each. ``MepoGNN`` is the mild case -- it is a fully supported benchmark model
that only trips on datasets carrying a *dynamic* graph.

.. autoclass:: epilearn.models.SpatialTemporal.STGCN.STGCN_c
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.DASTGN.DASTGN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.MepoGNN.MepoGNN
    :members:

.. autoclass:: epilearn.models.SpatialTemporal.STAN.STAN
    :members:

.. list-table::
   :header-rows: 1
   :widths: 16 46 38

   * - Class
     - What happens
     - Workaround
   * - ``STGCN_c``
     - ``RuntimeError: mat1 and mat2 shapes cannot be multiplied``. Its output layer
       is sized for the unpadded temporal convolution of the original paper, while the
       shared ``TimeBlock`` now pads and preserves the temporal length.
     - Use ``STGCN``.
   * - ``DASTGN``
     - ``RuntimeError: The size of tensor a (num_nodes) must match the size of tensor
       b (horizon)``. ``forward`` returns ``(batch, horizon, num_nodes)`` while every
       other model -- and ``generate_dataset``'s targets -- use
       ``(batch, num_nodes, horizon)``.
     - Transpose the output yourself, or use another graph model.
   * - ``MepoGNN``
     - Only with a **dynamic** graph: ``RuntimeError: einsum(): the number of
       subscripts in the equation (5) does not match the number of dimensions (4)``,
       because it needs a 5-D dynamic graph
       ``(batch, lookback, num_nodes, num_nodes, 1)`` while ``generate_dataset``
       produces a 4-D one. The bundled toy ``Dataset`` carries one, so every fold
       fails there. With a **static**-only graph it trains normally -- given
       ``dynamic_adj=None`` it derives its own 5-D mobility tensor from the static
       adjacency -- which is why it is a supported entry in
       ``SPATIOTEMPORAL_MODELS`` and runs through the benchmark CLI on the toy CSV
       (see :doc:`../Benchmark`).
     - Rebuild the ``Dataset`` without ``dynamic_graph=``, or call ``forward``
       directly with a 5-D ``dynamic_adj``.
   * - ``STAN``
     - ``AttributeError: 'tuple' object has no attribute 'size'``. ``forward`` returns
       ``(predictions, physical_predictions)``, and the generic training loop feeds
       that tuple straight into the loss.
     - Call ``forward`` directly and combine the two heads with your own loss.


Code Example
-------------

.. code-block:: python

    import torch
    from epilearn.models.SpatialTemporal import ColaGNN

    num_nodes=47
    num_features = 1
    lookback = 16 # inputs size
    horizon = 3 # predicts size


    graph = torch.round(torch.rand((num_nodes, num_nodes)))
    features = torch.round(torch.rand((10, lookback, num_nodes, num_features)))
    # targets are (samples, num_nodes, horizon), the layout generate_dataset produces
    node_target = torch.round(torch.rand((10, num_nodes, horizon)))

    model=ColaGNN(num_nodes = num_nodes, num_features=num_features, num_timesteps_input=lookback, num_timesteps_output=horizon, device='cpu')
    model.fit(
            train_input=features,
            train_target=node_target,
            train_graph=graph,
            val_input=None,
            val_target=None,
            val_graph=None,
            epochs=20,
            loss='mse'
            )
