Model&Dataset Customization
===================================

Dataset Customization
-----------------------

EpiLearn provides the :class:`~epilearn.data.dataset.Dataset` class to load all datasets,
including time series data, static graph, and dynamic graphs. What you pass depends
on the task:

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Task
     - ``x`` / ``y``
     - ``graph`` / ``dynamic_graph``
   * - temporal
     - ``[Length, Channels]`` / ``[Length, 1]``
     - not used
   * - spatial-temporal
     - ``[Length, num_nodes, Channels]`` / ``[Length, num_nodes]``
     - ``[num_nodes, num_nodes]`` / ``[Length, num_nodes, num_nodes]``
   * - spatial
     - ``[num_samples, num_nodes, Channels]`` / ``[num_samples, num_nodes]``
     - ``[num_nodes, num_nodes]`` / ``[num_samples, num_nodes, num_nodes]``

.. code-block:: python

    from epilearn.data import Dataset

    dataset = Dataset(x=features, y=targets)                       # temporal
    dataset = Dataset(x=node_features, y=targets,                  # spatial-temporal
                      graph=static_graph, dynamic_graph=dynamic_graph)

The target may also be one of the feature channels; for univariate data ``x`` and
``y`` are simply the **same** tensor. The spatial and spatial-temporal shapes look
alike, but the first dimension means something different — samples rather than time
— so for a spatial task ``graph`` is the graph shared by every sample while
``dynamic_graph`` gives each sample its own.

Everything else is optional metadata, but ``timestamps`` and ``regions`` are worth
filling in because the rolling-window splitter reports them back to you as fold
boundaries:

.. code-block:: python

    import torch
    import pandas as pd
    from epilearn.data import Dataset

    T, N, F = 300, 6, 3
    dataset = Dataset(x=torch.rand(T, N, F), y=torch.rand(T, N),
                      graph=torch.rand(N, N),
                      dynamic_graph=torch.rand(T, N, N),
                      timestamps=pd.date_range('2020-01-01', periods=T).tolist(),
                      regions=[f"r{i}" for i in range(N)],
                      feature_names=['cases', 'deaths', 'tests'],
                      target_names=['cases'])
    print(dataset, dataset.n_timesteps, dataset.n_regions, dataset.n_features)
    # Dataset(Spatiotemporal, x=(300, 6, 3), y=(300, 6), graph=(6, 6)) 300 6 3
    print(dataset.edge_index.shape)   # torch.Size([2, 36])  derived from graph

.. note::
   Numpy arrays are converted to ``torch.FloatTensor`` in the constructor, and
   ``edge_index`` / ``edge_weight`` are derived from ``graph`` if you do not supply
   them. ``timestamps`` defaults to ``range(T)`` and, for spatiotemporal input,
   ``regions`` defaults to ``range(N)``. Several attribute and keyword names changed
   in 0.1.0 (including ``dataset.features`` → ``dataset.feature_names``); see
   `MIGRATION.md <https://github.com/Emory-Melody/EpiLearn/blob/main/MIGRATION.md>`_.

Long-format CSVs go through ``from_csv``; ``from_tensor`` and ``from_numpy`` do the
same for ``.pt`` and ``.npy`` files, and ``save`` / ``load`` round-trip a built
dataset (``save`` requires an explicit ``path``):

.. code-block:: python

    dataset = Dataset.from_csv('cases.csv',
                               timestamp_col='date',
                               region_col='state',            # omit for a temporal dataset
                               feature_cols=['cases', 'deaths', 'tests'],
                               target_cols=['cases'],         # defaults to feature_cols[0]
                               graph_file='edges.csv')        # edge list: source,target,weight
    dataset.save('./my_dataset.pt')
    dataset = Dataset.load('./my_dataset.pt')

For more coding details, please refer to `Dataset Customization <https://github.com/Emory-Melody/EpiLearn/blob/main/examples/dataset_customization.ipynb>`_.


Transformation Customization
------------------------------

A transformation is just an ``nn.Module`` with
``forward(self, X, device='cpu', **kwargs) -> torch.Tensor``, taking the tensor for
one dataset component and returning the transformed one. ``Compose`` maps component
names — ``"features"``, ``"target"``, ``"graph"``, ``"dynamic_graph"``,
``"states"`` — to lists of such modules and applies each list in order. Two custom
examples, one stateless with an inverse and one that keeps state:

.. code-block:: python

    import torch
    import torch.nn as nn
    from epilearn.data import Dataset
    from epilearn.utils import transforms

    class log1p_feat(nn.Module):
        """Compress heavy-tailed count features."""
        def forward(self, X, device='cpu', **kwargs):
            return torch.log1p(torch.clamp(X, min=0.0))

        def denorm(self, X):                      # optional inverse
            return torch.expm1(X)

    class clip_outliers(nn.Module):
        def __init__(self, quantile=0.99):
            super().__init__()
            self.quantile = quantile
            self.cap = None                       # keep your own state here

        def forward(self, X, device='cpu', **kwargs):
            self.cap = torch.quantile(X.flatten().float(), self.quantile)
            return torch.clamp(X, max=self.cap)

    pipeline = transforms.Compose({
        "features": [log1p_feat(), transforms.normalize_feat()],
        "target":   [clip_outliers(0.99), transforms.normalize_target()]})

Three methods drive a pipeline: ``set_transforms`` registers it,
``apply_transforms`` runs it, and ``get_process_history`` reads back the fitted
constants (they replace the removed ``get_transformed()`` / ``ganerate_splits()``
/ ``get_splits()``). Registering without applying is the right default for
training, because ``rolling_train`` re-fits the normalization constants on each
fold's training window and reuses them for that fold's validation and test
windows, which is what keeps the evaluation leak-free. Apply eagerly only when you
are building splits by hand:

.. code-block:: python

    toy = Dataset(); toy.load_toy_dataset()
    dataset = Dataset(x=toy.x[:, :5], y=toy.y[:, :5])

    dataset.set_transforms(pipeline)              # lazy: registered, not applied
    print(dataset.get_transforms() is pipeline)   # True
    print(dataset.get_process_history())          # {}  -- nothing fitted yet

    # non-destructive: returns the transformed dict, leaves dataset.x/.y alone
    data, history = dataset.apply_transforms(inplace=False)
    print(sorted(data), sorted(history))
    # ['dynamic_graph', 'features', 'graph', 'states', 'target']
    # ['feat_mean', 'feat_std', 'target_mean', 'target_std']

    dataset.apply_transforms(inplace=True)        # == set_transforms(pipeline, apply_now=True)
    history = dataset.get_process_history()

Keep that ``history``: **nothing is auto-denormalized**, so to report errors in
original units you either pass ``inverse_normalize=True`` and
``process_history=history`` to ``evaluate_model``, or undo it yourself:

.. code-block:: python

    y_original = dataset.y * torch.tensor(history['target_std']) \
                          + torch.tensor(history['target_mean'])

.. warning::
   ``Compose.__call__`` returns a **tuple** ``(data, process_history)``, so
   ``data = transformation(data)`` silently binds the tuple; call it as
   ``data, history = pipeline({...})``.

   ``process_history`` is populated **only** by ``normalize_feat`` and
   ``normalize_target`` — ``Compose`` special-cases those two class names. A custom
   transform's state is not recorded there, so keep it on the transform instance
   (like ``clip_outliers.cap`` above).

   ``apply_transforms`` only *warns* (``No transforms set. Call set_transforms()
   first.``) and passes the data through unchanged if you never registered a
   pipeline.

To see the built-in transformations you can compose with, see :doc:`../API/utils`.


Model Customization
-----------------------

A customized model must inherit from the base class of its family — the example
below builds an LSTM; see also the `Model Customization notebook
<https://github.com/Emory-Melody/EpiLearn/blob/main/examples/model_customization.ipynb>`_.
Each base class supplies ``fit`` / ``train_epoch`` / ``evaluate`` / ``predict`` and
calls ``forward`` with a different signature — the contract your model must match:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Base class
     - ``forward`` is called as
     - Input shape
   * - ``models.Temporal.base.BaseModel``
     - ``forward(feature)``
     - ``[batch, lookback, channels]``
   * - ``models.SpatialTemporal.base.BaseModel``
     - ``forward(feature, graph, states, dynamic_graph)``
     - ``[batch, lookback, num_nodes, channels]``
   * - ``models.Spatial.base.BaseModel``
     - ``forward(x, edge_index, edge_weight)``
     - ``[batch * num_nodes, channels]``

**Step 1 — define the model.** Inherit from the base class of the matching family
and write ``__init__``, ``forward`` and ``initialize``. The task fills in
``num_features``, ``num_timesteps_input``, ``num_timesteps_output`` and ``device``
from the data (plus ``num_nodes`` when a graph is present), so those names are
fixed; everything else is yours and arrives through ``model_args`` in Step 2.
Accept ``**kwargs`` so a stray argument does not break the constructor. ``forward``
must return the label's shape: ``[batch, horizon]`` for a temporal model,
``[batch, num_nodes, horizon]`` for a spatial-temporal one.

.. code-block:: python

    import torch.nn as nn
    from epilearn.models.Temporal.base import BaseModel

    class CustomizedTemporal(BaseModel):
        def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                     hidden_size=32, num_layers=2, bidirectional=False,
                     device='cpu', **kwargs):
            super(CustomizedTemporal, self).__init__(device=device)
            self.horizon = num_timesteps_output
            self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True,
                                bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), self.horizon)

        def forward(self, feature, graph=None, states=None, dynamic_graph=None, **kargs):
            out, _ = self.lstm(feature)      # [batch, lookback, hidden * num_directions]
            return self.fc(out[:, -1, :])    # decode the last hidden state

        def initialize(self):
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

.. warning::
   ``initialize`` is **not** optional in practice, even though no base class defines
   it: ``train_model`` defaults to ``initialize=True`` and calls
   ``self.model.initialize()``, so a model without it dies with
   ``AttributeError: 'YourModel' object has no attribute 'initialize'`` — printed,
   not raised, because ``train_model`` swallows training exceptions and returns
   ``None``. ``rolling_train`` passes ``initialize=False``, so the same model
   appears to work there. Define ``initialize`` regardless, even as ``pass``.

**Step 2 — train it** with the same pipeline as in :doc:`task_building`, passing
your own hyperparameters through ``model_args``:

.. code-block:: python

    import torch
    from epilearn.data import Dataset
    from epilearn.utils import transforms
    from epilearn.tasks.forecast import Forecast

    # a noisy cosine as a stand-in dataset; x and y are the same univariate series
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 500)
    inputs = (torch.cos(2 * torch.pi * 3 * t) + 0.1 * torch.randn(500)).reshape(-1, 1)
    dataset = Dataset(x=inputs, y=inputs)
    dataset.set_transforms(transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target":   [transforms.normalize_target()]}))

    task = Forecast(prototype=CustomizedTemporal, lookback=36, horizon=3, device='cpu')
    result = task.rolling_train(dataset=dataset,
                                train_size=300, val_size=60, test_size=60, step_size=60,
                                train_loss='mse', epochs=40, batch_size=16, lr=1e-3,
                                model_args={"hidden_size": 32, "num_layers": 2,
                                            "bidirectional": False})
    print(result['aggregate_metrics'])
    # MSE: 0.0282 ± 0.0096   MAE: 0.1352 ± 0.0210   Coverage: 89.4% ± 3.0%

``model_args`` carries **only** your hyperparameters — do not repeat
``num_features``, ``num_timesteps_input``, ``num_timesteps_output`` or ``device``,
which the task derives from the split. The same keys are what
``optuna_model_args`` searches over when you enable Optuna.

A customized spatial-temporal model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The only differences are the base class, the extra ``num_nodes`` argument, and the
4-D input:

.. code-block:: python

    import torch
    import torch.nn as nn
    from epilearn.models.SpatialTemporal.base import BaseModel

    class CustomizedST(BaseModel):
        def __init__(self, num_nodes, num_features, num_timesteps_input,
                     num_timesteps_output, hidden=32, device='cpu', **kwargs):
            super(CustomizedST, self).__init__(device=device)
            self.num_nodes = num_nodes
            self.gcn = nn.Linear(num_features, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, num_timesteps_output)

        def forward(self, feature, graph=None, states=None, dynamic_graph=None, **kwargs):
            # feature: [batch, lookback, num_nodes, num_features]
            h = self.gcn(feature)                               # [B, L, N, H]
            if graph is not None:
                h = torch.einsum('ij,bljh->blih', graph, h)     # one diffusion step
            h = torch.relu(h)
            b, l, n, hid = h.shape
            h = h.permute(0, 2, 1, 3).reshape(b * n, l, hid)    # nodes into the batch
            h, _ = self.gru(h)
            out = self.out(h[:, -1, :])                         # [B*N, horizon]
            return out.reshape(b, n, -1)                        # [B, N, horizon]

        def initialize(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

Training is Step 2 again, on a dataset that carries a graph — add
``"graph": [transforms.normalize_adj()]`` to the ``Compose`` and pass
``model_args={'hidden': 32}``:

.. code-block:: python

    toy = Dataset(); toy.load_toy_dataset()
    dataset = Dataset(x=toy.x[:, :8], y=toy.y[:, :8], graph=toy.graph[:8, :8])
    dataset.set_transforms(transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target":   [transforms.normalize_target()],
        "graph":    [transforms.normalize_adj()]}))

    task = Forecast(prototype=CustomizedST, lookback=12, horizon=3, device='cpu')
    result = task.rolling_train(dataset=dataset,
                               train_size=350, val_size=60, test_size=60, step_size=60,
                               epochs=25, batch_size=32, model_args={'hidden': 32})

The output must be shaped ``[batch, num_nodes, horizon]`` to match the targets that
``generate_dataset`` produces. ``graph`` arrives as a dense ``[N, N]`` matrix
(already normalized if ``normalize_adj`` is in the pipeline), ``dynamic_graph`` as
``[batch, lookback, N, N]`` and ``states`` as ``[batch, lookback, N, n_states]`` —
each is ``None`` when the dataset does not carry it, so guard before using them.

.. tip::
   Register a finished model in ``epilearn/benchmark.py`` — add its name to
   ``TEMPORAL_MODELS``, ``SPATIOTEMPORAL_MODELS`` or ``FOUNDATION_MODELS`` — and it
   becomes available to the YAML benchmark runner without further changes.
