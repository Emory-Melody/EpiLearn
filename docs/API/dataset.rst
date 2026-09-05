Dataset
===================================

In EpiLearn, we use **Dataset** to load preprocessed datasets. For customized data, we can simply initialize the Dataset given features, graphs, and states.

.. note::

   ``Dataset`` was named ``UniversalDataset`` before version 0.1.0. The old name is kept as a
   deprecated alias (``from epilearn.data import UniversalDataset``) and the constructor
   signature is unchanged, so existing scripts keep working -- but new code should use
   ``Dataset``.


Dataset class
--------------------

.. autoclass:: epilearn.data.dataset.Dataset
    :members:


Preprocessed Datasets
-----------------------------------

We collect epidemic data from various sources including the followings:

The name in ``code font`` is the value to pass as ``name=``.

**Temporal Data**

   * ``Tycho_v1`` -- `Tycho v1.0.0 <https://www.tycho.pitt.edu/data/>`_: Including eight diseases collected across 50 US states and 122 US cities from 1916 to 2009.
   * ``Measles`` -- `Measles <https://github.com/msylau/measles_competing_risks/tree/master>`_: Contains measles infections in England and Wales across 954 urban centers (cities and towns) from 1944 to 1964.
   * ``JHU_covid`` -- Johns Hopkins University global COVID-19 case counts.

**Spatial&Temporal Data**

   * ``Covid_<Country>`` with a static graph -- ``Covid_China``, ``Covid_Brazil``, ``Covid_Austria``: covid infections with static graph. `[1] <https://github.com/littlecherry-art/DASTGN/tree/master>`_
   * ``Covid_<Country>`` with a dynamic graph -- ``Covid_England``, ``Covid_France``, ``Covid_Italy``, ``Covid_NewZealand``, ``Covid_Spain``: covid infections with dynamic graph. `[2] <https://github.com/HySonLab/pandemic_tgnn/tree/main>`_ `[3] <https://github.com/deepkashiwa20/MepoGNN/tree/main>`_

``covid_static.pt`` / ``covid_dynamic.pt`` are the two downloaded archives that back
those eight names -- they are not themselves values you can pass as ``name=``.

**Dataset Loading**

Passing ``name=`` downloads the archive into ``root`` on first use and reuses the local copy
afterwards. Auxiliary tables that used to live on the dataset object
(``anual_population``, ``coordinates``, ``index``) are now collected under
``dataset.metadata``.

.. code-block:: python

    from epilearn.data import Dataset

    measle_dataset = Dataset(name='Measles', root='./tmp/')
    print(measle_dataset)                       # Dataset(Temporal, x=(946, 1108))
    print(measle_dataset.feature_names[:3])     # ['Abingdon', 'Abram', 'Accrington']
    print(measle_dataset.metadata.keys())       # anual_population, anual_birth, coordinates

    jhu_dataset = Dataset(name='JHU_covid', root='./tmp/')
    print(jhu_dataset)                          # Dataset(Temporal, x=(3342, 1143))

For other countries, please use 'Covid\_'+'country' to acquire the correspnding covid dataset.
Three countries come with a static graph -- **China**, **Brazil** and **Austria** -- and load as
spatiotemporal data. Five more come with a dynamic graph: **England**, **France**, **Italy**,
**NewZealand** and **Spain** (note the capital ``Z``); their ``x`` is 2-D ``(T, N)``, so they
print as ``Temporal`` even though ``dynamic_graph`` is populated.

.. code-block:: python

    covid_dataset = Dataset(name='Covid_Brazil', root='./tmp/')
    print(covid_dataset)
    # Dataset(Spatiotemporal, x=(122, 27, 3), graph=(27, 27))

    covid_spain = Dataset(name='Covid_Spain', root='./tmp/')
    print(covid_spain.dynamic_graph.shape)      # torch.Size([62, 52, 52])

An unsupported country name raises ``ValueError`` listing the ones that are available.
In a source checkout, pass ``root='./datasets'`` instead of ``'./tmp/'`` to use the
copies already in the repository and skip the download entirely.

.. warning::

   The named loaders fill in ``x`` (and the graph), but leave ``y`` as ``None``, so a
   built-in dataset cannot go straight into a task: ``rolling_train`` dies with
   ``TypeError: 'NoneType' object is not subscriptable``. Choose a target first::

       m = Dataset(name='Measles', root='./datasets')
       dataset = Dataset(x=m.x.unsqueeze(-1), y=m.x)   # forecast each city's own series

   Rebuilding the ``Dataset`` like this also drops ``timestamps``, which is what you
   want: ``JHU_covid`` carries 1144 timestamp labels for 3342 timesteps and the
   dynamic-graph countries 62 for 122, and ``rolling_splits`` raises on that mismatch
   (``IndexError``, or ``ValueError: The truth value of a Index is ambiguous``).

The Tycho archive stores one variable-length series per disease, so it has to be indexed by
disease before it becomes a ``Dataset``:

.. code-block:: python

    import torch
    from epilearn.data import Dataset

    try:                                         # downloads ./tmp/Tycho_v1.pt, then raises
        Dataset(name='Tycho_v1', root='./tmp/')
    except ValueError:
        pass

    raw = torch.load('./tmp/Tycho_v1.pt', weights_only=False)
    print(list(raw.keys()))
    # ['DIPHTHERIA', 'HEPATITIS A', 'MEASLES', 'MUMPS', 'SMALLPOX']
    series = raw['MEASLES']                      # torch.Size([3772])

    tycho_measles = Dataset(x=series.unsqueeze(-1), y=series)
    print(tycho_measles)                         # Dataset(Temporal, x=(3772, 1), y=(3772,))

.. warning::

   ``Dataset(name='Tycho_v1', ...)`` raises ``ValueError: only one element tensors can be
   converted to Python scalars`` because the diseases have different lengths and cannot be
   stacked into a single tensor. The call still downloads the file, which is why the snippet
   above swallows the error and then reads ``./tmp/Tycho_v1.pt`` itself.


Customize Your Own Dataset
---------------------------

For your own data, form a dictionary with keys ``features``, ``graph``, ``dynamic_graph``,
``targets`` and ``states``, then pass the arrays to ``Dataset``. Not every argument is
required -- see `Dataset class`_ for the full signature. ``examples/example.pt`` in the
repository is exactly such a file:

.. code-block:: python

    import torch
    from epilearn.data import Dataset

    data = torch.load("examples/example.pt", weights_only=False)
    print(data.keys())
    # dict_keys(['features', 'graph', 'dynamic_graph', 'targets', 'states'])

    dataset = Dataset(x=data['features'],                    # (time, nodes, channels)
                      y=data['targets'],                     # (time, nodes)
                      states=data['states'],                 # e.g. SIR states per node
                      graph=torch.Tensor(data['graph']),     # (nodes, nodes); edge_index also works
                      dynamic_graph=data['dynamic_graph'])   # (time, nodes, nodes)

    print(dataset)
    # Dataset(Spatiotemporal, x=(539, 47, 4), y=(539, 47), graph=(47, 47))
    print(dataset.edge_index.shape)              # torch.Size([2, 2189])
    print(dataset.n_timesteps, dataset.n_regions, dataset.n_features)   # 539 47 4

``graph`` is converted to ``edge_index`` / ``edge_weight`` automatically, and ``timestamps`` /
``regions`` default to ``range(T)`` / ``range(N)`` when you do not pass them. A
dictionary-style ``.pt`` file like this one can also be handed to ``Dataset.from_tensor``
directly, since the keys are auto-detected (see
`Loading from tensor and numpy files`_). For more sample code in a real training process,
refer to `examples/dataset_customization.ipynb` on the github page.


Loading from a CSV file
---------------------------

``Dataset.from_csv`` reads **long-format** data: one row per (timestamp, region), with the
features and targets as columns. This is the usual shape of public surveillance exports.

.. code-block:: text

    time,node,f0,f1,f2,f3,y
    1,0,5.0,-0.10555339604616165,2.0,0.09615384787321091,5.0
    1,1,0.0,-0.13394306600093842,2.0,0.0,0.0
    2,0,3.0,-0.05203865468502045,3.0,0.06382978707551956,3.0
    ...

Regions are named by ``region_col``; the loader pivots the frame into an ``(T, N, F)`` feature
tensor. An optional ``graph_file`` is a second CSV of edges with ``source`` / ``target``
columns (and an optional weight column via ``graph_weight_col``), which becomes an ``(N, N)``
adjacency matrix aligned to the sorted region order.

.. code-block:: python

    from epilearn.data import Dataset

    dataset = Dataset.from_csv(
        file_path="datasets/toy_features.csv",
        timestamp_col="time",
        feature_cols=["f0", "f1", "f2", "f3"],
        target_cols=["y"],
        region_col="node",
        graph_file="datasets/toy_edges.csv",
    )

    print(dataset)
    # Dataset(Spatiotemporal, x=(539, 47, 4), y=(539, 47), graph=(47, 47))
    print(dataset.timestamps[:3], dataset.regions[:3])   # [1, 2, 3] [0, 1, 2]
    print(dataset.feature_names, dataset.target_names)   # ['f0','f1','f2','f3'] ['y']

Drop ``region_col`` for a single series -- the result is a ``Temporal`` dataset.
``target_cols`` is optional: omit it and the first entry of ``feature_cols`` becomes the
target, which is the common forecasting setup. The result is a normal ``Dataset``, so it goes
straight into a task: attach transforms and call ``rolling_train`` as in the
:doc:`../Quickstart`.

.. note::

   The timestamp column is date-parsed automatically when it looks like dates, so
   ``dataset.timestamps`` holds ``pandas.Timestamp`` objects rather than strings. Index with
   ``pd.Timestamp('2020-01-01')``, or pass ``parse_dates=False`` to keep the raw strings.
   Other keyword arguments are forwarded to :class:`~epilearn.data.loaders.csv_loader.CSVLoader`
   (and from there to ``pandas.read_csv``): ``fillna_method``, ``fillna_value``,
   ``strict_numeric``, ``sort_timestamps``, ``date_format``.


Loading from tensor and numpy files
-------------------------------------

``.pt`` and ``.npy`` / ``.npz`` files are read by ``Dataset.from_tensor`` and
``Dataset.from_numpy``. For dictionary-style ``.pt`` files the keys are auto-detected from a
list of aliases (``features``/``x``/``inputs``..., ``targets``/``y``/``labels``...,
``graph``/``adj``..., ``dynamic_graph``/``od``..., ``states``/``SIR``...), so the file written
by ``Dataset.save`` round-trips without any configuration.

.. code-block:: python

    dataset.save("my_dataset.pt")               # save() now requires an explicit path

    reloaded = Dataset.from_tensor("my_dataset.pt")
    print(reloaded)
    # Dataset(Spatiotemporal, x=(539, 47, 4), y=(539, 47), graph=(47, 47))

    same = Dataset.load("my_dataset.pt")        # equivalent for files written by save()

    # a bare .npy array is taken as the features
    arrays = Dataset.from_numpy("features.npy")  # file holding a (50, 6, 2) array
    print(arrays)                               # Dataset(Spatiotemporal, x=(50, 6, 2))


Loaders
---------------------------

The loaders behind the ``from_*`` constructors are also usable directly. They return a
:class:`~epilearn.data.core.TimeSeriesData` (or a ``LoadedData`` for ``load_from_file``, which
picks the loader from the file extension) rather than a ``Dataset``; wrap it with
``Dataset.from_time_series_data`` when you need the full dataset API. This is the hook to use
if you want to inspect or patch the parsed arrays before building the dataset.

.. code-block:: python

    from epilearn.data import (Dataset, load_csv, load_tensor, load_numpy,
                               load_from_file, DataLoaderRegistry)

    print(DataLoaderRegistry.supported_extensions())
    # ['csv', 'pt', 'pth', 'npy', 'npz']

    ts = load_csv("datasets/toy_features.csv",
                  timestamp_col="time",
                  feature_cols=["f0", "f1", "f2", "f3"],
                  target_cols=["y"],
                  region_col="node")
    print(ts)
    # TimeSeriesData(Spatiotemporal, features=(539, 47, 4), targets=(539, 47), T=539, N=47)

    loader_dataset = Dataset.from_time_series_data(ts)

    ts_pt  = load_tensor("my_dataset.pt")
    ts_npy = load_numpy("features.npy")

    loaded = load_from_file("datasets/toy_features.csv", timestamp_col="time",
                            feature_cols=["f0", "f1", "f2", "f3"], target_cols=["y"],
                            region_col="node")
    same_dataset = Dataset.from_time_series_data(loaded.to_time_series_data())

.. autofunction:: epilearn.data.loaders.load_csv

.. autofunction:: epilearn.data.loaders.load_tensor

.. autofunction:: epilearn.data.loaders.load_numpy

.. autofunction:: epilearn.data.loaders.load_from_file

.. autoclass:: epilearn.data.loaders.csv_loader.CSVLoader
    :members: load

.. autoclass:: epilearn.data.loaders.tensor_loader.TensorLoader
    :members: load

.. autoclass:: epilearn.data.loaders.tensor_loader.NumpyLoader
    :members: load

.. autoclass:: epilearn.data.core.TimeSeriesData

.. autoclass:: epilearn.data.core.LoadedData
    :members: to_time_series_data


Transformations
---------------------------

Transformations are attached with ``set_transforms``. By default they are stored and applied
by the task during training; pass ``apply_now=True`` to rewrite the dataset immediately.
``get_process_history`` returns the statistics that were used, which is what you need to map
predictions back to the original scale.

.. code-block:: python

    from epilearn.utils import transforms

    transformation = transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target": [transforms.normalize_target()],
        "graph": [transforms.normalize_adj()]})

    dataset.set_transforms(transformation, apply_now=True)

    print(dataset.x.mean(), dataset.x.std())        # ~0.0, ~1.0
    print(sorted(dataset.get_process_history()))
    # ['feat_mean', 'feat_std', 'target_mean', 'target_std']

    # equivalent, in two steps
    dataset.set_transforms(transformation)
    dataset.apply_transforms()                      # in place; inplace=False returns
                                                    # (transformed_dict, process_history)

.. note::

   ``Dataset.get_transformed()`` was removed in 0.1.0; use ``set_transforms`` /
   ``apply_transforms``. Note also that metrics are **not** automatically denormalized: with
   ``normalize_target()`` in the pipeline, reported errors are in normalized units unless you
   ask for ``inverse_normalize=True`` when evaluating.


Slicing and rolling splits
---------------------------

``get_slice`` returns a new ``Dataset`` restricted to a timestamp range and/or a subset of
regions. The range can be given as timestamp *values* (``start`` / ``end``, exclusive unless
``end_inclusive=True``; ``pandas.Timestamp`` for date-typed series) or as proportions of the
series (``start_rate`` / ``end_rate``). Slicing a region subset also slices ``graph`` and
``dynamic_graph``, and carries the transforms and ``process_history`` over.

.. code-block:: python

    train = dataset.get_slice(end_rate=0.7)
    test  = dataset.get_slice(start_rate=0.7)
    print(train, test)
    # Dataset(Spatiotemporal, x=(377, 47, 4), ...) Dataset(Spatiotemporal, x=(162, 47, 4), ...)

    # timestamp values, inclusive end, and a region subset
    window = dataset.get_slice(start=10, end=20, end_inclusive=True, regions=[0, 1, 2])
    print(window)                # Dataset(Spatiotemporal, x=(11, 3, 4), y=(11, 3), graph=(3, 3))

``rolling_splits`` yields the ``(train, val, test)`` triples used for rolling-origin
evaluation -- the protocol that ``rolling_train`` runs internally. ``expanding=True`` grows the
training window each fold; ``expanding=False`` slides a fixed-size one. ``val_size=0`` yields
``None`` for the validation dataset. ``step_size`` (default: ``test_size``, i.e.
non-overlapping test windows) controls how far the windows advance per fold, and ``regions``
restricts every split to a subset of regions.

.. code-block:: python

    for train, val, test in dataset.rolling_splits(train_size=300, val_size=50, test_size=50):
        print(train.n_timesteps, val.n_timesteps, test.n_timesteps,
              test.timestamps[0], test.timestamps[-1])

    # 300 50 50 351 400
    # 350 50 50 401 450
    # 400 50 50 451 500
    # 450 50 39 501 539     <- trailing partial fold, kept when >= test_size // 2 remains


Sliding windows: ``generate_dataset``
--------------------------------------

``generate_dataset`` turns a dataset into the supervised (lookback, horizon) samples a model
consumes. It returns a **dict** with keys ``features``, ``targets``, ``states``,
``dynamic_graph`` and ``graph`` -- in 0.0.x this was a tuple.

.. code-block:: python

    split = dataset.generate_dataset(X=dataset.x,
                                     Y=dataset.y,
                                     adj=dataset.graph,
                                     lookback_window_size=12,
                                     horizon_size=3)

    print(list(split.keys()))
    # ['features', 'targets', 'states', 'dynamic_graph', 'graph']
    print(split['features'].shape)      # torch.Size([525, 12, 47, 4])
    print(split['targets'].shape)       # torch.Size([525, 47, 3])
    print(split['graph'].shape)         # torch.Size([47, 47])

.. warning::

   Pass ``adj=dataset.graph`` explicitly. ``generate_dataset`` does not fall back to the
   dataset's own graph, so omitting it silently produces ``graph=None`` and graph-based models
   will fail. The same holds for ``states=`` and ``dynamic_adj=``.

These dicts are what the low-level ``task.train_model(train_split=..., val_split=...,
test_split=...)`` and ``task.evaluate_model(dataset=<split dict>)`` expect. Use ``interval=``
(set to ``max_lookback - lookback``) when you compare models with different lookbacks and want
them to predict the same time points.
