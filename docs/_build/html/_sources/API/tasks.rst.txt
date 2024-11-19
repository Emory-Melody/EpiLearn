Tasks
===================================

Forecast
----------

.. autoclass:: epilearn.tasks.forecast.Forecast
    :members:
    :undoc-members:  Tasks
===================================
...

.. code-block:: python

    import torch
    from epilearn.data import UniversalDataset
    from epilearn.tasks.forecast import Forecast
    from epilearn.models.SpatialTemporal import STGCN

    lookback = 36 # inputs size
    horizon = 3 # predicts size

    dataset = UniversalDataset()
    dataset.load_toy_dataset()
    dataset.graph = torch.FloatTensor(dataset.graph)

    task = Forecast(prototype=STGCN,
                    dataset=None,
                    lookback=lookback,
                    horizon=horizon,
                    device='cpu')

    result = task.train_model(dataset=dataset,
                            loss='mse',
                            epochs=2,
                            batch_size=5,
                            train_rate=0.6,
                            val_rate=0.2,
                            permute_dataset=True,
                            )



Detection
----------

.. autoclass:: epilearn.tasks.detection.Detection
    :members:
    :undoc-members:  

.. code-block:: python

    import torch
    from epilearn.data import UniversalDataset
    from epilearn.tasks.detection import Detection
    from epilearn.models.Spatial import GCN

    lookback = 1 # inputs size
    horizon = 2 # predicts size

    graph = torch.round(torch.rand((47,47)))
    features = torch.round(torch.rand((10,47,1,4))) # batch, nodes, time steps=1, channels
    node_target = torch.round(torch.rand((10,47))) # batch, nodes

    dataset = UniversalDataset(x=features,y=node_target,graph=graph)

    task = Detection(prototype=GCN, dataset=dataset, lookback=lookback, horizon=horizon, device='cpu')

    result = task.train_model(dataset=dataset, 
                            loss='ce', 
                            epochs=25,
                            train_rate=0.6,
                            val_rate=0.1,
                            permute_dataset=False,
                            )


