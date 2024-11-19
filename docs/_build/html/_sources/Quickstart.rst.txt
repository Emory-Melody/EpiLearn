Quickstart for EpiLearn
==========================

For forecast task, Epilearn initialzes a Forecast class with model prototyes and the dataset. Before building the task, a UniversalDataset is used to load the spatial&temporal data and transformations are defined and applied. 

.. code-block:: python

    import torch
    from epilearn.models.Temporal import DlinearModel
    from epilearn.data import UniversalDataset
    from epilearn.tasks.forecast import Forecast
    from epilearn.utils import transforms


    lookback = 16 # inputs size
    horizon = 1 # predicts size

    dataset = UniversalDataset(name='JHU_covid', root='./tmp/')
    inputs = dataset.x[0,:].unsqueeze(1)

    dataset = UniversalDataset(x=inputs, y=inputs)

    transformation = transforms.Compose({
                    "features": [transforms.normalize_feat()],
                    "target": [transforms.normalize_feat()]
                    })
    dataset.transforms = transformation

    task = Forecast(prototype=DlinearModel,
                        dataset=None,
                        lookback=lookback,
                        horizon=horizon,
                        device='cpu')

    result = task.train_model(dataset=dataset,
                            loss='mse',
                            epochs=50,
                            batch_size=16,
                            train_rate=0.6,
                            val_rate=0.1,
                            permute_dataset=False,
                            )

    eval_result = task.evaluate_model(model=task.model,
                                    features=task.train_split['features'],
                                    targets=task.train_split['targets'])
    
    forecasts = task.plot_forecasts(task.train_dataset, index_range=[0, -1])

.. image:: demo.png
   :alt: Demo
   :width: 500px
   :align: center





For detection task, the process is the same as the forecast task except that we initialze a Detection task.

.. code-block:: python

    import torch
    from epilearn.tasks.detection import Detection
    from epilearn.models.Spatial.GCN import GCN

    graph = torch.round(torch.rand((47, 47))) # nodes, nodes
    features = torch.round(torch.rand((10,47,1,4))) # batch, nodes, time steps=1, channels
    node_target = torch.round(torch.rand((10,47))) # batch, nodes

    dataset = UniversalDataset(x=features,y=node_target,graph=graph)

    lookback = 1 # inputs size
    horizon = 2 # predicts size; also seen as number of classes

    dataset.transforms = None
    task = Detection(prototype=GCN, dataset=dataset, lookback=lookback, horizon=horizon, device='cpu')

    result = task.train_model(dataset=dataset, 
                            loss='ce', 
                            epochs=25,
                            train_rate=0.6,
                            val_rate=0.1,
                            permute_dataset=False,
                            #   model_args=model_args
                            )

    train_evaluation = task.evaluate_model(model=task.model,
                                    features=task.train_split['features'],
                                    graph=task.adj, 
                                    dynamic_graph= task.train_split['dynamic_graph'], 
                                    states=task.train_split['states'], 
                                    targets=task.train_split['targets'])
