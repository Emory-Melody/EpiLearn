Quickstart for EpiLearn
==========================

For forecast task, ...

.. code-block:: python

    from epilearn.models.SpatialTemporal.STGCN import STGCN
    from epilearn.data import UniversalDataset
    from epilearn.utils import transforms
    from epilearn.tasks.forecast import Forecast
    # initialize settings
    lookback = 12 # inputs size
    horizon = 3 # predicts size
    # load toy dataset
    dataset = UniversalDataset()
    dataset.load_toy_dataset()
    # Adding Transformations
    transformation = transforms.Compose({
                    "features": [transforms.normalize_feat()],
                    "graph": [transforms.normalize_adj()]})
    dataset.transforms = transformation
    # Initialize Task
    task = Forecast(prototype=STGCN,
                    dataset=None, 
                    lookback=lookback, 
                    horizon=horizon, 
                    device='cpu')
    # Training
    result = task.train_model(dataset=dataset, 
                            loss='mse', 
                            epochs=50, 
                            batch_size=5, 
                            permute_dataset=True)
    # Evaluation
    evaluation = task.evaluate_model()


For detection task, ...

.. code-block:: python

    from epilearn.models.Spatial.GCN import GCN
    from epilearn.data import UniversalDataset
    from epilearn.utils import transforms
    from epilearn.tasks.detection import Detection
    # initialize settings
    lookback = 1 # inputs size
    horizon = 2 # predicts size; also seen as number of classes
    # load toy dataset
    dataset = UniversalDataset()
    dataset.load_toy_dataset()
    # Adding Transformations
    transformation = transforms.Compose({
                    " features": [],
                    " graph": []})
    dataset.transforms = transformation
    # Initialize Task
    task = Detection(prototype=GCN, 
                    dataset=None, 
                    lookback=lookback, 
                    horizon=horizon, 
                    device='cpu')
    # Training
    result = task.train_model(dataset=dataset, 
                            loss='ce', 
                            epochs=50, 
                            batch_size=5)
    # Evaluation
    evaluation = task.evaluate_model()
