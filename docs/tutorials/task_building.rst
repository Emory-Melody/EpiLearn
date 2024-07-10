Pipeline for Epidemic Modling
===================================


In this section, we will walk you through the building of epidemic modeling tasks using **Epilearn**, from dataset construction to model evaluation.

If you haven't installed or tried **Epilearn**, please refer to `Installation <https://epilearn-doc.readthedocs.io/en/latest/Installation.html>`_ and `Quickstart <https://epilearn-doc.readthedocs.io/en/latest/Quickstart.html>`_.

Dataset construction
----------------------

To begin with, we need to choose and load data using a UniversalDataset defined by Epilearn. 

1. Pre-compiled data is also available, which is adopted from other researches and processed. To view more available internal data, please refer to `Datasets <https://github.com/Emory-Melody/EpiLearn/tree/main/datasets>`_.
2. Users can also use external data, initializing UniversalDataset with the corresponding values directly.
3. Epilearn also provides simulation methods to generate simulated data. See  `Simulation <https://vermillion-malasada-a2864e.netlify.app/html/tutorials/simulation>`_ for more details.


Transformations 
----------------------

Epilearn provides numerous transformations including normalization, seasonal decomposition, converting to frequency domain, etc. The loading of these functions follows similar style to Torch, as shown below.

To customize your own transformation, simply add new a new class following the style shown in `Transformations <https://vermillion-malasada-a2864e.netlify.app/html/api/utils#transformation>`_.

.. code-block:: python

    transformation = transforms.Compose({
                                        "features": [transforms.normalize_feat()],
                                        "graph": [transforms.normalize_adj()]})
    dataset.transforms = transformation


Model Customization
----------------------

Epilearn supports three different models: 1. Spatial models (input size: [Batch*Nodes*Channels]) 2. Temporal models (Input size: [Batch*Window*Channels]) 3. Spatial-Temporal model (input size: [Batch*Window*Nodes*Channels])

For models implemented, please refer to `Models <https://vermillion-malasada-a2864e.netlify.app/html/api/models>`_.

To build a customized model, you can simply create a model class inherited from BaseModel with a forward function, as shown below. 

.. code-block:: python

    from .base import BaseModel

    class CustomizedModel(BaseModel):
        def __init__(self, 
                    num_nodes, 
                    num_features, 
                    num_timesteps_input, 
                    num_timesteps_output, 
                    device = 'cpu'):
            super(CustomizedModel, self).__init__(device=device)
            pass

        def forward(self, feature, graph, states, dynamic_graph, **kargs):
            pass

        def initialize(self):
            pass

Task Initialization and Model Evaluation
-------------------------------------------

Epilearn currently supports two tasks: Forecast and Detection. Forecast task takes in a UniversalDataset, model prototype and other configurations like lookback window size and the horizon size. The same setting applies to the Detection task. After initializing, you can try functions like .train_model() and .evaluate_model() to the model performance on the given dataset.

.. code-block:: python

    task = Forecast(prototype=STGCN,
                    dataset=None, 
                    lookback=lookback, 
                    horizon=horizon, 
                    device='cpu')
    task.train_model(customized_dataset, loss='mse', epoch=50, lr=0.001)
    task.evaluate_model()

