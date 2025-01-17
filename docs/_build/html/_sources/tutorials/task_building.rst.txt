Pipeline for Epidemic Modeling
===================================


In this section, we will walk you through the building of epidemic modeling tasks using **Epilearn**, from dataset construction to model evaluation.

If you haven't installed or tried **Epilearn**, please refer to `Installation <https://epilearn-doc.readthedocs.io/en/latest/Installation.html>`_ and `Quickstart <https://epilearn-doc.readthedocs.io/en/latest/Quickstart.html>`_.

0. Colab Tutorial
----------------------

If you are new to pytorch and EpiLearn, our `Lab Tutorial <https://colab.research.google.com/drive/13D5U-S-U2DhR9OKXdsGuGE2gR1fO2Y4T>`_ will help you understand how to use this package through a case study.


1. Dataset construction
-------------------------

To begin with, we need to choose and load data using a UniversalDataset defined by Epilearn. 

* Users can use external data, initializing UniversalDataset with the corresponding values directly.

.. code-block:: python

    UniversalDataset(x=Features,
                     states=SIR_States,
                     y=Labels,
                     graph=Static_Graph,
                     dynamic_graph=Dynamic_Graph)

* Pre-compiled data is also available, which is adopted from other researches. Simply input the dataset name and the location to store the data:

.. code-block:: python

    measle_dataset = UniversalDataset(name='Measles', root='./tmp/')   

To view more available internal data, please refer to `Datasets <https://github.com/Emory-Melody/EpiLearn/tree/main/datasets>`_.

* Epilearn also provides simulation methods to generate simulated data. See  `Simulation <https://vermillion-malasada-a2864e.netlify.app/html/tutorials/simulation>`_ for more details.


2. Transformations 
----------------------

Epilearn provides numerous transformations including normalization, seasonal decomposition, converting to frequency domain, etc. The loading of these functions follows similar style to Pytorch, as shown below.

To customize your own transformation, simply add a new class following the style shown in `Transformations <https://vermillion-malasada-a2864e.netlify.app/html/api/utils#transformation>`_.

.. code-block:: python

    transformation = transforms.Compose({
                                        "features": [transforms.normalize_feat()],
                                        "graph": [transforms.normalize_adj()]})
    dataset.transforms = transformation


3. Model Customization
----------------------

Epilearn supports three types of models: 1. Spatial models (input size: [Batch*Nodes*Channels]) 2. Temporal models (Input size: [Batch*Window*Channels]) 3. Spatial-Temporal model (input size: [Batch*Window*Nodes*Channels])

For models implemented in Epilearn, please refer to `Models <https://vermillion-malasada-a2864e.netlify.app/html/api/models>`_.

To build a customized model, you can simply create a model class inherited from BaseModel with a forward function, as shown below: 

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

4. Task Initialization and Model Evaluation
-------------------------------------------

Epilearn currently supports two tasks: Forecast and Detection. The Forecast task takes in a UniversalDataset, model prototype and other configurations like lookback window size and the horizon size. The same setting applies to the Detection task. After initializing, you can try functions like .train_model() and .evaluate_model() to test the models performance on given datasets.

* Initialize Task

.. code-block:: python

    task = Forecast(prototype=customized_model, 
                    lookback=lookback, 
                    horizon=horizon
                    )

* Train and evaluate_model

.. code-block:: python

    result = task.train_model(
                            dataset=customized_dataset, 
                            loss='mse',    # specificy the loss function to be Mean Squared Error (MSE)
                            epochs=5,      # number of training epochs/iterations
                            train_rate=0.6,  # 60% is used for training
                            val_rate=0.2,    # 20% is used for validation; the rest 20% is for testing
                            batch_size=5,  # batch size for training
                            device='cpu')  # use CPU for model training; set `device='cuda'` to enable GPU training
    
    task.evaluate_model()

