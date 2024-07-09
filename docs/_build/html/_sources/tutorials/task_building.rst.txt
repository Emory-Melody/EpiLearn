Pipeline for Epidemic Modling
===================================


In this section, we will walk you through the building of epidemic modeling tasks using **Epilearn**, from dataset construction to model evaluation.

If you haven't installed or tried **Epilearn**, please refer to `Installation <https://epilearn-doc.readthedocs.io/en/latest/Installation.html>`_ and `Quickstart <https://epilearn-doc.readthedocs.io/en/latest/Quickstart.html>`_.

Dataset construction
----------------------

1. Internal 2. External 3. Simulation

.. code-block:: python

    UniversalDataset


Transformations 
----------------------

1. supported transformations 2. how it works 3. customizing your own transformations

.. code-block:: python

    transformation = transforms.Compose({
                                        "features": [transforms.normalize_feat()],
                                        "graph": [transforms.normalize_adj()]})
    dataset.transforms = transformation


Model Customization
----------------------
1. Model categories 2. Internal Models 3. customized Models




Tasks Initialization
----------------------
1. supported tasks and functions 2. how it works

.. code-block:: python

    task = Forecast(prototype=STGCN,
                    dataset=None, 
                    lookback=lookback, 
                    horizon=horizon, 
                    device='cpu')


Evaluation
----------------------
1. training and testing 2. tuning configurations 3. visualizing results
