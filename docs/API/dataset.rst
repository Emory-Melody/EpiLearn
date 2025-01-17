Dataset
===================================

In EpiLearn, we use **UniversalDataset** to load preprocessed datasets. For customized data, we can simply initialize the UniversalDataset given features, graphs, and states.


UniversalDataset
--------------------

.. autoclass:: epilearn.data.dataset.UniversalDataset
    :members:


Preprocessed Datasets
===================================

We collect epidemic data from various sources including the followings:


**Temporal Data**

   * `Tycho_v1.0.0 <https://www.tycho.pitt.edu/data/>`_: Including eight diseases collected across 50 US states and 122 US cities from 1916 to 2009.
   * `Measles <https://github.com/msylau/measles_competing_risks/tree/master>`_: Contains measles infections in England and Wales across 954 urban centers (cities and towns) from 1944 to 1964.

**Spatial&Temporal Data**

   * **Covid_static**: Contains covid infections with static graph. `[1] <https://github.com/littlecherry-art/DASTGN/tree/master>`_
   * **Covid_dynamic**: Contains covid infections with dynamic graph. `[2] <https://github.com/HySonLab/pandemic_tgnn/tree/main>`_ `[3] <https://github.com/deepkashiwa20/MepoGNN/tree/main>`_

**Dataset Loading**

Loading Measle and Tycho Datasets:

.. code-block:: python

    from epilearn.data import UniversalDataset  

    tycho_dataset = UniversalDataset(name='Tycho_v1', root='./tmp/')    

    measle_dataset = UniversalDataset(name='Measles', root='./tmp/')    


For covid data, we support the Dataset from Johns Hopkings University:

.. code-block:: python

    from epilearn.data import UniversalDataset

    jhu_dataset = UniversalDataset(name='JHU_covid', root='./tmp/')


For other countries, please use 'Covid\_'+'country' to acquire the correspnding covid dataset. Currently, we support countries like China, Brazil, Austria, England, France, Italy, Newzealand, and Spain.

.. code-block:: python

    from epilearn.data import UniversalDataset

    covid_dataset = UniversalDataset(name='Covid_Brazil', root='./tmp/')


Customize Your Own Dataset
---------------------------

First, you should form your data as a dictionary with keys of features, graph, dynamic_graph, targets, and states. Here is an example:

.. code-block:: python

    data = torch.load("example.pt")
    
    data.keys()

.. code-block:: text

    dict_keys(['features', 'graph', 'dynamic_graph', 'targets', 'states'])

.. code-block:: python

    node_features = data['features']    # [time steps, nodes, channels]: torch.Size([539, 47, 4])

    static_graph = torch.Tensor(data['graph'])  # [nodes, nodes]: (47, 47)

    dynamic_graph = data['dynamic_graph']   # [time steps, nodes, nodes]: torch.Size([539, 47, 47])

    targets = data['targets']   # [time steps, nodes]: torch.Size([539, 47])

    node_status = data['states']    # [time steps, nodes]: torch.Size([539, 47])


Next, you can use your own data to establish a `UniversalDataset` class by passing the correponding parameters due to your needs. Not every parameters are required. You can refer to `UniversalDataset`_ to obtain detailed descriptions and customize your parameters.

.. code-block:: python
    
    from epilearn.data import UniversalDataset

    dataset_sample1 = UniversalDataset(x=node_features, 

                            states=node_status, # e.g. additional information of each node, e.g. SIR states

                            y=targets, # prediction target

                            graph=static_graph, # adjacency matrix, we also support edge index: edge_index = ...

                            dynamic_graph=dynamic_graph # # adjacency matrix

                            )
    
    dataset_sample2 = UniversalDataset(x=features,y=node_target,graph=graph)



For more sample code in a real training process, you can refer to `examples/dataset_customization.ipynb` on the github page.








