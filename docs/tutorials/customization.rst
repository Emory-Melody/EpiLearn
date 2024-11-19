Model&Dataset Customization
===================================

Dataset Customization
-----------------------

EpiLearn povides UniversalDataset class to load all datasets, including time series data, static graph, and dynamic graphs.

For temporal tasks, we need at least two inputs: time series features (**[Length, Channels]**) and prediction target (**[Length, 1]**). \ 
Note that the prediction target can be inlcuded as one of the channels of the time series features. If you only have univariate time series data, then the input x and y will be the **same**.

.. code-block:: python

    dataset = UniversalDataset(x=features, y=targets)

For spatial-temporal tasks, we also load static graph and dynamic graph. In this case, the shapes of inputs change to x: **[Length, num_nodes, channels]**; \
y: **[Length, num_nodes]**; graph: **[num_nodes, num_nodes]**; dynamic_graph: **[Length, num_nodes, num_nodes]**.

.. code-block:: python

    dataset = UniversalDataset(x=node_features, y=targets, graph=static_graph, dynamic_graph=dynamic_graph)

For spatial tasks, we are using a period of future states to predict a certain point of history states of the nodes. In this case, In this case, the shapes of inputs change to \ 
x: **[num_samples, num_nodes, channels]**; y: **[num_samples, num_nodes]**; graph: **[num_nodes, num_nodes]**; dynamic_graph: **[num_samples, num_nodes, num_nodes]**. Note that \ 
although the inputs shapes are the same as the spatial-temporal task, the meaning of the first dim is different, denoting number of samples instead of the length the time series. \ 
This means you can input multiple same(graph) or different(dynamic_graph) graph samples.

.. code-block:: python
    
    dataset = UniversalDataset(x=node_features, y=targets, graph=same_graph_each_sample, dynamic_graph=defferent_graph_each_sample)

For more coding details, please refer to `Dataset Customization <https://github.com/Emory-Melody/EpiLearn/blob/main/examples/dataset_customization.ipynb>`_.


Model Customization
-----------------------






