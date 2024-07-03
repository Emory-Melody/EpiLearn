Simulation
===================================

In this section, we provide a tutorial of the simulation methods in **EpiLearn**. In general, we focus on the simulation of static and dynamic properties including graph structure and node features.

Static Properties
------------------------------

Random Static Graph 
~~~~~~~~~~~~~~~~~~~~~~~~

A random static graph is a network where nodes are randomly connected with each other. This can be useful for simulating scenarios where connections between nodes do not change over time.

.. autoclass:: epilearn.tasks.detection.Detection
    :members:
    :undoc-members: 

 
When the function accepts inputs as number of nodes and connect probability, it will randomly generate a graph.

.. code-block:: python

    import epilearn as epi

    n_node = 25
    adj = epi.utils.simulation.get_random_graph(num_nodes=n_node, connect_prob=0.2, block_sizes=None, num_edges=None, graph_type='erdos_renyi')

    print(adj)


Static Features 
~~~~~~~~~~~~~~~~~~~~~~~~
Static features are fixed attributes assigned to nodes or edges in a graph. These features do not change over time.

Given static features, a graph can be obtained. The adj parameter is used to indicate the distance between nodes.

.. code-block:: python

    import epilearn as epi
    import torch

    feature = torch.rand(10,20)
    adj = torch.randint(10,100, (10,10))
    graph1 = epi.utils.simulation.get_graph_from_features(features=feature, adj=None)
    graph2 = epi.utils.simulation.get_graph_from_features(features=feature, adj=adj)


Gravity Model
~~~~~~~~~~~~~~~~~~~~~~~~
The gravity model is used to simulate the interaction between nodes based on their attributes and distance. It's often used in spatial analysis.

.. code-block:: python

    ...


Dynamic Properties
------------------------------
Mobility Simulation
~~~~~~~~~~~~~~~~~~~~~~~~
Mobility simulation models the movement of nodes over time, which can represent entities such as people or vehicles in a network.

.. code-block:: python

    TimeGeo?





SIR
~~~~~~~~~~~~~~~~~~~~~~~~
The SIR model is a simple mathematical model used to simulate the spread of a disease through a population.

.. code-block:: python

    import epilearn as epi

    model = epi.models.Temporal.SIR.SIR(horizon=190, infection_rate=0.05, recovery_rate=0.05)
    preds = model(initial_states.sum(0), steps = None)

    layout = epi.visualize.plot_series(preds.detach().numpy(), columns=['Suspected', 'Infected', 'Recovered'])


NetworkSIR
~~~~~~~~~~~~~~~~~~~~~~~~
The NetworkSIR model extends the SIR model by simulating the disease spread over a network, taking into account the network structure.

.. code-block:: python
    
    import epilearn as epi

    model = epi.models.SpatialTemporal.NetworkSIR.NetSIR(num_nodes=initial_graph.shape[0], horizon=120, infection_rate=0.05, recovery_rate=0.05)
    preds = model(initial_states, initial_graph, steps = None)

    layout = epi.visualize.plot_graph(preds.argmax(2)[15].detach().numpy(), initial_graph.to_sparse().indices().detach().numpy(), classes=['Suspected', 'Infected', 'Recovered'])
