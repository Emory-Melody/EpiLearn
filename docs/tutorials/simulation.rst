Simulation
===================================

In this section, we provide a tutorial of the simulation methods in **EpiLearn**. In general, we focus on the simulation of static and dynamic properties including graph structure and node features.

Static Properties
------------------------------

Random Static Graph 
~~~~~~~~~~~~~~~~~~~~~~~~

A random static graph is a network where nodes are randomly connected with each other. This can be useful for simulating scenarios where connections between nodes do not change over time.

.. autofunction:: epilearn.utils.simulation.get_random_graph

 
When the function accepts inputs as number of nodes and connect probability, it will randomly generate a graph.

.. code-block:: python

    import epilearn as epi

    adj = epi.utils.simulation.get_random_graph(num_nodes=25, connect_prob=0.2, block_sizes=None, num_edges=None, graph_type='erdos_renyi')


Static Features 
~~~~~~~~~~~~~~~~~~~~~~~~
Static features are fixed attributes assigned to nodes or edges in a graph. These features do not change over time.

.. autofunction:: epilearn.utils.simulation.get_graph_from_features


Given static node features, a graph can be obtained by calculating cosine similarity between each nodes. The adj parameter is used to indicate the distance between nodes, adding panelty on similarities.

.. code-block:: python

    import epilearn as epi
    import torch

    feature = torch.rand(10,20) # Randomly generate a 10x20 feature matrix. 10 is the number of node and 20 is the feature dimension.
    adj = torch.randint(10,100, (10,10)) # Randomly generate a adjacency matrix by 10 nodes.

    # The feature, adj and any other parameters can be replaced by your own.
    graph1 = epi.utils.simulation.get_graph_from_features(features=feature, adj=None)
    graph2 = epi.utils.simulation.get_graph_from_features(features=feature, adj=adj) # If providing adj, then the similarity will be divided by distance in adj repectively.


Gravity Model
~~~~~~~~~~~~~~~~~~~~~~~~
The gravity model is used to simulate the interaction between nodes based on their attributes and distance. It's often used in spatial analysis. For example, in epidemic, it can be used to capture the regional contact and transmission patterns invoked
by human mobility. The Gravity_model class follows the `equation <https://dl.acm.org/doi/abs/10.1145/3589132.3625596>`_ as shown in annotations. For parameter settings, follow the `empirical parameters <https://www.pnas.org/doi/epdf/10.1073/pnas.0906910106>`_:

.. list-table:: Empirical Parameters
   :widths: 25 25 50
   :header-rows: 1

   * - distance (km)
     - source
     - target
     - s (km)
   * - ≤ 300
     - 0.46 ± 0.01
     - 0.64 ± 0.01
     - 82 ± 2
   * - > 300
     - 0.35 ± 0.06
     - 0.37 ± 0.06
     - N/A


.. autoclass:: epilearn.utils.simulation.Gravity_model
    :members:
    :undoc-members:  
    :special-members: __init__


Given population numbers in each node (or say region) and distance between each node, edge weights can be obtained along with given each parameter.

.. code-block:: python

    import epilearn as epi

    node_populations = torch.tensor([1000, 2000, 1500]) # Assume we have three nodes, and the populations for the three nodes.
    distance_graph = torch.tensor([
        [0, 10, 20],
        [10, 0, 15],
        [20, 15, 0]
    ]) # Assume the distance between each node.

    # Create gravity model by giving the three parameters 
    gravity_model = epi.utils.simulation.Gravity_model(source=0.46, target=0.64, s=82)

    # Choose two nodes from the graph along with their distance, and get the influence between this node pair.
    source_population = 1000
    target_population = 2000
    distance = 10
    influence = gravity_model.get_influence(source_population, target_population, distance)
    
    # Use population for each node and the distances to generate a set of graph weight, forming an adjency matrix.
    mobility_graph = gravity_model.get_mobility_graph(node_populations, distance_graph)
    


Dynamic Properties
------------------------------
Mobility Simulation
~~~~~~~~~~~~~~~~~~~~~~~~
Mobility simulation models the movement of nodes over time, which can represent entities such as people or vehicles in a network.

.. autoclass:: epilearn.utils.simulation.TimeGeo
    :members:
    :undoc-members: 


First, we randomly generate a set of GPS coordinates. These coordinates will serve as the regions within which individuals will move. The generated GPS data simulates the geographic regions in which our individuals will perform their activities. This is essential for the Time_geo class to simulate realistic movements based on geographic locations.

.. code-block:: python

    import numpy as np

    np.random.seed(42)
    num_gps_points = 4528
    base_latitude = 35.51168469
    base_longitude = 139.6733776
    latitude_variation = 0.0005
    longitude_variation = 0.012

    gps_data = np.zeros((num_gps_points, 2))
    for i in range(num_gps_points):
        random_latitude_shift = np.random.uniform(-latitude_variation, latitude_variation)
        random_longitude_shift = np.random.uniform(-longitude_variation, longitude_variation)
        gps_data[i][0] = base_latitude + random_latitude_shift
        gps_data[i][1] = base_longitude + random_longitude_shift


Next we define several helper functions to process the trajectories of individuals:

1. **padding**: Pads the trajectory data to ensure consistent time intervals.
2. **fixed**: Aggregates the padded trajectory data into fixed time slots.
3. **to_fixed**: Converts trajectories to a fixed time resolution.
4. **to_std**: Standardizes the trajectories into a specific format.

.. code-block:: python

    def padding(traj, tim_size):
        def intcount(seq):
            a, b = np.array(seq[:-1]), np.array(seq[1:])
            return (a == a.astype(int)) + np.ceil(b) - np.floor(a) - 1
        locs = np.concatenate(([-1], traj['loc'], [-1]))
        tims = np.concatenate(([0], traj['tim'] % tim_size, [tim_size]))
        tims[-2] = tims[-1] if (tims[-2] < tims[-3]) else tims[-2]
        return np.concatenate([[locs[id]] * int(n) for id, n in enumerate(intcount(tims))]).astype(int)

    def fixed(pad_traj, slot = 30):
        return np.array([np.argmax(np.bincount((pad_traj + 1)[(slot*i):(slot*i+slot)])) - 1 for i in range(int(len(pad_traj)/slot))])

    def to_fixed(traj, tim_size, slot = 30):
        a = fixed(padding(traj, tim_size), slot)
        return np.where(a==-1, a[-1], a)

    def to_std(traj, tim_size, detrans, time_slot=10):
        id = np.append(True, traj[1:] != traj[:-1])
        loc, tim = np.array(list(map(detrans,  traj[id]))), np.arange(0, tim_size, time_slot)[id]
        sta = np.append(tim[1:], tim_size) - tim
        return {'loc': loc, 'tim':tim, 'sta': sta}


Define a TimeGeo function utilizes the Time_geo class to simulate trajectories based on the input data and parameters.

.. code-block:: python

    from tqdm import tqdm

    def TimeGeo(data, param):
        TG = {}
        gen_bar = tqdm(data.items())
        for uid, trajs in gen_bar:
            gen_bar.set_description("TimeGeo - Generating trajectories for user: {}".format(uid))

            locations = np.sort(np.unique(np.concatenate([trajs[traj]['loc'] for traj in trajs])))
            trans = lambda x:np.where(locations == x)[0][0]
            detrans = lambda x:locations[x]

            input = np.array([to_fixed({'loc': list(map(trans, traj['loc'])), 'tim': traj['tim'], 'sta': traj['sta']}, param.tim_size, 10) for traj in trajs.values()])
            time_geo = epi.utils.simulation.Time_geo(param.GPS[np.ix_(locations)], np.bincount(input.flatten()) / np.cumprod(input.shape)[-1], simu_slot=param.tim_size//10)
            TG[uid] = {id: to_std(r['trace'][:, 0], param.tim_size, detrans) for id, r in enumerate(time_geo.pop_info)}
    
        return TG


Define Parameters class holds the data type and GPS information needed for the simulation.

.. code-block:: python

    class Parameters:
        def __init__(self, data_type):
            self.data_type = data_type

        def data_info(self, GPS):
            self.GPS = GPS
            self.tim_size = 1440

    data_type = 'Example Data'
    param = Parameters(data_type)

    param.data_info(gps_data)


Finally define a sample data set and run the TimeGeo function to simulate the movement patterns.

.. code-block:: python

    data = {
        1: {
            0: {
                'loc': np.array([3744, 1901, 1995, 2653, 2743, 2744, 2654, 2743, 2745], dtype=int),
                'tim': np.array([6075.83333333, 6113.5, 6503.81666667, 6518.23333333, 6568.8, 6640.73333333, 6664.25, 6708.61666667, 6747.83333333], dtype=np.float64),
                'sta': np.array([37.66666667, 390.31666667, 14.41666667, 50.56666667, 71.93333333, 23.51666667, 44.36666667, 39.21666667, 99.75], dtype=np.float64)
            },
            1: {
                'loc': np.array([2653, 2744, 2654, 2653, 2654, 2653], dtype=int),
                'tim': np.array([297580.38333333, 297645.8, 297661.83333333, 297682.41666667, 297693.05, 297788.05], dtype=np.float64),
                'sta': np.array([65.41666667, 16.03333333, 20.58333333, 10.63333333, 95.0, 10.58333333], dtype=np.float64)
            }
        },
        3: {
            0: {
                'loc': np.array([2244, 2173, 2248, 2395, 2547, 2477, 1853], dtype=int),
                'tim': np.array([11740.61666667, 11757.53333333, 11778.65, 11819.31666667, 11833.56666667, 12035.68333333, 12064.13333333], dtype=np.float64),
                'sta': np.array([16.91666667, 21.11666667, 40.66666667, 14.25, 202.11666667, 28.45, 1383.16666667], dtype=np.float64)
            }
        },
        5: {
            0: {
                'loc': np.array([2741, 2654, 2653, 2743, 2341], dtype=int),
                'tim': np.array([89855.71666667, 89881.15, 89909.1, 90020.8, 90216.95], dtype=np.float64),
                'sta': np.array([25.43333333, 27.95, 111.7, 196.15, 34.91666667], dtype=np.float64)
            },
            1: {
                'loc': np.array([2653, 2744, 2653, 2823, 2744], dtype=int),
                'tim': np.array([236819.35, 236892.11666667, 236908.31666667, 236923.75, 237041.66666667], dtype=np.float64),
                'sta': np.array([72.76666667, 16.2, 15.43333333, 117.91666667, 32.93333333], dtype=np.float64)
            },
            2: {
                'loc': np.array([2653, 2652, 2744, 2653, 2743], dtype=int),
                'tim': np.array([75474.86666667, 75494.3, 75563.6, 75598.66666667, 75629.6], dtype=np.float64),
                'sta': np.array([19.43333333, 69.3, 35.06666667, 30.93333333, 142.6], dtype=np.float64)
            }
        }
    }

    simulated_data = TimeGeo(data, param)



SIR
~~~~~~~~~~~~~~~~~~~~~~~~
The SIR model is a simple mathematical model used to simulate the spread of a disease through a population. This class has three vital initial parameters:

1. horizon: The total number of time steps for the simulation.
2. infection_rate: The rate at which susceptible individuals get infected.
3. recovery_rate: The rate at which infected individuals recover.

.. code-block:: python

    import epilearn as epi

    # Generate random static graph
    initial_graph = simulation.get_random_graph(num_nodes=25, connect_prob=0.20)
    initial_states = torch.zeros(25,3) # [S,I,R]
    initial_states[:, 0] = 1
    # Set infected individual: 3
    initial_states[3, 0] = 0
    initial_states[3, 1] = 1
    initial_states[10, 0] = 0
    initial_states[10, 1] = 1

    # Create an instance of the SIR model with specified parameters.
    model = epi.models.Temporal.SIR.SIR(horizon=190, infection_rate=0.05, recovery_rate=0.05)

    # Run the model to generate predictions.
    # Steps control the number of steps for the simulation. If None, it runs for the full horizon.
    preds = model(initial_states.sum(0), steps = None)

    # Plot the simulation.
    layout = epi.visualize.plot_series(preds.detach().numpy(), columns=['Suspected', 'Infected', 'Recovered'])


NetworkSIR
~~~~~~~~~~~~~~~~~~~~~~~~
The NetworkSIR model extends the SIR model by simulating the disease spread over a network, taking into account the network structure. This class has four vital initial parameters:

1. num_nodes: The number of nodes in the network.
2. horizon: The total number of time steps for the simulation.
3. infection_rate: The rate at which susceptible nodes get infected.
4. recovery_rate: The rate at which infected nodes recover.

.. code-block:: python

    import epilearn as epi

    # Generate random static graph
    initial_graph = simulation.get_random_graph(num_nodes=25, connect_prob=0.20)
    initial_states = torch.zeros(25,3) # [S,I,R]
    initial_states[:, 0] = 1
    # Set infected individual: 3
    initial_states[3, 0] = 0
    initial_states[3, 1] = 1
    initial_states[10, 0] = 0
    initial_states[10, 1] = 1

    # Create an instance of the NetworkSIR model with specified parameters:
    model = epi.models.SpatialTemporal.NetworkSIR.NetSIR(num_nodes=initial_graph.shape[0], horizon=120, infection_rate=0.05, recovery_rate=0.05)
    
    # Run the model to generate predictions.
    # Steps control the number of steps for the simulation. If None, it runs for the full horizon.
    preds = model(initial_states, initial_graph, steps = None)

    # Plot the simulation.
    layout = epi.visualize.plot_graph(preds.argmax(2)[15].detach().numpy(), initial_graph.to_sparse().indices().detach().numpy(), classes=['Suspected', 'Infected', 'Recovered'])
