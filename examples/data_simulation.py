#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import torch
from epilearn.models.Temporal.SIR import SIR, SEIR, SIS
from epilearn.models.SpatialTemporal.NetworkSIR import NetSIR
from epilearn.models.SpatialTemporal.DMP import DMP
from epilearn.models.Temporal.SIR import SIR
from epilearn.utils import utils, simulation
from epilearn.utils.simulation import Time_geo
from epilearn.data.dataset import UniversalDataset
from epilearn import visualize


# ### Generate random static graph

# In[2]:


# generate random static graph
initial_graph = simulation.get_random_graph(num_nodes=15, connect_prob=0.4)
initial_states = torch.zeros(15,3) # [S,I,R]
initial_states[:, 0] = 1
# set infected individual: 3
initial_states[3, 0] = 0
initial_states[3, 1] = 1
initial_states[4, 0] = 0
initial_states[4, 1] = 1


# ### Initialize NetworkSIR model and perform simulation based on the generated inital graph

# In[3]:


model = NetSIR(num_nodes=initial_graph.shape[0], horizon=120, infection_rate=0.05, recovery_rate=0.05) # infection_rate, recover_rate, fixed_population
preds = model(initial_states, initial_graph, steps = None)


# In[4]:


preds.argmax(2).shape


# ### Plot simulation result at a certain time point

# In[5]:


layout = visualize.plot_graph(preds.argmax(2)[15].detach().numpy(), initial_graph.to_sparse().indices().detach().numpy(), classes=['Suspected', 'Infected', 'Recovered'])


# In[6]:


import torch
import numpy as np
import seaborn as sb
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_graph(states: np.array, graph : np.array, classes=None, layout=None):
    """
    Plots a graph using NetworkX where nodes are colored based on their state and annotated with class labels.
    The graph's layout can be specified; otherwise, a spring layout is used.

    Parameters
    ----------
    states : np.array
        Array containing the state of each node used to determine the color of nodes.
    graph : np.array
        A 2D array where each column represents an edge between two nodes (start and end).
    classes : list, optional
        List of class labels corresponding to each node. Default is None.
    layout : dict, optional
        A dictionary defining the position of each node for custom layouts. If None, a spring layout is used. Default is None.

    Returns
    -------
    dict
        The positions of the nodes in the plot, useful for customizing the layout or for further graphical analysis.
    """
    graph = [(graph[0, i], graph[1, i]) for i in range(len(graph[0]))]

    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    g.add_edges_from(graph)
    color_map = {
        0:'red',
        1:'blue',
        2:'green',
        3:'yellow',
        4:'purple'
    }
    colors = []
    labels = {}
    for node in states:
        colors.append(color_map[node])
        labels[node] = classes[node]

    plt.figure()

    if layout is not None:
        nx.draw(g, pos=layout, node_color=colors, font_size=10, node_size=100)
    else:
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos = pos, node_color=colors, font_size=10, node_size=100)

    patches = []
    for i, item in enumerate(color_map.items()):
        if i>=len(classes):
            break
        patches.append(mpatches.Patch(color=item[1], label=classes[item[0]]))


    plt.legend(handles=patches, fontsize=16)
    plt.show()

    return pos


# In[7]:


plot_graph(preds.argmax(2)[15].detach().numpy(), initial_graph.to_sparse().indices().detach().numpy(), classes=['Suspected', 'Infected', 'Recovered'])


# ### Simulate using Dynamic Message Passing (DMP)

# In[8]:


recover = torch.rand(25)
dmp = DMP(num_nodes=25, recover_rate=recover)


# In[9]:


# DMP is under maintaining...
# dmp_simulation = dmp(None, initial_graph)


# ### Simulate with SIR (above individual-level)

# In[10]:


model = SIR(horizon=190, infection_rate=0.05, recovery_rate=0.05) # infection_rate, recover_rate, fixed_population
preds = model(initial_states.sum(0), steps = None)


# In[11]:


layout = visualize.plot_series(preds.detach().numpy(), columns=['Suspected', 'Infected', 'Recovered'])


# ### Construct graph from feature

# In[12]:


feature = torch.rand(10,20)
adj = torch.randint(10,100, (10,10))
graph1 = simulation.get_graph_from_features(features=feature, adj=None)
graph2 = simulation.get_graph_from_features(features=feature, adj=adj)


# In[13]:


graph1


# In[14]:


graph2


# ### Simulate trace with TimeGEO

# In[15]:


# prepare initial data
import numpy as np

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


# In[16]:


def gen_gps_data():
    """
    Generates a set of GPS coordinates centered around a base location with random variations.
    
    The function initializes with a fixed random seed to ensure the data is reproducible. It sets up
    a base latitude and longitude, and defines maximum allowable variations in these values. It then
    generates a specified number of GPS points within these variation limits.
    
    Returns:
        numpy.ndarray: An array of shape (num_gps_points, 2) where each row contains the latitude
                       and longitude of a generated GPS point.
    """

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Total number of GPS data points to generate
    num_gps_points = 4528

    # Base GPS coordinates around which the points will be generated
    base_latitude = 35.51168469
    base_longitude = 139.6733776

    # Maximum variation from the base coordinates
    latitude_variation = 0.0005  # Variation in latitude
    longitude_variation = 0.012  # Variation in longitude

    # Initialize an array to store the GPS data points
    gps_data = np.zeros((num_gps_points, 2))  # Each row will hold one latitude and one longitude

    # Generate random GPS data points
    for i in range(num_gps_points):
        random_latitude_shift = np.random.uniform(-latitude_variation, latitude_variation)
        random_longitude_shift = np.random.uniform(-longitude_variation, longitude_variation)
        gps_data[i][0] = base_latitude + random_latitude_shift  # Calculate the latitude
        gps_data[i][1] = base_longitude + random_longitude_shift  # Calculate the longitude

    return gps_data


# In[17]:


class Parameters:
    def __init__(self, data_type):
        self.data_type = data_type

    def data_info(self, GPS):
        self.GPS = GPS
        self.tim_size = 1440

data_type = 'Example Data'
param = Parameters(data_type)

gps_data = gen_gps_data()
param.data_info(gps_data)


# In[18]:


from tqdm import tqdm

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

def TimeGeo(data, param):
    TG = {}
    gen_bar = tqdm(data.items())
    for uid, trajs in gen_bar:
        gen_bar.set_description("TimeGeo - Generating trajectories for user: {}".format(uid))

        locations = np.sort(np.unique(np.concatenate([trajs[traj]['loc'] for traj in trajs])))
        trans = lambda x:np.where(locations == x)[0][0]
        detrans = lambda x:locations[x]

        input = np.array([to_fixed({'loc': list(map(trans, traj['loc'])), 'tim': traj['tim'], 'sta': traj['sta']}, param.tim_size, 10) for traj in trajs.values()])
        time_geo = Time_geo(param.GPS[np.ix_(locations)], np.bincount(input.flatten()) / np.cumprod(input.shape)[-1], simu_slot=param.tim_size//10)
        TG[uid] = {id: to_std(r['trace'][:, 0], param.tim_size, detrans) for id, r in enumerate(time_geo.pop_info)}
 
    return TG


# In[19]:


simulated_data = TimeGeo(data, param)

