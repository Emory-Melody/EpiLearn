#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import matplotlib.pyplot as plt
import os
import sys
# import ipdb; ipdb.set_trace()
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from epilearn.models.SpatialTemporal.STGCN import STGCN

from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.models.Spatial.GAT import GAT
from epilearn.models.Spatial.GIN import GIN

from epilearn.data import UniversalDataset
from epilearn.utils import utils, transforms
from epilearn.utils import simulation
from epilearn.tasks.detection import Detection


# ### Configs

# In[2]:


# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 1 # inputs size
horizon = 2 # predicts size; also seen as number of classes

epochs = 50 # training epochs
batch_size = 25 # training batch size


# ### Initialize Dataset

# In[3]:


# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()


# ### Initialize model and task

# In[4]:


task = Detection(prototype=GCN, dataset=None, lookback=lookback, horizon=horizon, device='cpu')


# ### Add transformations

# In[5]:


# transformation = transforms.Compose({
#                                  'features':[transforms.normalize_feat()], 
#                                  'graph': [transforms.normalize_adj()], 
#                                  'dynamic_graph': [transforms.normalize_adj()], 
#                                  'states': []
#                                  })
transformation = transforms.Compose({
                                 'features':[], 
                                 'graph': [], 
                                 'dynamic_graph': [], 
                                 'states': []
                                 })
dataset.transforms = transformation


# ### Train model

# In[6]:


config = None
result = task.train_model(dataset=dataset, config=config, loss='ce', epochs=5) # instead of config, we can also dircetly input some parameters


# ### Train on Simulated dataset

# In[7]:


# Simulation Process
from epilearn.models.SpatialTemporal.NetworkSIR import NetSIR

# generate 10 samples
num_nodes = 25
# generate random static graph: 25 nodes
initial_graph = simulation.get_random_graph(num_nodes=num_nodes, connect_prob=0.15)
initial_states = torch.zeros(num_nodes,3) # [S,I,R]
initial_states[:, 0] = 1

graph = initial_graph
x = []
y = []
for i in range(100): 
    # set infected individual
    idx = torch.randint(0,num_nodes, (1,))
    initial_states[idx.item(), 0] = 0
    initial_states[idx.item(), 1] = 1

    model = NetSIR(num_nodes=initial_graph.shape[0], horizon=100, infection_rate=0.01, recovery_rate=0.0384) # infection_rate, recover_rate, fixed_population
    preds = model(initial_states, initial_graph, steps = None)
    x.append(torch.nn.functional.one_hot(preds[-1].argmax(1)))
    y.append(initial_states.argmax(1))
x = torch.stack(x)
y = torch.stack(y)


# In[8]:


dataset = UniversalDataset(x=x,y=y,graph=initial_graph)
dataset.transforms = transformation
task = Detection(prototype=GCN, dataset=dataset, lookback=lookback, horizon=horizon, device='cpu')


# In[9]:


result = task.train_model(dataset=dataset, loss='ce', epochs=5)

