#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import epilearn
from epilearn.data import UniversalDataset
from epilearn.models.Spatial import GCN


# In[2]:


dataset = UniversalDataset(name='Covid_Austria', root='./test_downloads/')


# In[3]:


dataset = UniversalDataset(name='Tycho_v1', root='./tmp_data/')


# In[4]:


dataset.features


# In[5]:


dataset = UniversalDataset()
dataset.load_toy_dataset()


# In[6]:


data = {"features": dataset.x, 'graph': dataset.graph, 'dynamic_graph': dataset.dynamic_graph.squeeze(), 'targets': dataset.y, 'states': dataset.states.argmax(2)}
torch.save(data, "interface_example.pt")


# In[7]:


print(dataset.x.shape)
print(dataset.y.shape)
print(dataset.states.shape)
print(dataset.edge_index.shape)
print(dataset.edge_weight.shape)





