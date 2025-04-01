#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import matplotlib.pyplot as plt
os.chdir("..")
os.sys.path.append(os.path.join(os.path.abspath(''), '..'))

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
dataset.x.shape


# ### Initialize model and task

# In[4]:


task = Detection(prototype=GCN, dataset=None, lookback=lookback, horizon=horizon, device='cpu')


# ### Add transformations

# In[5]:


transformation = transforms.Compose({
                                 'features':[transforms.add_time_embedding(embedding_dim=4, fourier=False),
                                             transforms.convert_to_frequency(ftype="fft"),
                                             transforms.normalize_feat()], 
                                 'graph': [transforms.normalize_adj()], 
                                 'dynamic_graph': [transforms.normalize_adj()], 
                                 'states': []
                                 })

'''transformation = transforms.Compose({
                                 'features':[], 
                                 'graph': [], 
                                 'dynamic_graph': [], 
                                 'states': []
                                 })'''
dataset.transforms = transformation


# ### Train model

# In[6]:


config = None
result = task.train_model(dataset=dataset, config=config, loss='ce', epochs=5) # instead of config, we can also dircetly input some parameters

