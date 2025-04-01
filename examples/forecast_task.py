#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import matplotlib.pyplot as plt
import sys
# import ipdb; ipdb.set_trace()
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.MepoGNN import MepoGNN
from epilearn.models.SpatialTemporal.EpiGNN import EpiGNN
from epilearn.models.SpatialTemporal.DASTGN import DASTGN
from epilearn.models.SpatialTemporal.ColaGNN import ColaGNN
from epilearn.models.SpatialTemporal.EpiColaGNN import EpiColaGNN
from epilearn.models.SpatialTemporal.CNNRNN_Res import CNNRNN_Res
from epilearn.models.SpatialTemporal.ATMGNN import MPNN_LSTM, ATMGNN

from epilearn.models.Temporal.Dlinear import DlinearModel
from epilearn.models.Temporal.LSTM import LSTMModel
from epilearn.models.Temporal.GRU import GRUModel

from epilearn.data import UniversalDataset
from epilearn.utils import utils, transforms
from epilearn.tasks.forecast import Forecast


# ### Configs

# In[2]:


# initial settings
device = torch.device('cuda')
torch.manual_seed(7)

lookback = 12 # inputs size
horizon = 3 # predicts size

# permutation is True when using STGCN
permute = False

epochs = 10 # training epochs
batch_size = 50 # training batch size


# ### Initialize dataset

# In[3]:


# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()


# ### Initialize model and task
# * prototype supports all models imported at the first cell

# In[4]:


task = Forecast(prototype=EpiGNN, dataset=None, lookback=lookback, horizon=horizon)


# ### Add transformations to dataset

# In[5]:


transformation = transforms.Compose({"features":[transforms.normalize_feat()], 
                                 'graph': [transforms.normalize_adj()], 
                                 'dynamic_graph': [transforms.normalize_adj()], 
                                 'states': []
                                 })
dataset.transforms = transformation


# ### Train Model
# * for epicolagnn, loss='epi_cola' else loss='mse
# * for STGCN, permute_dataset=True

# In[6]:

model_args = {"num_nodes": dataset.x.shape[1], "num_features": 4, "num_timesteps_input": lookback, "num_timesteps_output":horizon, "nhids": 16, "device": device}

result = task.train_model(dataset=dataset,
                          loss='mse',
                          epochs=5,
                          batch_size=5,
                          train_rate=0.6,
                          val_rate=0.2,
                          lr=1e-3,
                          permute_dataset=permute,
                          model_args=model_args,
                          verbose=True,
                          device=device
                          )


# ### Evaluate model

# In[7]:


evaluation = task.evaluate_model()


# ### Try more datasets

# In[8]:


# load other datasets
datasets = [dataset]
raw_data = torch.load("datasets/covid_static.pt")
for name in ['Brazil', 'Austria', 'China']:
    data = raw_data[name]
    dataset = UniversalDataset()
    dataset.x = data['features']
    dataset.y = data['features'][:,:,0]
    dataset.graph = data['graph']
    dataset.states = data['features']
    dataset.dynamic_graph = None

    dataset.transforms = transformation
    datasets.append(dataset)


# In[9]:


for i, dataset in enumerate(datasets):
    print(f"dataset {i}")
    model = task.train_model(dataset=dataset, config=config, loss='mse', epochs=50, batch_size=50, permute_dataset=permute) # instead of config, we can also dircetly input some parameters


# ### Try temporal models

# In[10]:


task = Forecast(prototype=LSTMModel, dataset=None, lookback=lookback, horizon=horizon, device='cpu')
num_nodes = 47
mae_list=[]
rmse_list=[]
for region in range(num_nodes):
    print("region", region)
    result = task.train_model(dataset=datasets[-1], config=config, loss='mse', epochs=50, batch_size=50, region_idx=1, permute_dataset=False)
    mae_list.append(result['mae'])
    rmse_list.append(result['rmse'])

mae = torch.FloatTensor(mae_list)
rmse = torch.FloatTensor(rmse_list)
print(f"mae:{mae.mean()} {mae.std()}")
print(f"rmse:{rmse.mean()} {rmse.std()}")

