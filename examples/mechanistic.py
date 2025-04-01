#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("..")
import torch
from epilearn.models.Temporal.SIR import SIR, SEIR, SIS
from epilearn.models.SpatialTemporal.NetworkSIR import NetSIR
from epilearn import visualize


# In[2]:


data = torch.zeros(3)
# data.data[0] = torch.randint(1000, 10000, (1,))
# data.data[1] = torch.randint(100, 1000, (1,))
# data.data[2] = torch.randint(10, 100, (1,))
data.data[0] = 3416
data.data[1] = 210
data.data[2] = 65
data = data.float()
print(data)
model = SIR(horizon=100, infection_rate=0.1, recovery_rate=0.0384) # infection_rate, recover_rate, fixed_population


# In[3]:


preds = model(data, steps = None) # steps = None or horizon
visualize.plot_series(preds, columns = ['suspected', 'infected', 'recovered'])


# In[ ]:




