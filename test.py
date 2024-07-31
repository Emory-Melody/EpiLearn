# import os
# import torch
# from epilearn.models.Temporal.SIR import SIR, SEIR, SIS
# from epilearn.models.SpatialTemporal.NetworkSIR import NetSIR
# from epilearn.models.SpatialTemporal.DMP import DMP
# from epilearn.models.Temporal.SIR import SIR
# from epilearn.utils import utils, simulation
# from epilearn.utils.simulation import Time_geo
# from epilearn.data.dataset import UniversalDataset
# from epilearn import visualize

# # generate random static graph
# initial_graph = simulation.get_random_graph(num_nodes=25, connect_prob=0.20)
# initial_states = torch.zeros(25,3) # [S,I,R]
# initial_states[:, 0] = 1
# # set infected individual: 3
# initial_states[3, 0] = 0
# initial_states[3, 1] = 1
# initial_states[10, 0] = 0
# initial_states[10, 1] = 1

# recover = torch.rand(25)
# dmp = DMP(num_nodes=25, recover_rate=recover)

# dmp_simulation = dmp(None, initial_graph)

import torch
from epilearn.models.Temporal.GRU import GRUModel
from epilearn.models.Temporal.LSTM import LSTMModel
from epilearn.models.Temporal.Dlinear import DlinearModel
from epilearn.models.Temporal.CNN import CNNModel
from epilearn.data import UniversalDataset
from epilearn.utils import transforms
from epilearn.tasks.forecast import Forecast
import urllib.request

# initialize settings
lookback = 8 # inputs size
horizon = 1 # predicts size
# url_data = "https://drive.google.com/uc?export=download&id=13b8_3OSAyIzTrM7lfWbhvxHlhWIiwCK5"
# urllib.request.urlretrieve(url_data, 'JHU_covid.pt')
jhu_data = torch.load('datasets/JHU_covid.pt')
print(jhu_data.keys())

print(jhu_data['feature_names'])
print(jhu_data['feature_names'][100])

data = jhu_data['features'][100].float().unsqueeze(1)
print('length: ', data.shape[0])

import matplotlib.pyplot as plt
# visualize the data starting from
plt.plot(data)
plt.xlabel('day')
plt.ylabel('confirmed cases')

dataset = UniversalDataset(x=data) # data shape should be length*1

# Adding Transformations
transformation = transforms.Compose({
                "features": [transforms.normalize_feat()]})
dataset.transforms = transformation

# Initialize Task
task = Forecast(prototype=CNNModel,  # add model
                lookback=lookback,  # history data length
                horizon=horizon,    # future data length
                ) 

# Training
result = task.train_model(dataset=dataset,
                          loss='mse',   # loss function; using MSE as default
                          epochs=5,    # training epochs
                          lr = 0.01,    # learning rate of the model
                          train_rate=0.6, # 60% is used for training
                          val_rate=0.2, # 20% is used for validation; the rest 20% is for testing
                          batch_size=5,
                          device='mps')

# Evaluation
evaluation = task.evaluate_model()

predictions, groundtruth = task.plot_forecasts(index_range=(0,-1))

# To use other popular temporal models, simply import and the change the prototypes in the forecast class
from epilearn.models.Temporal.LSTM import LSTMModel
from epilearn.models.Temporal.Dlinear import DlinearModel
from epilearn.models.Temporal.CNN import CNNModel

models = {"LSTM": LSTMModel, "DLinear": DlinearModel, "CNN": CNNModel}

for name, model in models.items():
  # Loading new prototype
  print(f"Using {name}")
  task.prototype = model
  # Training
  result = task.train_model(dataset=dataset,
                            loss='mse',   # loss function; using MSE as default
                            epochs=5,    # training epochs
                            lr = 0.01,    # learning rate of the model
                            train_rate=0.6, # 60% is used for training
                            val_rate=0.2, # 20% is used for validation; the rest 20% is for testing
                            batch_size=5,
                            device='mps')
  # Evaluation
  evaluation = task.evaluate_model()
  # visualization
  predictions, groundtruth = task.plot_forecasts(index_range=(0,-1))