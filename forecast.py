import torch
import os
import matplotlib.pyplot as plt

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.MepoGNN import MepoGNN
from epilearn.models.SpatialTemporal.EpiGNN import EpiGNN
from epilearn.models.SpatialTemporal.DASTGN import DASTGN
from epilearn.models.SpatialTemporal.ColaGNN import ColaGNN
from epilearn.models.SpatialTemporal.EpiColaGNN import EpiColaGNN
from epilearn.models.SpatialTemporal.CNNRNN_Res import CNNRNN_Res
from epilearn.models.SpatialTemporal.ATMGNN import ATMGNN

from epilearn.data import UniversalDataset
from epilearn.utils import utils, transforms, metrics
from epilearn.tasks.forecast import Forecast

device = torch.device('cpu')
torch.manual_seed(7)

# initialize configs
lookback = 12 # inputs size
horizon = 1 # predicts size
# permutation is True when using STGCN
permute = True
epochs = 50 # training epochs
batch_size = 50 # training batch size

task = Forecast(prototype=STGCN, dataset=None, lookback=lookback, horizon=horizon, device='cpu')

config = None
# for epicolagnn, loss='epi_cola', else loss='mse
# for STGCN, permute_dataset=True

# load toy dataset
datasets = []
dataset0 = UniversalDataset()
dataset0.load_toy_dataset()

raw_data = torch.load("datasets/benchmark.pt")
for name in ['Brazil', 'Austria', 'China']:
    data = raw_data[name]
    dataset = UniversalDataset()
    dataset.x = data['features']
    dataset.y = data['features'][:,:,0]
    dataset.graph = data['graph']
    dataset.states = data['features']
    dataset.dynamic_graph = None
    datasets.append(dataset)
datasets.append(dataset0)
for i, dataset in enumerate(datasets):
    print(f"dataset {i}")
    model = task.train_model(dataset=dataset, config=config, loss='mse', epochs=5, permute_dataset=True) # instead of config, we can also dircetly input some parameters