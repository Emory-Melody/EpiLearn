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
from epilearn.utils import utils, transforms
from epilearn.tasks.forecast import Forecast



# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 12 # inputs size
horizon = 3 # predicts size

# permutation is True when using STGCN
permute = True

epochs = 50 # training epochs
batch_size = 50 # training batch size

# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()

task = Forecast(prototype=GraphLSTM, dataset=None, lookback=lookback, horizon=horizon, device='cpu')


config = None
# for epicolagnn, loss='epi_cola', else loss='mse
# for STGCN, permute_dataset=True
model = task.train_model(dataset=dataset, config=config, loss='mse', epochs=5, permute_dataset=False) # instead of config, we can also dircetly input some parameters