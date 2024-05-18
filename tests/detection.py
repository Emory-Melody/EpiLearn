import torch
import os
import matplotlib.pyplot as plt

from epilearn.models.SpatialTemporal.STGCN import STGCN

from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.models.Spatial.GAT import GAT
from epilearn.models.Spatial.DCRNN import DCRNN
from epilearn.models.Spatial.GIN import GIN

from epilearn.data import UniversalDataset
from epilearn.utils import utils, transforms
from epilearn.tasks.detection import Detection

# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 1 # inputs size
horizon = 2 # predicts size; also seen as number of classes

epochs = 50 # training epochs
batch_size = 25 # training batch size

# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()


task = Detection(prototype=GCN, dataset=None, lookback=lookback, horizon=horizon, device='cpu')

config = None
result = task.train_model(dataset=dataset, config=config, loss='ce', epochs=5) # instead of config, we can also dircetly input some parameters



