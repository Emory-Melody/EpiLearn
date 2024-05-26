import torch
import os
import matplotlib.pyplot as plt
#os.chdir("..")
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from epilearn.models.SpatialTemporal.STGCN import STGCN

from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.models.Spatial.GAT import GAT
from epilearn.models.Spatial.GIN import GIN

from epilearn.data import UniversalDataset
from epilearn.utils import utils, transforms
from epilearn.utils import simulation
from epilearn.tasks.detection import Detection


# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 13 # inputs size
horizon = 2 # predicts size; also seen as number of classes

epochs = 50 # training epochs
batch_size = 25 # training batch size



# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
task = Detection(prototype=GCN, dataset=None, lookback=lookback, horizon=horizon, device=device)



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




config = None
result = task.train_model(dataset=dataset, config=config, loss='ce', epochs=50) # instead of config, we can also dircetly input some parameters