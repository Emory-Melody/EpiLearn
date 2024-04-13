import torch
from torch_geometric.data import Data
import numpy as np
from .base import Dataset

class UniversalDataset(Dataset):
    def __init__(   self,
                    x = None,
                    states = None,
                    y = None,
                    edge_index = None,
                    edge_weight = None,
                    edge_attr = None,

                    l_size = None, # lookback window size
                    h_size = None # horizon size

                ):
        
        super().__init__()

        self.x = x # N*D; L*D; L*N*D; 
        self.y = y # N*1; L*1; L*N*1
        self.edge_index = edge_index # None; 2*Links; L*2*Links
        self.edge_weight = edge_weight # same as edge_index
        self.edge_attr = edge_attr # same as edge_index
        self.states = states # same as x

        self.l_size = l_size
        self.h_size = h_size

    def download(self):
        pass

    def save(self):
        pass

    def get_slice(self, timestamp):
        try:
            x = self.x[timestamp]
        except:
            x = None
        try:
            y= self.y[timestamp]
        except:
            y = None
        try:
            states = self.states[timestamp]
        except:
            states = None
        try:
            edge_index = self.edge_index[timestamp]
        except:
            edge_index = None
        try:
            edge_attr = self.edge_attr[timestamp]
        except:
            edge_attr = None
        return Data(x = x,
                    y = y,
                    edge_index = edge_index,
                    edge_attr = edge_attr,
                    states = states 
                    )
    
    def load_toy_dataset(self):
        data = torch.load("dreamy/data/toy.pt")
        self.x = data['feature']
        self.y = data['label']
        self.states = data['state']
        self.edge_index = data['edge_index']
        self.edge_weight = data['edge_weight']
        



