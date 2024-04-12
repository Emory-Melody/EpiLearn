import torch
from torch_geometric.data import Data
from .base import Dataset

class UniversalDataset(Dataset):
    def __init__(   self,
                    x = None,
                    states = None,
                    y = None,
                    edge_index = None,
                    edge_attr = None,

                    lb_size = None, # look back window size
                    fp_size = None # forward prediction size

                ):
        

        super().__init__(   x = x,
                            y = y,
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            states = states
                        )

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