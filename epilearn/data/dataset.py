import torch
from torch_geometric.data import Data
import numpy as np

from .base import Dataset

class UniversalDataset(Data):
    def __init__(
                self,
                x=None, # timestep * Num nodes * features 
                states=None,
                y=None,
                dynamic_graph=None,
                edge_index=None,
                edge_weight=None,
                edge_attr = None
                ):
        
        super().__init__()

        self.x = x # N*D; L*D; L*N*D; 
        self.y = y # N*1; L*1; L*N*1
        self.dynamic_graph = dynamic_graph
        self.edge_index = edge_index # None; 2*Links; L*2*Links
        self.edge_weight = edge_weight # same as edge_index
        self.edge_attr = edge_attr # same as edge_index
        self.states = states # same as x

    def download(self):
        pass

    def save(self):
        pass

    def generate_dataset(self, X = None, Y = None, states = None, dynamic_adj = None, lookback_window_size = 1, horizon_size = 1, permute = False, feat_idx = None, target_idx = None):
        """
        Takes node features for the graph and divides them into multiple samples
        along the time-axis by sliding a window of size (num_timesteps_input+
        num_timesteps_output) across it in steps of 1.
        :param X: Node features of shape (num_vertices, num_features,
        num_timesteps)
        :return:
            - Node features divided into multiple samples. Shape is
            (num_samples, num_vertices, num_features, num_timesteps_input).
            - Node targets for the samples. Shape is
            (num_samples, num_vertices, num_features, num_timesteps_output).
        """
        if X is None:
            X = self.x
        if Y is None:
            Y = self.y

        if feat_idx is not None:
            X = X[:, :, feat_idx]

        indices = [(i, i + (lookback_window_size + horizon_size)) for i in range(X.shape[0] - (lookback_window_size + horizon_size) + 1)]
        target = []
        for i, j in indices:
            target.append(Y[i + lookback_window_size: j])
        targets = torch.stack(target)
        if permute:
            targets = targets.transpose(1,2)

        input_list = [[None, X], [None, states], [None, dynamic_adj]]
        for m, inputs in enumerate(input_list):
            if inputs[1] is not None:
                tmp = []
                for i, j in indices:
                    tmp.append(inputs[1][i: i + lookback_window_size])
                input_list[m][0] = torch.stack(tmp)
                if permute:
                    inputs[0] = inputs[0].transpose(1,2)

        if target_idx is not None:
            targets = targets[:,target_idx, :]


        return input_list[0][0], targets, input_list[1][0], input_list[2][0]
        


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
        data1 = np.load("epilearn/data/graphs.npy")
        data2 = np.load("epilearn/data/features.npy", allow_pickle= True)

        self.x = torch.FloatTensor(data2.tolist()['node'])
        self.y = torch.FloatTensor(data2.tolist()['node'])[:,:,0]
       
        self.dynamic_graph = torch.FloatTensor(data2.tolist()['od'])
        self.states = torch.FloatTensor(data2.tolist()['SIR'])

        self.graph = data1
        self.edge_index = torch.FloatTensor(data1).to_sparse_coo().indices()
        self.edge_weight = torch.FloatTensor(data1).to_sparse_coo().values()
        



