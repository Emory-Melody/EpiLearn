import torch
from torch_geometric.data import Data
import numpy as np

from .base import Dataset

class UniversalDataset(Dataset):
    """
    UniversalDataset class is designed to handle various types of graph data,
    enabling operations on datasets that include features, states, dynamic graphs, and edge attributes.

    Parameters
    ----------
    x : torch.Tensor, optional
        Node features tensor of shape (num_samples, num_nodes, num_features). Represents the node features over multiple timesteps.
    states : torch.Tensor, optional
        Tensor representing various states of nodes, similar in structure to node features.
    y : torch.Tensor, optional
        Tensor representing target labels or values for each node, structured similar to node features.
    graph : torch.Tensor or scipy.sparse matrix, optional
        Static graph structure as an adjacency matrix.
    dynamic_graph : torch.Tensor, optional
        Dynamic graph information over time, providing evolving adjacency matrices.
    edge_index : torch.LongTensor, optional
        Tensor containing edge indices, typically of shape (2, num_edges), for defining which nodes are connected.
    edge_weight : torch.Tensor, optional
        Edge weights corresponding to the edge_index, providing the strength or capacity of connections.
    edge_attr : torch.Tensor, optional
        Attributes or features for each edge, aligned with the structure defined in edge_index.
    """
    def __init__(
                self,
                x=None, # timestep * Num nodes * features 
                states=None,
                y=None,
                graph=None,
                dynamic_graph=None,
                edge_index=None,
                edge_weight=None,
                edge_attr = None
                ):
        
        super().__init__()

        self.x = x # N*D; L*D; L*N*D; 
        self.y = y # N*1; L*1; L*N*1
        self.graph = graph
        self.dynamic_graph = dynamic_graph
        self.edge_index = edge_index # None; 2*Links; L*2*Links
        self.edge_weight = edge_weight # same as edge_index
        self.edge_attr = edge_attr # same as edge_index
        self.states = states # same as x
        self.output_dim = None

        self.transforms = None

        if self.y is not None:
            self.output_dim = self.y.shape
        
        if self.graph is not None and self.edge_index is None:
            sparse_adj = self.graph.to_sparse()
            self.edge_index = sparse_adj.indices()
            self.edge_weight = sparse_adj.values()
    
    def get_transformed(self, transformations=None):
        if transformations is None:
            if self.transforms is not None:
                transformations = self.transforms
            else:
                raise AttributeError("transformation does not exists!")
            
        input_data = {"features": self.x, 
                      "graph": self.graph, 
                      "dynamic_graph": self.dynamic_graph, 
                      "states": self.states
                    }
        transformed_data = self.transforms(input_data)

        return transformed_data['features'], transformed_data['graph'], transformed_data['dynamic_graph'], transformed_data['states']

        
    def __getitem__(self, index):
        batch_y = None if self.y is None else self.y[index].unsqueeze(0)
        batch_dynamic = None if self.dynamic_graph is None else self.dynamic_graph[index].unsqueeze(0)

        if self.y is not None:
            return Data(x=self.x[index], y=batch_y, edge_index=self.edge_index, edge_attr=self.edge_weight, dynamic_graph=batch_dynamic)
        else:
            return Data(x=self.x[index], edge_index=self.edge_index, edge_attr=self.edge_weight, dynamic_graph=batch_dynamic)
        
    def __len__(self):
        return len(self.x)

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
    


    

# class SpatialDataset(Dataset):
#     def __init__(   self,
#                     x = None,
#                     states = None,
#                     y = None,
#                     adj_m=None,
#                     edge_index = None,
#                     edge_attr = None
#                 ):
        
#         super().__init__()

#         self.x = x # N*D; L*D; L*N*D; 
#         self.y = y # N*1; L*1; L*N*1
#         self.adj_m = adj_m
#         self.edge_index = edge_index # None; 2*Links; L*2*Links
#         self.edge_attr = edge_attr # same as edge_index
#         self.states = states # same as x
#         self.output_dim = None
        
        
#         assert self.x is not None, "Input should not be NoneType!"
#         if self.y is not None:
#             assert len(self.x)==len(self.y), "Input and Output dim do not match!"
#             self.output_dim = self.y.shape
            
        
#         if self.edge_index is None and self.adj_m is None:
#             raise ValueError("There is no graph in Dataset, or you may specify your graph with parameter adj_m or edge_index!")
            
#         if self.adj_m is not None and self.edge_index is not None:
#             raise ValueError("There may be conflicts between your parameter adj_m and edge_index!")
        
#         if self.adj_m is not None and self.edge_index is None:
#             rows, cols = torch.where(self.adj_m != 0)
#             weights = self.adj_m[rows, cols]

#             edge_index = torch.stack([rows.long(), cols.long()], dim=0)
#             edge_weight = weights.clone().detach()
            
#             self.edge_index, self.edge_attr = edge_index, edge_weight
            
        
#         if self.adj_m is None and self.edge_index is not None:
#             num_nodes = self.x.shape[1] # or specify?
#             if self.edge_attr is None:
#                 self.edge_attr = torch.ones(self.edge_index.shape[1], 1)
#             if len(self.edge_attr.shape) == 1:
#                 self.edge_attr = torch.reshape(self.edge_attr, (self.edge_attr.shape[0], 1))
#             adj_matrix = torch.zeros((num_nodes, num_nodes, self.edge_attr.shape[1]))
#             for i in range(self.edge_index.shape[1]):
#                 src = self.edge_index[0, i]
#                 dest = self.edge_index[1, i]
#                 adj_matrix[src, dest] = self.edge_attr[i]
#             self.adj_m = adj_matrix.squeeze()
#             #print(self.adj_m.shape)
            
        
#     def __getitem__(self, index):
#         if self.y is not None:
#             return Data(x=self.x[index], y=self.y[index], edge_index=self.edge_index, edge_attr=self.edge_attr, adj_m=self.adj_m)
#         else:
#             return Data(x=self.x[index], edge_index=self.edge_index, edge_attr=self.edge_attr, adj_m=self.adj_m)
        
    
    
#     def __len__(self):
#         return len(self.x)
        



