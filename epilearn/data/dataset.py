import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import numpy as np
import os
import urllib.request

from .base import Dataset

class UniversalDataset(Dataset):
    """
    UniversalDataset class is designed to handle various types of graph data,
    enabling operations on datasets that include features, states, dynamic graphs, and edge attributes.

    Parameters
    ----------
    name : str, optional
           Name of the deataset to be loaded (Supported dataset only).
    root : str, optional
           Location of the dataset to be downloaded (Supported dataset only).
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
                name=None,
                root='./',
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

        if name is not None:
            if not os.path.exists(root):
                    os.mkdir(root)
                    self.root = root
            if name == 'JHU_covid':
                if not os.path.exists(f"{root}/JHU_covid.pt"):
                    print("downloading JHU Covid Dataset")
                    url_feature = "https://drive.google.com/uc?export=download&id=13i9OpTweVYOvSOET-91-ZNVMJRURwxb0"
                    urllib.request.urlretrieve(url_feature, f"{root}/JHU_covid.pt")
                data = torch.load(f"{root}/JHU_covid.pt")
                for k, v in data.items():
                    if k=='features':
                        self.x = v.float()
                    if k=='feature_names':
                        self.features=v
                    if k=='index':
                        self.index=v

            elif name == 'Measles':
                if not os.path.exists(f"{root}/measles.pt"):
                    print("downloading Measles Dataset")
                    url_feature = "https://drive.google.com/uc?export=download&id=1-kLtvyUGN_mYJL5MgafEAd30KyfSmE4U"
                    urllib.request.urlretrieve(url_feature, f"{root}/measles.pt")

                data = torch.load(f"{root}/measles.pt")
                self.x = torch.FloatTensor(np.array(data['weekly_infection'])).T
                self.features = list(data['weekly_infection'].columns)
                self.anual_population = data['anual_population']
                self.anual_birth = data['anual_birth']
                self.coordinates = data['coordinates']

            elif name == 'Tycho_v1':
                if not os.path.exists(f"{root}/Tycho_v1.pt"):
                    print("downloading Tycho_v1 Dataset")
                    url_feature = "https://drive.google.com/uc?export=download&id=13gHDO6Rh5gZwqo8MLDyyiLtPCd9gD0nD"
                    urllib.request.urlretrieve(url_feature, f"{root}/Tycho_v1.pt")
                data = torch.load(f"{root}/Tycho_v1.pt")
                self.features = list(data.keys())
                self.x = []
                for v in data.values():
                    self.x.append(v.float())  

            elif name.split('_')[0] == 'Covid':
                country = name.split('_')[1]
                if not os.path.exists(f"{root}/covid_static.pt"):
                    print("downloading Covid Static Dataset") 
                    url_feature = "https://drive.google.com/uc?export=download&id=1-l1yVWlKxB0VLwj0IqJRAbruUvyrBUUI"
                    urllib.request.urlretrieve(url_feature, f"{root}/covid_static.pt")
                if not os.path.exists(f"{root}/covid_dynamic.pt"):
                    print("downloading Covid Dynamic Dataset") 
                    url_feature = "https://drive.google.com/uc?export=download&id=1-lUm1uaSDgbto2IV8aalZ3_lXh7LS7z3"
                    urllib.request.urlretrieve(url_feature, f"{root}/covid_dynamic.pt")

                try:
                    static_data = torch.load(f"{root}/covid_static.pt")
                    data = static_data[country]
                    self.graph = data['graph']
                except:
                    dynamic_data = torch.load(f"{root}/covid_dynamic.pt")
                    data = dynamic_data[country]
                    self.dynamic_graph = data['Dynamic_graph']

                self.x = data['features']
                self.features = data['feature_names']
                self.timestamp = data['time_stamp'] 
            else:
                raise ValueError("Dataset Not Found!")

        self.transforms = None

        if self.y is not None:
            self.output_dim = self.y.shape
        
        if self.graph is not None and self.edge_index is None:
            sparse_adj = torch.FloatTensor(self.graph).to_sparse()
            self.edge_index = sparse_adj.indices()
            self.edge_weight = sparse_adj.values()
    
    def get_transformed(self, input_data=None, transformations=None):
        if transformations is None:
            if self.transforms is not None:
                transformations = self.transforms
        if input_data is None:
            input_data = {"features": self.x, 
                        "target": self.y,
                        "graph": self.graph, 
                        "dynamic_graph": self.dynamic_graph, 
                        "states": self.states
                        }
        if transformations is not None:
            input_data = transformations(input_data)

        return input_data # input_data['features'], input_data['target'], input_data['graph'], input_data['dynamic_graph'], input_data['states']

    def get_split(self, inputs, idx1, idx2):
        if inputs is None:
            return None, None, None
        
        train_outputs = inputs[:idx1, ...]
        val_outputs = inputs[idx1:idx2, ...]
        test_outputs = inputs[idx2:, ...]

        return train_outputs, val_outputs, test_outputs
    
    def ganerate_splits(self, train_rate=0.6, val_rate=0.1):
        dataset = {
            "features": self.x,
            "target": self.y,
            "graph": self.graph,
            "dynamic_graph": self.dynamic_graph,
            "states": self.states
        }

        transformed_dataset = self.get_transformed(dataset)

        adj_static = self.graph

        split_line1 = int(self.x.shape[0] * train_rate)
        split_line2 = int(self.x.shape[0] * (train_rate + val_rate))

        train_features, val_features, test_features = self.get_split(transformed_dataset['features'], split_line1, split_line2)
        train_target, val_target, test_target = self.get_split(transformed_dataset['target'], split_line1, split_line2)
        train_states, val_states, test_states = self.get_split(transformed_dataset['states'], split_line1, split_line2)
        train_dynamic_adj, val_dynamic_adj, test_dynamic_adj = self.get_split(transformed_dataset['dynamic_graph'], split_line1, split_line2)
        
        train_dataset = {
            "features": train_features,
            "target": train_target,
            "graph": adj_static,
            "dynamic_graph": train_dynamic_adj,
            "states": train_states
        }
        val_dataset = {
            "features": val_features,
            "target": val_target,
            "graph": adj_static,
            "dynamic_graph": val_dynamic_adj,
            "states": val_states
        }
        test_dataset = {
            "features": test_features,
            "target": test_target,
            "graph": adj_static,
            "dynamic_graph": test_dynamic_adj,
            "states": test_states
        }

        return train_dataset, val_dataset, test_dataset

        
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

    def generate_dataset(self, X = None, Y = None, states = None, dynamic_adj = None, lookback_window_size = 1, horizon_size = 1, permute = False):
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

        indices = [(i, i + (lookback_window_size + horizon_size)) for i in range(X.shape[0] - (lookback_window_size + horizon_size) + 1)]
        target = []
        for i, j in indices:
            target.append(Y[i + lookback_window_size: j])
        
        targets = torch.stack(target) if len(target) > 0 else torch.Tensor([[[]]])
        if permute:
            targets = targets.transpose(1,2)

        input_list = [[None, X], [None, states], [None, dynamic_adj]]
        for m, inputs in enumerate(input_list):
            if inputs[1] is not None:
                tmp = []
                for i, j in indices:
                    tmp.append(inputs[1][i: i + lookback_window_size])
                input_list[m][0] = torch.stack(tmp) if len(tmp) else torch.Tensor([[[]]])
                if permute:
                    inputs[0] = inputs[0].transpose(1,2)

        return input_list[0][0], targets, input_list[1][0], input_list[2][0]

    
    def load_toy_dataset(self):
        if not os.path.exists("./datasets"):
            os.mkdir("./datasets/")
        if not os.path.exists("./datasets/features.npy"):
            print("downloading toy features")
            url_feature = "https://drive.google.com/uc?export=download&id=10VRjabU1m0pluQKOTQ-GKxF3sK9bLFYF"
            urllib.request.urlretrieve(url_feature, './datasets/features.npy')

        if not os.path.exists("./datasets/graphs.npy"):
            print("downloading toy graphs")
            url_graph = "https://drive.google.com/uc?export=download&id=10ZR4k19wdWXQdPN53Tz3QAQxBiZUXEPr"
            
            urllib.request.urlretrieve(url_graph, './datasets/graphs.npy')

        data1 = np.load(f"./datasets/graphs.npy")
        data2 = np.load(f"./datasets/features.npy",  allow_pickle= True)

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
        



