import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import numpy as np
import os
import urllib.request

import pandas as pd
import warnings


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
                data = torch.load(f"{root}/JHU_covid.pt", weights_only=False)
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

                data = torch.load(f"{root}/measles.pt", weights_only=False)
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
                data = torch.load(f"{root}/Tycho_v1.pt", weights_only=False)
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
            sparse_adj = torch.Tensor(self.graph).float().to_sparse()
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
        # import ipdb; ipdb.set_trace()
        adj_static = transformed_dataset['graph'] if transformed_dataset['graph'] is not None else self.graph

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

    def generate_dataset(self, X = None, Y = None, states = None, dynamic_adj = None, lookback_window_size = 1, horizon_size = 1, ahead=0, permute = False):
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
        # import ipdb; ipdb.set_trace()
        indices = [(i, i + (lookback_window_size + ahead +horizon_size)) for i in range(X.shape[0] - (lookback_window_size + ahead + horizon_size) + 1)]
        target = []
        for i, j in indices:
            target.append(Y[i + lookback_window_size+ahead: j])
        
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

        self.graph = torch.FloatTensor(data1)
        self.edge_index = torch.FloatTensor(data1).to_sparse_coo().indices()
        self.edge_weight = torch.FloatTensor(data1).to_sparse_coo().values()
    @classmethod
    def from_csv(
        cls,
        feature_csv: str,
        node_id_col: str,
        time_col: str,
        feature_cols: list,
        target_cols: list | None = None,
        edge_csv: str | None = None,
        source_col: str = "source",
        target_col: str = "target",
        strict_numeric: bool = True,
    ):
        """
        Load dataset from CSV files and build a UniversalDataset without changing existing behaviors.

        Args:
            feature_csv: Path to the CSV containing time series features/targets. Must include
                `time_col`, `node_id_col`, `feature_cols`, and optionally `target_cols`.
            node_id_col: Column name for node identifiers.
            time_col: Column name for timestamps (sorted ascending).
            feature_cols: List of feature column names (numeric).
            target_cols: Optional list of target column names (numeric).
            edge_csv: Optional path to an edges CSV with two columns: `source_col`, `target_col`.
            source_col, target_col: Column names for edges CSV.
            strict_numeric: If True, raise on any non-numeric entries in features/targets.
                            If False, warn and keep NaNs.

        Returns:
            UniversalDataset(x=[T,N,F], y=[T,N] or [T,N,Ty], graph=[N,N], edge_index=[2,E])
        """
        # --- 1) Read and column check
        df = pd.read_csv(feature_csv)
        must_have = [node_id_col, time_col] + list(feature_cols) + (list(target_cols) if target_cols else [])
        miss = [c for c in must_have if c not in df.columns]
        if miss:
            raise ValueError(f"CSV is missing required columns: {miss}")

        # --- 2) Numeric check: accept numbers only
        issue = []
        for c in feature_cols + (target_cols or []):
            before = df[c].copy()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            bad = df[c].isna() & before.notna()
            if bad.any():
                issue.append((c, int(bad.sum())))
        if issue:
            msg = "Found non-numeric entries: " + ", ".join([f"{c}({n})" for c, n in issue])
            if strict_numeric:
                raise ValueError(msg + " Please ensure these columns are purely numeric.")
            else:
                warnings.warn(msg + " NaNs will be kept.")

        # --- 3) Drop rows with all-NaN in key columns (avoid pivot failure); other missingness controlled by strict_numeric
        key_cols = list(feature_cols) + (list(target_cols) if target_cols else [])
        df = df.dropna(subset=key_cols, how="all")

        # Sort & stable indices
        times = sorted(df[time_col].unique().tolist())
        nodes = sorted(df[node_id_col].unique().tolist())
        node2idx = {n: i for i, n in enumerate(nodes)}

        # --- 4) Pivot features -> x:[T,N,F]
        piv_feat = df.pivot_table(index=time_col, columns=node_id_col, values=feature_cols)
        # Complete grid for MultiIndex (feature, node) to ensure stable column order
        col_index = pd.MultiIndex.from_product([feature_cols, nodes])
        piv_feat = piv_feat.reindex(index=times, columns=col_index)
        # If NaNs remain (e.g., some time-node-feature combos missing), handle per strict_numeric
        if piv_feat.isna().any().any():
            nan_cnt = int(piv_feat.isna().sum().sum())
            msg = f"Missing data in {nan_cnt} cells (some time-node-feature combinations are missing)."
            if strict_numeric:
                raise ValueError(msg + " Please fill in and try again.")
            else:
                warnings.warn(msg + " Will keep as NaN; subsequent modeling/processing may fail.")

        T = len(times)
        F = len(feature_cols)
        N = len(nodes)
        x_np = piv_feat.values.reshape(T, F, N).transpose(0, 2, 1)  # [T,N,F]
        x = torch.tensor(x_np, dtype=torch.float32)

        # --- 5) Pivot targets -> y:[T,N] or [T,N,Ty]
        y = None
        if target_cols:
            piv_y = df.pivot_table(index=time_col, columns=node_id_col, values=target_cols)
            col_index_y = pd.MultiIndex.from_product([target_cols, nodes])
            piv_y = piv_y.reindex(index=times, columns=col_index_y)
            if piv_y.isna().any().any():
                nan_cnt = int(piv_y.isna().sum().sum())
                msg = f"Target columns have {nan_cnt} missing cells."
                if strict_numeric:
                    raise ValueError(msg + " Please fill in and try again.")
                else:
                    warnings.warn(msg + " Will keep as NaN.")
            Ty = len(target_cols)
            y_np = piv_y.values.reshape(T, Ty, N).transpose(0, 2, 1)  # [T,N,Ty]
            if Ty == 1:
                y_np = y_np[..., 0]  # [T,N]
            y = torch.tensor(y_np, dtype=torch.float32)

        # --- 6) Graph structure: edge_csv -> edge_index & graph
        graph = None
        edge_index = None
        edge_weight = None
        if edge_csv:
            edf = pd.read_csv(edge_csv)
            for c in [source_col, target_col]:
                if c not in edf.columns:
                    raise ValueError(f"Edge CSV is missing column: {c}")
            # Keep only edges whose endpoints appear in the nodes set
            edf = edf[edf[source_col].isin(nodes) & edf[target_col].isin(nodes)].copy()
            # De-duplicate
            edf.drop_duplicates(subset=[source_col, target_col], inplace=True)
            src = edf[source_col].map(node2idx).to_numpy()
            dst = edf[target_col].map(node2idx).to_numpy()
            if len(src) > 0:
                ei = np.vstack([src, dst])
                edge_index = torch.as_tensor(ei, dtype=torch.long)
                # Dense adjacency (undirected, unweighted)
                graph = torch.zeros((N, N), dtype=torch.float32)
                graph[edge_index[0], edge_index[1]] = 1.0
                graph[edge_index[1], edge_index[0]] = 1.0
                # Assign uniform weight 1 (could be None; other logic allows)
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
            else:
                # No valid edges; leave empty
                edge_index = None
                graph = torch.zeros((N, N), dtype=torch.float32)

        # --- 7) Build instance (consistent with existing behavior)
        ds = cls(x=x, y=y, graph=graph, dynamic_graph=None,
                 edge_index=edge_index, edge_weight=edge_weight, edge_attr=None)
        # Convenience attributes (does not affect existing logic)
        ds.node_id_mapping = node2idx
        ds.time_index = times
        ds.feature_names = list(feature_cols)
        ds.target_names = list(target_cols) if target_cols else []
        return ds

    # Alias (as requested)
    load_from_csv = from_csv


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
        



