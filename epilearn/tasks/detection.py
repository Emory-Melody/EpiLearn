import torch

from ..utils import utils, metrics  
from .base import BaseTask

class Detection(BaseTask):
    def __init__(self, prototype = None, model = None, dataset = None, lookback = None, horizon = None, device = 'cpu'):
        super().__init__(prototype, model, dataset, lookback, horizon, device)

    def train_model(self,
                    dataset=None,
                    config=None,
                    permute_dataset=True,
                    train_rate=0.6,
                    val_rate=0.2,
                    loss='ce', 
                    epochs=1000, 
                    batch_size=10,
                    lr=1e-3, 
                    region_idx=None,
                    initialize=True, 
                    verbose=False, 
                    patience=100, 
                    ):
        if config is not None:
            permute_dataset = config.permute
            train_rate = config.train_rate
            val_rate = config.val_rate
            loss = config.loss
            epochs=config.epochs
            batch_size=config.batch_size
            lr=config.lr
            initialize=config.initialize
            patience=config.patience

        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please input dataset or use load_dataset() first!")
        else:
            self.dataset = dataset
        
        if not hasattr(self, "model"):
            raise RuntimeError("model not exists, please use load_model() to load model first!")

        train_split, val_split, test_split, ((features, norm), adj_norm) = self.get_splits(self.dataset, train_rate, val_rate, region_idx, permute_dataset)

        try:
            self.model = self.prototype(
                                        num_nodes=adj_norm.shape[0],
                                        num_features=train_split[0].shape[-1],
                                        num_timesteps_input=self.lookback,
                                        num_timesteps_output=self.horizon,
                                        device=self.device
                                        ).to(device=self.device)
            print("spatial-temporal model loaded!")
        except:
            try:
                self.model = self.prototype( 
                                        num_features=train_split[0].shape[2],
                                        num_timesteps_input=self.lookback,
                                        num_timesteps_output=self.horizon,
                                        device=self.device).to(device=self.device)
                                        
                print("temporal model loaded!")
            except:
                self.model = self.prototype(
                                        num_features=train_split[0].shape[-1],
                                        num_classes=self.horizon,
                                        device=self.device).to(device=self.device)
                print("spatial model loaded!")

        # train
        self.model.fit(
                train_input=train_split[0], 
                train_target=train_split[1], 
                train_states = train_split[2],
                train_graph=adj_norm, 
                train_dynamic_graph=train_split[3],
                val_input=val_split[0], 
                val_target=val_split[1], 
                val_states=val_split[2],
                val_graph=adj_norm,
                val_dynamic_graph=val_split[3],
                verbose=True,
                batch_size=batch_size,
                epochs=epochs,
                loss=loss)
        
        # evaluate
        out = self.model.predict(feature=test_split[0], graph=adj_norm, states=test_split[2], dynamic_graph=test_split[3])
        if type(out) is tuple:
            out = out[0]
        preds = out.detach().cpu().argmax(2)
        # metrics
        acc = metrics.get_ACC(preds, test_split[1])
        print(f"Test ACC: {acc.item()}")
        
        return {'acc': acc}


    def evaluate_model(self,
                    model=None,
                    dataset=None,
                    config=None,
                    features=None,
                    graph=None,
                    dynamic_graph=None,
                    norm={"std":1, 'mean':0},
                    states=None,
                    targets=None,
                    ):
        if model is None:
            if not hasattr(self, "model"):
                raise RuntimeError("model not exists, please use load_model() to load model first!")
            model = self.model

        # evaluate
        out = self.model.predict(feature=features, graph=graph, states=states, dynamic_graph=dynamic_graph)
        preds = out.detach().cpu().argmax(1)
        # metrics
        acc = metrics.get_ACC(preds, targets)
        print(f"Test ACC: {acc.item()}")
    


    def get_splits(self, dataset=None, train_rate=0.6, val_rate=0.2, preprocess=False, region_idx=None, permute=False):
        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please use load_dataset() to load dataset first!")
        dataset.x = dataset.x.float()
        # preprocessing
        if preprocess:
            features, mean, std = utils.normalize(dataset.x)
            adj_norm = utils.normalize_adj(dataset.graph)
            features = features.to(self.device)
            adj_norm = adj_norm.to(self.device)
        else:
            features = dataset.x.to(self.device)
            adj_norm = dataset.graph.to(self.device)
            mean = [0]
            std = [1]

        if hasattr(dataset, "dynamic_graph") and dataset.dynamic_graph is not None:
            adj_dynamic_norm = utils.normalize_adj(dataset.dynamic_graph)
            adj_dynamic_norm = adj_dynamic_norm.to(self.device)
        else:
            adj_dynamic_norm = None

        split_line1 = int(features.shape[0] * train_rate)
        split_line2 = int(features.shape[0] * (train_rate + val_rate))

        train_feature = features[:split_line1, :, :]
        val_feature  = features[split_line1:split_line2, :, :]
        test_feature  = features[split_line2:, :, :]

        train_target = dataset.y[:split_line1, :]
        val_target = dataset.y[split_line1:split_line2, :]
        test_target = dataset.y[split_line2:, :]
        try:
            train_states = dataset.states[:split_line1, :, :]
            val_states = dataset.states[split_line1:split_line2, :, :]
            test_states = dataset.states[split_line2:, :, :]
        except:
            train_states = None
            val_states = None
            test_states = None

        if hasattr(dataset, 'dynamic_graph') and dataset.dynamic_graph is not None:
            train_graph = dataset.dynamic_graph[:split_line1, :, :]
            val_graph = dataset.dynamic_graph[split_line1:split_line2, :, :]
            test_graph = dataset.dynamic_graph[split_line2:, :, :]
        else:
            train_graph = None
            val_graph = None
            test_graph = None

        return (train_feature, train_target, train_states, train_graph), (val_feature, val_target, val_states, val_graph), (test_feature, test_target, test_states, test_graph), ((features, (std[0], mean[0])), adj_norm)

        
