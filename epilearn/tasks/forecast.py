import torch

from ..utils import utils
from .base import BaseTask

class Forecast(BaseTask):
    def __init__(self, prototype = None, model = None, dataset = None, lookback = None, horizon = None, device = 'cpu'):
        super().__init__(prototype, model, dataset, lookback, horizon, device)


    def train_model(self,
                    dataset=None,
                    config=None,
                    permute_dataset=False,
                    train_rate=0.6,
                    val_rate=0.2,
                    loss='mse', 
                    epochs=1000, 
                    batch_size=10,
                    lr=1e-3, 
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

        train_split, val_split, test_split, ((features, norm), adj_norm) = self.get_splits(self.dataset, train_rate, val_rate, permute_dataset)

        # initialize model
        self.model = self.prototype(
            num_nodes=adj_norm.shape[0],
            num_features=train_split[0].shape[3],
            num_timesteps_input=self.lookback,
            num_timesteps_output=self.horizon,
            
            ).to(device=self.device)

        # train
        self.model.fit(
                train_input=train_split[0], 
                train_target=train_split[1], 
                train_states = train_split[2],
                train_graph=adj_norm, 
                val_input=val_split[0], 
                val_target=val_split[1], 
                val_states=val_split[2],
                val_graph=adj_norm,
                verbose=True,
                batch_size=batch_size,
                epochs=epochs)
        
        # evaluate
        out = self.model.predict(feature=test_split[0], graph=adj_norm)
        preds = out.detach().cpu()*norm[0]+norm[1]
        targets = test_split[1].detach().cpu()*norm[0]+norm[1]
        # MAE
        mae = utils.get_MAE(preds, targets)
        print(f"Test MAE: {mae.item()}")


    def evaluate_model(self,
                    model=None,
                    dataset=None,
                    config=None,
                    features=None,
                    graph=None,
                    norm={"std":1, 'mean':0},
                    states=None,
                    targets=None,
                    batch_size=10,
                    ):
        if model is None:
            if not hasattr(self, "model"):
                raise RuntimeError("model not exists, please use load_model() to load model first!")
            model = self.model

        # evaluate
        out = self.model.predict(feature=features, graph=graph)
        preds = out.detach().cpu()*norm['std']+norm['mean']
        targets = targets[1].detach().cpu()*norm[0]+norm[1]
        # MAE
        mae = utils.get_MAE(preds, targets)
        print(f"Test MAE: {mae.item()}")
    


    def get_splits(self, dataset=None, train_rate=0.6, val_rate=0.2, permute=False):
        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please use load_dataset() to load dataset first!")
            
        # preprocessing
        features, mean, std = utils.normalize(dataset.x)
        adj_norm = utils.normalize_adj(dataset.graph)

        features = features.to(self.device)
        adj_norm = adj_norm.to(self.device)

        split_line1 = int(features.shape[0] * train_rate)
        split_line2 = int(features.shape[0] * (train_rate + val_rate))

        train_original_data = features[:split_line1, :, :]
        val_original_data = features[split_line1:split_line2, :, :]
        test_original_data = features[split_line2:, :, :]

        train_input, train_target, train_states, train_adj = dataset.generate_dataset(X=train_original_data, 
                                                             Y=train_original_data[:, :, 0], 
                                                             lookback_window_size=self.lookback,
                                                             horizon_size=self.horizon, 
                                                             permute=permute)
        val_input, val_target, val_states, val_adj = dataset.generate_dataset(X=val_original_data, 
                                                         Y=val_original_data[:, :, 0], 
                                                         lookback_window_size=self.lookback, 
                                                         horizon_size=self.horizon, 
                                                         permute=permute)
        test_input, test_target, test_states, test_adj = dataset.generate_dataset(X=test_original_data, 
                                                           Y=test_original_data[:, :, 0], 
                                                           lookback_window_size=self.lookback, 
                                                           horizon_size=self.horizon, 
                                                           permute=permute)

        return (train_input, train_target, train_states, train_adj), (val_input, val_target, val_states, val_adj), (test_input, test_target, test_states, test_adj), ((features, (std[0], mean[0])), adj_norm)

        
