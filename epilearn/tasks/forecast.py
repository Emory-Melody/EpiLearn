import torch

from ..utils import utils, metrics  
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
        # initialize model
            self.model = self.prototype(
                num_nodes=adj_norm.shape[0],
                num_features=train_split[0].shape[3],
                num_timesteps_input=self.lookback,
                num_timesteps_output=self.horizon,
                
                ).to(device=self.device)
            print("spatial-temporal model loaded!")
        except:
            self.model = self.prototype( num_features=train_split[0].shape[2],
                                    num_timesteps_input=self.lookback,
                                    num_timesteps_output=self.horizon)

            print("temporal model loaded!")

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
        preds = out.detach().cpu()*norm[0]+norm[1]
        targets = test_split[1].detach().cpu()*norm[0]+norm[1]
        # metrics
        mae = metrics.get_MAE(preds, targets)
        rmse = metrics.get_RMSE(preds, targets)
        print(f"Test MAE: {mae.item()}")
        print(f"Test RMSE: {rmse.item()}")
        return mae, rmse


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
        preds = out.detach().cpu()*norm['std']+norm['mean']
        targets = targets[1].detach().cpu()*norm[0]+norm[1]
        # metrics
        mae = metrics.get_MAE(preds, targets)
        rmse = metrics.get_RMSE(preds, targets)
        print(f"Test MAE: {mae.item()}")
        print(f"Test RMSE: {rmse.item()}")
    


    def get_splits(self, dataset=None, train_rate=0.6, val_rate=0.2, region_idx=None, permute=False):
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

        if hasattr(dataset, "dynamic_graph"):
            adj_dynamic_norm = utils.normalize_adj(dataset.dynamic_graph)
            adj_dynamic_norm = adj_dynamic_norm.to(self.device)
        else:
            adj_dynamic_norm = None

        split_line1 = int(features.shape[0] * train_rate)
        split_line2 = int(features.shape[0] * (train_rate + val_rate))

        train_original_data = features[:split_line1, :, :]
        val_original_data = features[split_line1:split_line2, :, :]
        test_original_data = features[split_line2:, :, :]

        train_original_states = dataset.states[:split_line1, :, :]
        val_original_states = dataset.states[split_line1:split_line2, :, :]
        test_original_states = dataset.states[split_line2:, :, :]


        train_input, train_target, train_states, train_adj = dataset.generate_dataset(
                                                                                        X=train_original_data, 
                                                                                        Y=train_original_data[:, :, 0], 
                                                                                        states=train_original_states,
                                                                                        dynamic_adj = adj_dynamic_norm,
                                                                                        lookback_window_size=self.lookback,
                                                                                        horizon_size=self.horizon, 
                                                                                        permute=permute)
        val_input, val_target, val_states, val_adj = dataset.generate_dataset(
                                                                                X=val_original_data, 
                                                                                Y=val_original_data[:, :, 0], 
                                                                                states=val_original_states,
                                                                                dynamic_adj = adj_dynamic_norm,
                                                                                lookback_window_size=self.lookback, 
                                                                                horizon_size=self.horizon, 
                                                                                permute=permute)
        test_input, test_target, test_states, test_adj = dataset.generate_dataset(
                                                                                    X=test_original_data, 
                                                                                    Y=test_original_data[:, :, 0], 
                                                                                    states=test_original_states,
                                                                                    dynamic_adj = adj_dynamic_norm,
                                                                                    lookback_window_size=self.lookback, 
                                                                                    horizon_size=self.horizon, 
                                                                                    permute=permute)
        
        if region_idx is not None:
            train_input = train_input[:,:,region_idx,:]
            val_input = val_input[:,:,region_idx,:]
            test_input = test_input[:,:,region_idx,:]

            train_target = train_target[:,:,region_idx]
            val_target = val_target[:,:,region_idx]
            test_target = test_target[:,:,region_idx]

            train_states = train_states[:,:,region_idx]
            val_states = val_states[:,:,region_idx]
            test_states = test_states[:,:,region_idx]

            features = features[:,region_idx,:]



        return (train_input, train_target, train_states, train_adj), (val_input, val_target, val_states, val_adj), (test_input, test_target, test_states, test_adj), ((features, (std[0], mean[0])), adj_norm)

        
