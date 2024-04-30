import torch

from ..utils import utils
from .base import BaseTask

class Forecast(BaseTask):
    def __init__(self, model, dataset = None, lookback = None, horizon = None, device = 'cpu'):
        super().__init__(model, dataset, lookback, horizon, device)

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

        train_split, val_split, test_split = self.get_splits(self.dataset, permute_dataset, train_rate, val_rate)




    def evaluate_model(self,
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
        pass
    


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

        # prepare datasets
        train_rate = 0.6 
        val_rate = 0.2

        split_line1 = int(features.shape[0] * train_rate)
        split_line2 = int(features.shape[0] * (train_rate + val_rate))

        train_original_data = features[:split_line1, :, :]
        val_original_data = features[split_line1:split_line2, :, :]
        test_original_data = features[split_line2:, :, :]

        train_input, train_target = dataset.generate_dataset(X=train_original_data, 
                                                             Y=train_original_data[:, :, 0], 
                                                             lookback_window_size=self.lookback,
                                                             horizon_size=self.horizon, 
                                                             permute=permute)
        val_input, val_target = dataset.generate_dataset(X=val_original_data, 
                                                         Y=val_original_data[:, :, 0], 
                                                         lookback_window_size=self.lookback, 
                                                         horizon_size=self.horizon, 
                                                         permute=permute)
        test_input, test_target = dataset.generate_dataset(X=test_original_data, 
                                                           Y=test_original_data[:, :, 0], 
                                                           lookback_window_size=self.lookback, 
                                                           horizon_size=self.horizon, 
                                                           permute=permute)

        return (train_input, train_target), (val_input, val_target), (test_input, test_target)

        
