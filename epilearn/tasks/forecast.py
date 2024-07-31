import torch

from ..visualize import plot_series
from ..utils import utils, metrics  
from .base import BaseTask

class Forecast(BaseTask):
    """
    The Forecast class extends the BaseTask class, focusing on the training and evaluation of forecast models for 
    time-series prediction tasks. It includes functionalities specific to handling time-series data, especially 
    in settings that involve spatial-temporal dynamics. The class supports model initialization, training, evaluation, 
    and preprocessing, facilitating the application of various neural network architectures and configurations.
    """
    def __init__(self, prototype = None, model = None, dataset = None, lookback = None, horizon = None, device = 'cpu'):
        super().__init__(prototype, model, dataset, lookback, horizon, device)
        self.feat_mean = 0
        self.feat_std = 1
        self.device = device


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
                    device = None,
                    ):
        """
        Trains the forecast model using the provided dataset and configuration settings. It handles data splitting, model 
        initialization, and the training process, and also evaluates the model on the test set, reporting metrics such as 
        MAE and RMSE.
        """
        if device is not None:
            self.device = device
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
        
        self.region_index = region_idx
        self.train_split, self.val_split, self.test_split, ((features, norm), adj) = self.get_splits(self.dataset, train_rate, val_rate, region_idx, permute_dataset)

        try:
        # initialize model
            self.model = self.prototype(
                num_nodes=adj.shape[0],
                num_features=self.train_split[0].shape[3],
                num_timesteps_input=self.lookback,
                num_timesteps_output=self.horizon,
                device=self.device
                ).to(self.device)
            print("spatial-temporal model loaded!")
        except:
            self.model = self.prototype( num_features=self.train_split[0].shape[2],
                                    num_timesteps_input=self.lookback,
                                    num_timesteps_output=self.horizon,
                                    device=self.device).to(self.device)
            print("temporal model loaded!")

        # train
        self.model.fit(
                train_input=self.train_split[0], 
                train_target=self.train_split[1], 
                train_states = self.train_split[2],
                train_graph=adj, 
                train_dynamic_graph=self.train_split[3],
                val_input=self.val_split[0], 
                val_target=self.val_split[1], 
                val_states=self.val_split[2],
                val_graph=adj,
                val_dynamic_graph=self.val_split[3],
                verbose=True,
                batch_size=batch_size,
                epochs=epochs,
                loss=loss)
        
        # evaluate
        self.test_graph = adj
        self.test_feature = self.test_split[0]
        self.test_target = self.test_split[1]
        self.test_states = self.test_split[2]
        self.test_dynamic_graph = self.test_split[3]

        out = self.model.predict(feature=self.test_feature, 
                                 graph=self.test_graph, 
                                 states=self.test_states, 
                                 dynamic_graph=self.test_dynamic_graph
                                 )
        if type(out) is tuple:
            out = out[0]
        preds = out.detach().cpu()*norm[0]+norm[1]
        targets = self.test_target.detach().cpu()*norm[0]+norm[1]
        # metrics
        mse = metrics.get_MSE(preds, targets)
        mae = metrics.get_MAE(preds, targets)
        rmse = metrics.get_RMSE(preds, targets)
        print(f"Test MSE: {mse.item()}")
        print(f"Test MAE: {mae.item()}")
        print(f"Test RMSE: {rmse.item()}")
        
        return {"mse": mse.item(), "mae":mae.item(), "rmse":rmse.item()}


    def evaluate_model(self,
                    model=None,
                    config=None,
                    features=None,
                    graph=None,
                    dynamic_graph=None,
                    norm=None,
                    states=None,
                    targets=None,
                    ):
        """
        Evaluates the trained model on a new dataset or using preloaded features and graphs. It outputs prediction accuracy 
        metrics such as MAE and RMSE for the forecasted values.
        """
        if model is None:
            if not hasattr(self, "model"):
                raise RuntimeError("model not exists, please use load_model() to load model first!")
            model = self.model
        
        features = self.test_feature if features is None else features
        graph = self.test_graph if graph is None else graph
        states = self.test_states if states is None else states
        dynamic_graph = self.test_dynamic_graph if dynamic_graph is None else dynamic_graph
        targets = self.test_target if targets is None else targets
        mean = self.feat_mean[0] if norm is None else norm['mean']
        std = self.feat_std[0] if norm is None else norm['std']

        # evaluate
        out = self.model.predict(feature=features, 
                                 graph=graph, 
                                 states=states, 
                                 dynamic_graph=dynamic_graph
                                 )
        if type(out) is tuple:
            out = out[0]
        preds = out.detach().cpu()*std+mean
        targets = targets.detach().cpu()*std+mean
        # metrics
        mse = metrics.get_MSE(preds, targets)
        mae = metrics.get_MAE(preds, targets)
        rmse = metrics.get_RMSE(preds, targets)
        print(f"Test MSE: {mse.item()}")
        print(f"Test MAE: {mae.item()}")
        print(f"Test RMSE: {rmse.item()}")
        
        return {"mse": mse.item(), "mae":mae.item(), "rmse":rmse.item()}
    


    def get_splits(self, dataset=None, train_rate=0.6, val_rate=0.2, region_idx=None, permute=False):
        """
        Splits the provided dataset into training, validation, and testing sets based on specified rates. It also handles 
        preprocessing to normalize the data and prepare it for the model.
        """
        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please use load_dataset() to load dataset first!")
            
        # preprocessing
        features, adj, adj_dynamic, states = dataset.get_transformed()

        feat_mean, feat_std = dataset.transforms.feat_mean, dataset.transforms.feat_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        split_line1 = int(features.shape[0] * train_rate)
        split_line2 = int(features.shape[0] * (train_rate + val_rate))

        self.train_original_data = features[:split_line1, ...]
        self.val_original_data = features[split_line1:split_line2, ...]
        self.test_original_data = features[split_line2:, ...]
        
        train_original_states = None
        val_original_states = None
        test_original_states = None

        if states is not None:
            train_original_states = states[:split_line1, ...]
            val_original_states = states[split_line1:split_line2, ...]
            test_original_states = states[split_line2:, ...]


        train_input, train_target, train_states, train_adj = dataset.generate_dataset(
                                                                                        X=self.train_original_data, 
                                                                                        Y=self.train_original_data[..., 0], 
                                                                                        states=train_original_states,
                                                                                        dynamic_adj = adj_dynamic,
                                                                                        lookback_window_size=self.lookback,
                                                                                        horizon_size=self.horizon, 
                                                                                        permute=permute)
        val_input, val_target, val_states, val_adj = dataset.generate_dataset(
                                                                                X=self.val_original_data, 
                                                                                Y=self.val_original_data[..., 0], 
                                                                                states=val_original_states,
                                                                                dynamic_adj = adj_dynamic,
                                                                                lookback_window_size=self.lookback, 
                                                                                horizon_size=self.horizon, 
                                                                                permute=permute)
        test_input, test_target, test_states, test_adj = dataset.generate_dataset(
                                                                                    X=self.test_original_data, 
                                                                                    Y=self.test_original_data[..., 0], 
                                                                                    states=test_original_states,
                                                                                    dynamic_adj = adj_dynamic,
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

        return (train_input, train_target, train_states, train_adj), (val_input, val_target, val_states, val_adj), (test_input, test_target, test_states, test_adj), ((features, (feat_std[0], feat_mean[0])), adj)
    
    def plot_forecasts(self, index_range=None):
        data = self.test_original_data
        if data.shape[-1] >1:
            raise ValueError("Multi channel is not supported. Please use single channel data.")
        if self.region_index is not None:
            data = data[:, self.region_index,:]


        # save groundtruth and predictions
        predictions = torch.FloatTensor()
        groundtruth = torch.FloatTensor()
        
        with torch.no_grad():
            
            for i in range(0, len(data)-self.horizon-self.lookback, self.horizon):
                history = torch.FloatTensor(data[i:i+self.lookback])
                label = data[i+self.lookback:i+self.lookback+self.horizon]

                output = self.model(history.unsqueeze(0).to(self.device))
                output = output.detach().cpu()
                preds = (output*self.feat_std + self.feat_mean).squeeze(0)
                label = (label*self.feat_std + self.feat_mean).squeeze(-1)

                groundtruth = torch.cat([groundtruth, label])
                predictions = torch.cat([predictions, preds])
        result = torch.stack([predictions, groundtruth], dim=1)
        result = result[index_range[0]:index_range[1]]
        plot_series(result, columns = ['prediction', 'groundtruth'])

        return predictions, groundtruth

        
