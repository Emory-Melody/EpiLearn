import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
import torch.nn.init as init
import torch

class VARMAXModel:
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, nhid=256, dropout=0.5, use_norm=False):
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.nhid = nhid
        self.dropout = dropout
        self.use_norm = use_norm
        self.model = None
        self.order = (1, 0)
        self.trend = 'c'

    def fit(self, train_input, train_target, val_input=None, val_target=None, epochs=1000, batch_size=10, verbose=False, patience=100):
      
        train_input = train_input.numpy()
        train_target = train_target.numpy()
        val_input = val_input.numpy()
        val_target = val_target.numpy()
        
        subset_samples = 10
        train_input = train_input[:subset_samples]
        train_target = train_target[:subset_samples]
        all_forecasts = []

        for data in train_input:

            if data.shape[0] > max(self.order):
                model = VARMAX(data, order=self.order, trend=self.trend)
                results = model.fit(disp=False)
                forecast = results.forecast(steps=self.num_timesteps_output)
                average_forecast = forecast.mean(axis=1)
                all_forecasts.append(average_forecast)
            else:
                all_forecasts.append(np.array([np.nan] * self.num_timesteps_output))
        
        train_target_subset = train_target[:subset_samples]

        mse_loss = np.mean((train_target_subset - all_forecasts) ** 2)
        print("MSE Loss:", mse_loss)

        return all_forecasts

    def predict(self, feature):
        test_input = feature
        test_input = test_input.numpy()
        all_forecasts = []
        for data in test_input:
            # Check if data is sufficient for the given order
            if data.shape[0] > max(self.order):
                model = VARMAX(data, order=self.order, trend=self.trend)
                results = model.fit(disp=False)
                forecast = results.forecast(steps=self.num_timesteps_output)
                average_forecast = forecast.mean(axis=1)
                all_forecasts.append(average_forecast)
            else:
                all_forecasts.append(np.array([np.nan] * self.num_timesteps_output))

        return torch.tensor(all_forecasts)
    