import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
import torch.nn.init as init
import torch

class VARMAXModel:
    """
    Vector Autoregression Moving-Average with eXogenous variables (VARMAX) Model

    Parameters
    ----------
    num_features : int
        Number of features in each timestep of the input data.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    
    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output) representing the predicted values for the future timesteps.
        Each element corresponds to a predicted value for a future timestep.

    """
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, order=(1, 0), trend='c'):
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.model = None
        self.order = order
        self.trend = trend

    def fit(self, train_input, train_target, train_states=None, val_input=None, val_target=None, epochs=1000, batch_size=10, verbose=False, patience=100, **kwargs):
        """
        Parameters
        ----------
        train_input : numpy.ndarray
            The input training data array, expected to be in the shape 
            (batch_size, num_timesteps_input, num_features), where:
            - `batch_size` is the number of training samples,
            - `num_timesteps_input` is the number of timesteps used as input,
            - `num_features` is the number of features per timestep.
        train_target : numpy.ndarray
            The target training data array corresponding to the inputs, shaped 
            (batch_size, num_timesteps_output).
        val_input : numpy.ndarray, optional
            The input validation data array, following the same format as `train_input`.
        val_target : numpy.ndarray, optional
            The target validation data array, following the same format as `train_target`.
        epochs : int, optional
            The number of epochs to train the model for. Default is 1000.
        batch_size : int, optional
            The size of batches to use when training the model. Default is 10.
        verbose : bool, optional
            If True, the model will print out progress during training. Default is False.
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped. Default is 100.

        Returns
        -------
        list
            A list of forecasted values for each batch in the training data, each forecast corresponding 
            to the future timesteps as defined by `num_timesteps_output`.

        Notes
        -----
        This method internally converts the input PyTorch Tensors to numpy arrays if not already provided in that format, 
        fits a VARMAX model for each subset of data, and performs forecasting. It is designed to handle smaller subsets 
        of the data (defined by `subset_samples`) to demonstrate the model's fitting capability.
        """
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

    def predict(self, feature, **kwargs):
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

    



import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import torch

class ARIMAModel:
    """
    Autoregressive Integrated Moving Average (ARIMA) Model

    Parameters
    ----------
    num_features : int
        Number of features in each timestep of the input data.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    order : tuple, optional
        The (p, d, q) order of the ARIMA model. If only two values are provided,
        they are interpreted as (p, d) with q defaulting to 0.
    trend : str, optional
        The trend parameter for the ARIMA model. Default is 'c'.
    
    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output) representing the predicted values
        for the future timesteps. Each element corresponds to a predicted value for a future timestep.
    """
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, order=(1, 0), trend='c'):
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.model = None
        self.order = order
        self.trend = trend

    def _get_order(self):
        # If order is provided as two values, assume q=0.
        if len(self.order) == 2:
            return (self.order[0], self.order[1], 0)
        return self.order

    def fit(self, train_input, train_target, train_states=None, val_input=None, val_target=None,
            epochs=1000, batch_size=10, verbose=False, patience=100, **kwargs):
        """
        Parameters
        ----------
        train_input : torch.Tensor
            The input training data tensor, expected to be in the shape 
            (batch_size, num_timesteps_input, num_features).
        train_target : torch.Tensor
            The target training data tensor, shaped (batch_size, num_timesteps_output).
        val_input : torch.Tensor, optional
            The input validation data tensor.
        val_target : torch.Tensor, optional
            The target validation data tensor.
        epochs : int, optional
            Number of epochs to train the model. (Not used in ARIMA fitting)
        batch_size : int, optional
            Batch size (not used in ARIMA fitting).
        verbose : bool, optional
            If True, prints progress and error messages.
        patience : int, optional
            Early stopping patience (not used in ARIMA fitting).

        Returns
        -------
        list
            A list of forecasted values (each a numpy array of length num_timesteps_output)
            for each batch in the training data.
        """
        # Convert PyTorch tensors to numpy arrays
        train_input = train_input.numpy()
        train_target = train_target.numpy()
        if val_input is not None:
            val_input = val_input.numpy()
        if val_target is not None:
            val_target = val_target.numpy()
        
        subset_samples = 10  # Process only a subset of samples for demonstration
        train_input = train_input[:subset_samples]
        train_target = train_target[:subset_samples]
        all_forecasts = []

        order_tuple = self._get_order()

        # Loop over each sample in the training data.
        for data in train_input:
            # ARIMA is univariate. If multiple features are provided, use the first feature.
            series = data[:, 0] if self.num_features > 1 else data.flatten()

            # Ensure the series is long enough for the given order
            if series.shape[0] > max(order_tuple):
                try:
                    model = ARIMA(series, order=order_tuple, trend=self.trend)
                    results = model.fit()
                    forecast = results.forecast(steps=self.num_timesteps_output)
                    # Ensure forecast is a numpy array
                    forecast = np.asarray(forecast)
                    all_forecasts.append(forecast)
                except Exception as e:
                    if verbose:
                        print("ARIMA fitting error:", e)
                    all_forecasts.append(np.array([np.nan] * self.num_timesteps_output))
            else:
                all_forecasts.append(np.array([np.nan] * self.num_timesteps_output))
        
        # Compute Mean Squared Error on the training subset.
        train_target_subset = train_target[:subset_samples]
        mse_loss = np.mean((train_target_subset - np.array(all_forecasts)) ** 2)
        print("MSE Loss:", mse_loss)

        return all_forecasts

    def predict(self, feature, **kwargs):
        """
        Parameters
        ----------
        feature : torch.Tensor
            The input data tensor for which to forecast future timesteps.
            Expected shape: (batch_size, num_timesteps_input, num_features).

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, num_timesteps_output) with the forecasted values.
        """
        test_input = feature.numpy()
        all_forecasts = []
        order_tuple = self._get_order()

        for data in test_input:
            series = data[:, 0] if self.num_features > 1 else data.flatten()

            if series.shape[0] > max(order_tuple):
                try:
                    model = ARIMA(series, order=order_tuple, trend=self.trend)
                    results = model.fit()
                    forecast = results.forecast(steps=self.num_timesteps_output)
                    forecast = np.asarray(forecast)
                    all_forecasts.append(forecast)
                except Exception as e:
                    all_forecasts.append(np.array([np.nan] * self.num_timesteps_output))
            else:
                all_forecasts.append(np.array([np.nan] * self.num_timesteps_output))
        
        return torch.tensor(all_forecasts)
    
