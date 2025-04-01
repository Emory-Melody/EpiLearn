import torch.nn as nn
from .base import BaseModel
import torch.nn.init as init
import torch


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DlinearModel(BaseModel):
    """
    Dynamic Linear Model

    Parameters
    ----------
    num_features : int
        Number of features in each timestep of the input data.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.

    Attributes
    ----------
    decomposition : function
        Method to decompose the time series data into seasonal and trend components.
    Linear_Transform : torch.nn.Linear
        Linear transformation layer to project the decomposed input data to the output space.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output) representing the predicted values for the future timesteps.
        Each element corresponds to a predicted value for a specific future timestep. The output is averaged across the feature dimension to reduce it to a single predictive value per timestep.

    """
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, device='cpu'):
        super(DlinearModel, self).__init__(device=device)
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.decomposition = series_decomp(kernel_size=3) 
        # Single Linear layer to handle all features
        self.Linear_Transform = nn.Linear(num_timesteps_input * num_features, num_timesteps_output * num_features)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor representing time series data for each batch. Expected shape is 
            (batch_size, num_timesteps_input, num_features), where `batch_size` is the number of samples in the batch,
            `num_timesteps_input` is the number of input time steps, and `num_features` represents features at each time step.

        Returns
        -------
        torch.Tensor
            The output of the model, a tensor of shape (batch_size, num_timesteps_output) that represents the predicted values 
            for the future time steps, reduced to a single predictive value per time step by averaging across the feature dimension.
        """
        # Decompose and process as before
        seasonal_init, trend_init = self.decomposition(x)
        batch_size = x.shape[0]
        
        # Flatten the input for linear transformation
        seasonal_flat = seasonal_init.view(batch_size, -1)
        trend_flat = trend_init.view(batch_size, -1)
        
        # Apply linear transformations
        seasonal_output_flat = self.Linear_Transform(seasonal_flat)
        trend_output_flat = self.Linear_Transform(trend_flat)
        
        # Reshape back and combine outputs
        seasonal_output = seasonal_output_flat.view(batch_size, self.num_timesteps_output, self.num_features)
        trend_output = trend_output_flat.view(batch_size, self.num_timesteps_output, self.num_features)
        output = seasonal_output + trend_output
        
        # Reduce feature dimension by averaging across features
        output = output.mean(dim=2)  # Average across the feature dimension
        # print(output.shape)  # Should now be [50, 3]
        return output

    def reset_parameters(self):
        # Here you reset parameters for each layer explicitly
        self.Linear_Transform.reset_parameters()
        # Add any other layers that need resetting
        # Note: Add here if decomposition or its components have resettable parameters

    def initialize(self):
        # This initializes parameters by resetting them for all children layers automatically
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
  
