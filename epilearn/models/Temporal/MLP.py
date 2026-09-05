import torch
import torch.nn as nn
import torch.nn.init as init
from .base import BaseModel


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron (MLP) Model for Time Series Forecasting
    
    A simple feedforward neural network that flattens the input time series
    and processes it through multiple fully connected layers.

    Parameters
    ----------
    num_features : int
        Number of features in each timestep of the input data.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    hidden_dims : list of int, optional
        List of hidden layer dimensions. Default: [256, 128].
    dropout : float, optional
        Dropout rate for regularization. Default: 0.2.
    activation : str, optional
        Activation function ('relu', 'gelu', 'tanh'). Default: 'relu'.
    use_batch_norm : bool, optional
        Whether to use batch normalization after each hidden layer. Default: False.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output) representing the predicted values 
        for the future timesteps.

    Examples
    --------
    >>> model = MLPModel(num_features=3, num_timesteps_input=24, num_timesteps_output=12)
    >>> x = torch.randn(32, 24, 3)  # batch_size=32
    >>> output = model(x)
    >>> output.shape
    torch.Size([32, 12])
    """
    
    def __init__(self, 
                 num_features, 
                 num_timesteps_input, 
                 num_timesteps_output,
                 hidden_dims=[256, 128],
                 dropout=0.2,
                 activation='relu',
                 use_batch_norm=False,
                 device='cpu',
                 **kwargs):
        super(MLPModel, self).__init__(device=device)
        
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Calculate input dimension (flatten time series)
        input_dim = num_features * num_timesteps_input
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_timesteps_output))
        
        # Create sequential model
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, **kwargs):
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the model. Expected shape is 
            (batch_size, num_timesteps_input, num_features), where
            `batch_size` is the number of samples in the batch,
            `num_timesteps_input` is the number of input timesteps,
            and `num_features` is the number of features for each timestep.

        Returns
        -------
        torch.Tensor
            The output of the model, a tensor of shape 
            (batch_size, num_timesteps_output), representing the predicted values 
            for the future timesteps.
        """
        # x shape: (batch_size, num_timesteps_input, num_features)
        
        batch_size = x.shape[0]
        
        # Flatten the input: (batch_size, num_timesteps_input * num_features)
        x_flat = x.reshape(batch_size, -1)
        
        # Pass through MLP
        output = self.mlp(x_flat)  # (batch_size, num_timesteps_output)
        
        return output
    
    def initialize(self):
        """
        Initialize model parameters using Xavier/Glorot initialization for weights
        and zeros for biases.
        """
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
