import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import BaseModel

class CNNModel(BaseModel):
    """
        Convolutional Neural Network for Time Series Forecasting

        Parameters
        ----------
        num_features : int
            Number of features in the input data.
        num_timesteps_input : int
            Number of input timesteps.
        num_timesteps_output : int
            Number of output timesteps to predict.
        conv1_hid : int, optional
            Number of filters in first convolutional layer. Default: 16.
        conv2_hid : int, optional
            Number of filters in second convolutional layer. Default: 32.
        kernel_size : int, optional
            Kernel size for convolutional layers. Default: 3.
        linear_hid : int, optional
            Number of hidden units in the linear layer. Default: 128.
        dropout : float, optional
            Dropout rate. Default: 0.5.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, num_timesteps_output) representing the predicted values for the future timesteps.
            Each element corresponds to a predicted value for a future timestep.
            
        """
    def __init__(self, 
                 num_features, 
                 num_timesteps_input, 
                 num_timesteps_output, 
                 conv1_hid=16,
                 conv2_hid=32,
                 kernel_size=3,
                 linear_hid=128,
                 dropout=0.5, 
                 device='cpu',
                 **kwargs):
        super(CNNModel, self).__init__(device=device)
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.dropout_rate = dropout
        
        # Calculate padding to maintain sequence length after convolution
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels=self.num_features,
                               out_channels=conv1_hid,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding) # first convolutional layer
        self.conv2 = nn.Conv1d(in_channels=conv1_hid,
                               out_channels=conv2_hid,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding) # second convolutional layer
        self.pool = nn.MaxPool1d(kernel_size=2,
                                 stride=2,
                                 padding=0) # Max Pooling
        self.fc1 = nn.Linear(conv2_hid * (num_timesteps_input // 4), linear_hid) # first linear layer
        self.fc2 = nn.Linear(linear_hid, num_timesteps_output) # second linear layer

        self.dropout = nn.Dropout(dropout)


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
            for the future timesteps. Each element corresponds to a predicted value 
            for a future timestep.
        """
        # feature extrction using convolution layers
        x = x.permute([0,2,1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten and perform prediction using fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        