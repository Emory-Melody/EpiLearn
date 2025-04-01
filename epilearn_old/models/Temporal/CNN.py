import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import BaseModel

class CNNModel(BaseModel):
    """
        Single-layer Gated Recurrent Unit (GRU) Network

        Parameters
        ----------
        num_features : int
            Number of features in the input data.
        num_timesteps_input : int
            Number of input timesteps.
        num_timesteps_output : int
            Number of output timesteps to predict.
        nhid : int, optional
            Number of hidden units in the GRU layer. Default: 256.
        dropout : float, optional
            Dropout rate for the GRU layer. Default: 0.5.
        use_norm : bool, optional
            Whether to use Layer Normalization after the GRU layer. Default: False.

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
                 dropout=0.5, 
                 conv1_config={'hid': 16, 'kernel': 3, 'stride': 1, 'padding': 1}, 
                 conv2_config={'hid': 32, 'kernel': 3, 'stride': 1, 'padding': 1},
                 maxpool_config={'kernel': 2, 'stride': 2, 'padding': 0}, 
                 linear_hid=128, 
                 device='cpu'):
        super(CNNModel, self).__init__(device=device)
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.dropout = dropout

        self.conv1 = nn.Conv1d(in_channels=self.num_features,
                               out_channels=conv1_config['hid'],
                               kernel_size=conv1_config['kernel'],
                               stride=conv1_config['stride'],
                               padding=conv1_config['padding']) # first convolutional layer
        self.conv2 = nn.Conv1d(in_channels=conv1_config['hid'],
                               out_channels=conv2_config['hid'],
                               kernel_size=conv2_config['kernel'],
                               stride=conv2_config['stride'],
                               padding=conv2_config['padding']) # second convolutional layer
        self.pool = nn.MaxPool1d(kernel_size=maxpool_config['kernel'],
                                 stride=maxpool_config['stride'],
                                 padding=maxpool_config['padding']) # Max Pooling
        self.fc1 = nn.Linear(conv2_config['hid'] * (num_timesteps_input // 4), linear_hid) # first linear layer
        self.fc2 = nn.Linear(linear_hid, num_timesteps_output) # second linear layer

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
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
        
        