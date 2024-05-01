from .base import BaseModel
import torch.nn as nn
from .base import BaseModel
import torch.nn.init as init
import torch

class LSTMModel(BaseModel):
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, nhid=256, dropout=0.5, use_norm=False):
        super(LSTMModel, self).__init__()
        self.num_features = num_features  # Number of features per timestep
        self.num_timesteps_input = num_timesteps_input  # Number of timesteps in input sequence
        self.num_timesteps_output = num_timesteps_output  # Number of timesteps to predict
        self.nhid = nhid  # Number of hidden units in the LSTM
        self.dropout = dropout
        self.use_norm = use_norm

        # LSTM layer
        self.lstm = nn.LSTM(num_features, nhid, batch_first=True)

        # Optional normalization
        if self.use_norm:
            self.norm = nn.LayerNorm(nhid)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer
        self.out = nn.Linear(nhid, num_timesteps_output)

    def forward(self, x):
        # x should have the shape (batch_size, num_timesteps_input, num_features)
        lstm_out, _ = self.lstm(x)

        if self.use_norm:
            lstm_out = self.norm(lstm_out)

        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)

        # We only use the last timestep's output to predict the future series
        lstm_out = lstm_out[:, -1, :]

        # Output layer to predict the next steps
        output = self.out(lstm_out)
        return output

    def initialize(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        if self.use_norm:
            self.norm.reset_parameters()

        # Initialize the output layer
        init.xavier_uniform_(self.out.weight)
        self.out.bias.data.fill_(0)

# class LSTMNet(BaseModel):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(LSTMNet, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # x: [batch_size, seq_len, num_features]
#         out, (h_n, c_n) = self.lstm(x)  # LSTM输出，同时输出最终隐藏状态和细胞状态
#         out = self.fc(out[:, -1, :])  # 只取序列中的最后一个时间点的输出
#         return out  # 
    
#     def initialize(self):
#         pass
    