import torch.nn as nn
from .base_model import BaseModel


class LSTMNet(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]
        out, (h_n, c_n) = self.lstm(x)  # LSTM输出，同时输出最终隐藏状态和细胞状态
        out = self.fc(out[:, -1, :])  # 只取序列中的最后一个时间点的输出
        return out  # 调整输出形状以匹配 y 的形状