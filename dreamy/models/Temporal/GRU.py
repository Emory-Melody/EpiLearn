import torch.nn as nn
from .base_model import BaseModel


class GRUNet(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]
        out, _ = self.gru(x)  # GRU输出
        out = self.fc(out[:, -1, :])  # 只取序列中的最后一个时间点的输出
        return out  # 调整输出形状以匹配 y 的形状