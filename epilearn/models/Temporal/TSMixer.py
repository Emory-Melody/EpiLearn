"""
TSMixer Model for Time Series Forecasting

This is a channel-dependent model that mixes both temporal and channel information.
It uses residual blocks with separate temporal and channel mixing MLPs.

Paper: TSMixer: An All-MLP Architecture for Time Series Forecasting
https://arxiv.org/abs/2303.06053

Adapted for EpiLearn by processing all channels together.
For spatiotemporal input (batch, time, nodes, features), it flattens nodes and features
to create (batch, time, nodes*features), treating each node-feature combination as a channel.
"""

import torch
import torch.nn as nn
from .base import BaseModel


class ResBlock(nn.Module):
    """
    Residual block with temporal and channel mixing (aligned with tslib).
    
    The temporal mixing operates on the time dimension (applying linear across time for each channel).
    The channel mixing operates on the channel dimension (applying linear across channels for each time step).
    """
    def __init__(self, seq_len, n_channels, d_model, dropout=0.1):
        super(ResBlock, self).__init__()
        
        # Temporal mixing: [B, L, D] -> permute to [B, D, L] -> Linear -> permute back
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )
        
        # Channel mixing: [B, L, D] -> Linear across D
        self.channel = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D] where L is sequence length, D is number of channels
        Returns:
            x: [B, L, D]
        """
        # Temporal mixing with residual connection (same as tslib)
        # x: [B, L, D] -> [B, D, L] -> temporal -> [B, D, L] -> [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        
        # Channel mixing with residual connection (same as tslib)
        x = x + self.channel(x)
        
        return x


class TSMixerModel(BaseModel):
    """
    TSMixer: Channel-dependent time series forecasting model.
    
    This model treats each channel as a separate entity and applies
    both temporal mixing (across time) and channel mixing (across features)
    using simple MLP blocks.
    
    For spatiotemporal data:
    - Input (batch, time, nodes, features) is flattened to (batch, time, nodes*features)
    - The model treats nodes*features as the total number of channels
    - Output is (batch, horizon) averaged across all channels
    
    Args:
        num_features: Number of input features per node
        num_timesteps_input: Length of input sequence
        num_timesteps_output: Length of output sequence (horizon)
        num_nodes: Number of spatial nodes (optional, for spatiotemporal)
        d_model: Hidden dimension for mixing layers
        e_layers: Number of residual blocks
        dropout: Dropout rate
        device: Device to run the model on
    """
    
    def __init__(
        self,
        num_features: int,
        num_timesteps_input: int,
        num_timesteps_output: int,
        num_nodes: int = None,
        d_model: int = 64,
        e_layers: int = 2,
        dropout: float = 0.1,
        device: str = 'cpu',
        **kwargs
    ):
        super(TSMixerModel, self).__init__()
        
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.d_model = d_model
        self.e_layers = e_layers
        self.device = device
        
        # Calculate total channels (nodes * features for spatiotemporal)
        if num_nodes is not None and num_nodes > 1:
            self.total_channels = num_nodes * num_features
        else:
            self.total_channels = num_features
        
        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(num_timesteps_input, self.total_channels, d_model, dropout)
            for _ in range(e_layers)
        ])
        
        # Output projection: project from input length to output length
        self.projection = nn.Linear(num_timesteps_input, num_timesteps_output)
        
        # Move to device
        self.to(device)
        
    def forward(self, x, A_q=None, A_h=None, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input tensor
                - 3D: (batch, time, features)
                - 4D: (batch, time, nodes, features)
            A_q, A_h: Adjacency matrices (not used, for compatibility)
            
        Returns:
            out: Output tensor 
                - 3D input: (batch, horizon)
                - 4D input: (batch, nodes, horizon)
        """
        # Track if input is spatiotemporal
        is_spatiotemporal = x.dim() == 4

        # Handle 4D input (spatiotemporal): flatten nodes and features
        if is_spatiotemporal:
            batch, time, nodes, features = x.shape
            x = x.reshape(batch, time, nodes * features)  # [B, T, N*F]
        else:
            nodes, features = 1, x.shape[-1]
        
        # x: [B, L, D] where D = total_channels
        
        # Apply residual blocks (same as tslib)
        for block in self.blocks:
            x = block(x)
        
        # Project from input length to output length (same as tslib)
        # x: [B, L, D] -> [B, D, L] -> projection -> [B, D, H] -> [B, H, D]
        enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)  # [B, H, D]
        
        # enc_out: (batch, pred_len, nodes*features)
        if is_spatiotemporal:
            # Reshape to (batch, pred_len, nodes, features)
            enc_out = enc_out.reshape(batch, self.num_timesteps_output, nodes, features)
            # Take last feature (target) for each node: (batch, pred_len, nodes)
            out = enc_out[:, :, :, -1]
            # Transpose to (batch, nodes, pred_len) to match target shape
            out = out.permute(0, 2, 1)
        else:
            # For temporal data, take last column (target)
            out = enc_out[:, :, -1]  # [B, H]
        
        return out
    
    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            return self.forward(feature.to(self.device))
    
    def initialize(self):
        """Re-initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TSMixerLargeModel(TSMixerModel):
    """Larger TSMixer variant with more layers and hidden dimensions."""
    
    def __init__(
        self,
        num_features: int,
        num_timesteps_input: int,
        num_timesteps_output: int,
        num_nodes: int = None,
        d_model: int = 128,
        e_layers: int = 4,
        dropout: float = 0.1,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            num_nodes=num_nodes,
            d_model=d_model,
            e_layers=e_layers,
            dropout=dropout,
            device=device,
            **kwargs
        )


class TSMixerSmallModel(TSMixerModel):
    """Smaller TSMixer variant for faster training."""
    
    def __init__(
        self,
        num_features: int,
        num_timesteps_input: int,
        num_timesteps_output: int,
        num_nodes: int = None,
        d_model: int = 32,
        e_layers: int = 1,
        dropout: float = 0.1,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            num_nodes=num_nodes,
            d_model=d_model,
            e_layers=e_layers,
            dropout=dropout,
            device=device,
            **kwargs
        )
