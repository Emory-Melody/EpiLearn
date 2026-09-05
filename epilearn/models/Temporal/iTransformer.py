"""
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

iTransformer is a channel-dependent model that inverts the typical transformer approach:
- Instead of treating time steps as tokens, it treats channels/features as tokens
- Attention is applied across channels, allowing the model to capture multivariate correlations
- This makes it particularly effective for multivariate time series forecasting

For spatiotemporal data (batch, time, nodes, features), the model flattens nodes and features
into a single channel dimension: (batch, time, nodes * features)

Reference:
    Liu et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (2024)
    https://arxiv.org/abs/2310.06625
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseModel


class DataEmbedding_inverted(nn.Module):
    """
    Inverted embedding that treats each channel as a token.
    
    Input:  (batch, seq_len, n_vars)
    Output: (batch, n_vars, d_model)
    
    Each variable/channel becomes a token, with its time series values as features.
    This is the key innovation of iTransformer (aligned with tslib).
    """
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        Args:
            x: (batch, seq_len, n_vars)
            x_mark: Optional time features (not used in basic version, for tslib compatibility)
        Returns:
            (batch, n_vars, d_model)
        """
        # x: (batch, seq_len, n_vars) -> (batch, n_vars, seq_len)
        x = x.permute(0, 2, 1)
        # (batch, n_vars, seq_len) -> (batch, n_vars, d_model)
        x = self.value_embedding(x)
        return self.dropout(x)


class iTransformerModel(BaseModel):
    """
    iTransformer: Inverted Transformer for Time Series Forecasting
    
    This model inverts the traditional transformer approach by treating channels
    (features/variables) as tokens instead of time steps. This allows the model
    to capture correlations between different channels effectively.
    
    For spatiotemporal data with shape (batch, time, nodes, features), the model
    automatically flattens nodes and features into a single channel dimension.
    
    Parameters
    ----------
    num_features : int
        Number of features per node (for temporal) or total channels (for spatiotemporal).
    num_timesteps_input : int
        Number of input timesteps (lookback window).
    num_timesteps_output : int
        Number of output timesteps to predict (horizon).
    num_nodes : int, optional
        Number of nodes for spatiotemporal data. Default: 1 (temporal only).
    d_model : int, optional
        Dimension of the model embeddings. Default: 64.
    n_heads : int, optional
        Number of attention heads. Default: 4.
    e_layers : int, optional
        Number of encoder layers. Default: 2.
    d_ff : int, optional
        Dimension of feedforward network. Default: 4 * d_model (256 for default d_model=64).
    dropout : float, optional
        Dropout rate. Default: 0.1.
    activation : str, optional
        Activation function ('relu' or 'gelu'). Default: 'relu'.
    device : str, optional
        Device to run model on. Default: 'cpu'.
        
    Input Shapes
    ------------
    Temporal: (batch, time, features)
    Spatiotemporal: (batch, time, nodes, features) -> automatically flattened to (batch, time, nodes*features)
    
    Output Shape
    ------------
    (batch, horizon) - returns prediction for target variable (last column)
    """
    
    def __init__(
        self,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        num_nodes=1,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=None,
        dropout=0.1,
        activation='relu',
        device='cpu',
        **kwargs
    ):
        super(iTransformerModel, self).__init__(device=device)
        
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        
        # Default d_ff to 4 * d_model (same as tslib)
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_ff = d_ff
        
        self._nowcast = bool(kwargs.get('nowcast', False))

        # Total number of channels (for spatiotemporal: nodes * features)
        self.n_vars = num_nodes * num_features
        
        # Inverted embedding: embed each channel using the time dimension
        self.enc_embedding = DataEmbedding_inverted(
            seq_len=num_timesteps_input,
            d_model=d_model,
            dropout=dropout
        )
        
        # Transformer encoder (using PyTorch's built-in, equivalent to tslib's custom encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False  # Post-norm like tslib
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Projection layer: d_model -> pred_len for each channel
        self.projection = nn.Linear(d_model, num_timesteps_output, bias=True)
        
        # Move to device
        self.to(device)
    
    def initialize(self):
        """Re-initialize model weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, **kwargs):
        """
        Forward pass of iTransformer (aligned with tslib implementation).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Can be:
            - Temporal: (batch, time, features)
            - Spatiotemporal: (batch, time, nodes, features)
            
        Returns
        -------
        torch.Tensor
            Predictions:
            - 3D input: (batch, horizon)
            - 4D input: (batch, nodes, horizon)
        """
        # Track if input is spatiotemporal
        is_spatiotemporal = x.dim() == 4
        
        # Handle spatiotemporal input: (batch, time, nodes, features) -> (batch, time, nodes*features)
        if is_spatiotemporal:
            batch_size, seq_len, n_nodes, n_features = x.shape
            x = x.reshape(batch_size, seq_len, n_nodes * n_features)
        else:
            batch_size = x.shape[0]
            n_nodes, n_features = 1, x.shape[-1]
        
        # x: (batch, seq_len, n_vars)
        _, _, N = x.shape

        # Instance normalization
        if self._nowcast:
            # Masked normalization: exclude -1 sentinels from stats
            valid = (x != -1).float()
            x_clean = x * valid
            count = valid.sum(1, keepdim=True).clamp(min=1)
            means = (x_clean.sum(1, keepdim=True) / count).detach()
            x_centered = (x_clean - means) * valid
            stdev = torch.sqrt((x_centered ** 2).sum(1, keepdim=True) / count + 1e-5)
            x = x_centered / stdev
        else:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev
        
        # Inverted embedding: each channel becomes a token
        # (batch, seq_len, n_vars) -> (batch, n_vars, d_model)
        enc_out = self.enc_embedding(x, None)
        
        # Transformer encoder: attention across channels
        # (batch, n_vars, d_model) -> (batch, n_vars, d_model)
        enc_out = self.encoder(enc_out)
        
        # Project to prediction length for each channel and permute
        # (batch, n_vars, d_model) -> (batch, n_vars, pred_len) -> (batch, pred_len, n_vars)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer (same as tslib)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.num_timesteps_output, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.num_timesteps_output, 1))
        
        # dec_out: (batch, pred_len, nodes*features)
        if is_spatiotemporal:
            # Reshape to (batch, pred_len, nodes, features)
            dec_out = dec_out.reshape(batch_size, self.num_timesteps_output, n_nodes, n_features)
            # Take last feature (target) for each node: (batch, pred_len, nodes)
            output = dec_out[:, :, :, -1]
            # Transpose to (batch, nodes, pred_len) to match target shape
            output = output.permute(0, 2, 1)
        else:
            # For temporal data, take last column (target)
            output = dec_out[:, :, -1]  # (batch, pred_len)
        
        return output
    
    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Make predictions.
        
        Parameters
        ----------
        feature : torch.Tensor
            Input features.
            
        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, horizon)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(feature.to(self.device))
