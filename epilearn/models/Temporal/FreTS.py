"""
FreTS Model for Time Series Forecasting

This is a channel-dependent model that operates in the frequency domain.
It uses FFT to transform the signal, applies MLPs in the frequency domain,
and then transforms back using inverse FFT.

Paper: FreTS: Frequency-domain MLPs are More Effective Learners in Time Series Forecasting
https://arxiv.org/abs/2311.06184

Adapted for EpiLearn by processing all channels together.
For spatiotemporal input (batch, time, nodes, features), it flattens nodes and features
to create (batch, time, nodes*features), treating each node-feature combination as a channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class FreTSModel(BaseModel):
    """
    FreTS: Frequency-domain MLP model for time series forecasting.
    
    This model operates primarily in the frequency domain:
    1. Embeds each time point with learnable embeddings
    2. Applies frequency-domain MLPs on both temporal and channel dimensions
    3. Uses FFT for temporal/channel transformations
    
    For spatiotemporal data:
    - Input (batch, time, nodes, features) is flattened to (batch, time, nodes*features)
    - The model treats nodes*features as the total number of channels
    - Output is (batch, horizon) averaged across all channels
    
    Args:
        num_features: Number of input features per node
        num_timesteps_input: Length of input sequence
        num_timesteps_output: Length of output sequence (horizon)
        num_nodes: Number of spatial nodes (optional, for spatiotemporal)
        embed_size: Embedding dimension
        hidden_size: Hidden dimension for output projection
        channel_independence: If True, skip channel mixing (faster but less expressive)
        sparsity_threshold: Threshold for soft shrinkage
        device: Device to run the model on
    """
    
    def __init__(
        self,
        num_features: int,
        num_timesteps_input: int,
        num_timesteps_output: int,
        num_nodes: int = None,
        embed_size: int = 128,
        hidden_size: int = 256,
        channel_independence: bool = False,
        sparsity_threshold: float = 0.01,
        device: str = 'cpu',
        **kwargs
    ):
        super(FreTSModel, self).__init__()
        
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channel_independence = channel_independence
        self.sparsity_threshold = sparsity_threshold
        self.device = device
        
        # Calculate total channels
        if num_nodes is not None and num_nodes > 1:
            self.total_channels = num_nodes * num_features
        else:
            self.total_channels = num_features
        
        self.scale = 0.02
        
        # Learnable embeddings for value encoding
        self.embeddings = nn.Parameter(torch.randn(1, embed_size))
        
        # Frequency-domain MLP parameters for channel mixing
        self.r1 = nn.Parameter(self.scale * torch.randn(embed_size, embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(embed_size, embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(embed_size))
        
        # Frequency-domain MLP parameters for temporal mixing
        self.r2 = nn.Parameter(self.scale * torch.randn(embed_size, embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(embed_size, embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(embed_size))
        
        # Output projection (same as tslib)
        self.fc = nn.Sequential(
            nn.Linear(num_timesteps_input * embed_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_timesteps_output)
        )
        
        # Move to device
        self.to(device)
        
    def tokenEmb(self, x):
        """
        Embed each value with learnable embeddings.
        
        Args:
            x: [B, L, N] input tensor
        Returns:
            embedded: [B, N, L, D] where D is embed_size
        """
        # x: [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)
        # [B, N, L] -> [B, N, L, 1]
        x = x.unsqueeze(3)
        # Multiply with embeddings: [B, N, L, 1] * [1, D] -> [B, N, L, D]
        return x * self.embeddings
    
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        """
        Frequency-domain MLP.
        
        Applies complex-valued linear transformation in frequency domain.
        
        Args:
            B: Batch size
            nd: Number of elements in non-FFT dimension
            dimension: Dimension along which FFT was taken
            x: Complex tensor after FFT
            r, i: Real and imaginary weight matrices
            rb, ib: Real and imaginary biases
        Returns:
            y: Complex tensor after MLP
        """
        # Initialize output tensors
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        
        # Complex-valued linear: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - 
            torch.einsum('bijd,dd->bijd', x.imag, i) + rb
        )
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + 
            torch.einsum('bijd,dd->bijd', x.real, i) + ib
        )
        
        # Stack and apply soft shrinkage for sparsity
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        
        return y
    
    def MLP_temporal(self, x, B, N, L):
        """
        Frequency temporal learner.
        
        Applies FFT along time dimension, MLP in frequency domain, then IFFT back.
        """
        # x: [B, N, L, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.num_timesteps_input, dim=2, norm='ortho')
        return x
    
    def MLP_channel(self, x, B, N, L):
        """
        Frequency channel learner.
        
        Applies FFT along channel dimension, MLP in frequency domain, then IFFT back.
        """
        # x: [B, N, L, D] -> [B, L, N, D]
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.total_channels, dim=2, norm='ortho')
        x = x.permute(0, 2, 1, 3)  # [B, N, L, D]
        return x
    
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
        # import ipdb; ipdb.set_trace()
        # Handle 4D input (spatiotemporal): flatten nodes and features
        if is_spatiotemporal:
            batch, time, nodes, features = x.shape
            x = x.reshape(batch, time, nodes * features)  # [B, L, N*F]
        else:
            nodes, features = 1, x.shape[-1]
        
        B, L, N = x.shape
        
        # Embed values: [B, N, L, D]
        x = self.tokenEmb(x)
        bias = x  # Residual connection
        
        # Channel mixing (optional)
        if not self.channel_independence:
            x = self.MLP_channel(x, B, N, L)
        
        # Temporal mixing
        x = self.MLP_temporal(x, B, N, L)
        
        # Add residual
        x = x + bias
        
        # Output projection: [B, N, L*D] -> [B, N, H] -> [B, H, N] (same as tslib)
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # [B, H, N]
        
        # x: (batch, pred_len, nodes*features)
        if is_spatiotemporal:
            # Reshape to (batch, pred_len, nodes, features)
            x = x.reshape(batch, self.num_timesteps_output, nodes, features)
            # Take last feature (target) for each node: (batch, pred_len, nodes)
            out = x[:, :, :, -1]
            # Transpose to (batch, nodes, pred_len) to match target shape
            out = out.permute(0, 2, 1)
        else:
            # For temporal data, take last column (target)
            out = x[:, :, -1]  # [B, H]
        
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


class FreTSLargeModel(FreTSModel):
    """Larger FreTS variant with larger embeddings."""
    
    def __init__(
        self,
        num_features: int,
        num_timesteps_input: int,
        num_timesteps_output: int,
        num_nodes: int = None,
        embed_size: int = 256,
        hidden_size: int = 512,
        channel_independence: bool = False,
        sparsity_threshold: float = 0.01,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            num_nodes=num_nodes,
            embed_size=embed_size,
            hidden_size=hidden_size,
            channel_independence=channel_independence,
            sparsity_threshold=sparsity_threshold,
            device=device,
            **kwargs
        )


class FreTSSmallModel(FreTSModel):
    """Smaller FreTS variant for faster training."""
    
    def __init__(
        self,
        num_features: int,
        num_timesteps_input: int,
        num_timesteps_output: int,
        num_nodes: int = None,
        embed_size: int = 64,
        hidden_size: int = 128,
        channel_independence: bool = False,
        sparsity_threshold: float = 0.01,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            num_nodes=num_nodes,
            embed_size=embed_size,
            hidden_size=hidden_size,
            channel_independence=channel_independence,
            sparsity_threshold=sparsity_threshold,
            device=device,
            **kwargs
        )
