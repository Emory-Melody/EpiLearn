import torch
import torch.nn as nn
import torch.nn.init as init
import math
from .base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Handle both even and odd d_model values
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model, cos indices are fewer than sin indices
        cos_indices = d_model // 2
        pe[:, 1::2] = torch.cos(position * div_term[:cos_indices])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts time series into patches
    """
    def __init__(self, patch_len, stride, d_model, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Linear projection for each patch
        self.linear = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch_size, num_features, seq_len)
        Returns: (batch_size, num_features, num_patches, d_model)
        """
        batch_size, num_features, seq_len = x.shape
        
        # Calculate number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        
        # Extract patches
        patches = []
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, :, start_idx:end_idx]  # (batch_size, num_features, patch_len)
            patches.append(patch)
        
        # Stack patches: (batch_size, num_features, num_patches, patch_len)
        patches = torch.stack(patches, dim=2)
        
        # Project patches to d_model dimension
        # Reshape for linear layer: (batch_size * num_features * num_patches, patch_len)
        patches_flat = patches.reshape(-1, self.patch_len)
        patches_embedded = self.linear(patches_flat)  # (batch_size * num_features * num_patches, d_model)
        
        # Reshape back: (batch_size, num_features, num_patches, d_model)
        patches_embedded = patches_embedded.reshape(batch_size, num_features, num_patches, self.d_model)
        patches_embedded = self.dropout(patches_embedded)
        
        return patches_embedded, num_patches


class FlattenHead(nn.Module):
    """
    Flatten and project to output
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super(FlattenHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: (batch_size, n_vars, num_patches, d_model)
        Returns: (batch_size, target_window)
        """
        x = self.flatten(x)  # (batch_size, n_vars, num_patches * d_model)
        x = self.linear(x)  # (batch_size, n_vars, target_window)
        x = self.dropout(x)

        # Average across features
        x = x.mean(dim=1)  # (batch_size, target_window)
        return x


class PatchTSTModel(BaseModel):
    """
    PatchTST: A Time Series Forecasting Model using Patching and Transformers
    
    This implementation follows the PatchTST architecture with channel independence,
    where each feature/channel is processed independently through the transformer.

    Parameters
    ----------
    num_features : int
        Number of features in each timestep of the input data.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    patch_len : int, optional
        Length of each patch. Default: 4.
    stride : int, optional
        Stride for patch extraction. Default: 1.
    d_model : int, optional
        Dimension of the model (embedding dimension). Default: 128.
    n_heads : int, optional
        Number of attention heads in transformer. Default: 8.
    num_layers : int, optional
        Number of transformer encoder layers. Default: 1.
    dim_feedforward : int, optional
        Dimension of feedforward network in transformer. Default: 256.
    dropout : float, optional
        Dropout rate. Default: 0.1.
    activation : str, optional
        Activation function ('relu' or 'gelu'). Default: 'gelu'.
    norm_first : bool, optional
        If True, layer norm is applied before attention/feedforward. Default: True.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output) representing the predicted values 
        for the future timesteps.
    """
    
    def __init__(self,
                 num_features,
                 num_timesteps_input,
                 num_timesteps_output,
                 patch_len=4,
                 stride=1,
                 d_model=128,
                 n_heads=8,  # FIXED: changed from nhead to n_heads for config compatibility
                 num_layers=1,
                 dim_feedforward=16,
                 dropout=0.1,
                 activation='gelu',
                 norm_first=True,
                 device='cpu',
                 **kwargs):
        super(PatchTSTModel, self).__init__(device=device)

        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        # CRITICAL FIX: Ensure patch_len doesn't exceed input length
        # If patch_len > num_timesteps_input, adjust it to valid value
        if patch_len > num_timesteps_input:
            patch_len = max(1, num_timesteps_input // 2)  # Use half the input length
            print(f"WARNING: patch_len adjusted to {patch_len} (was > lookback={num_timesteps_input})")

        # Ensure stride doesn't exceed patch_len
        if stride > patch_len:
            stride = max(1, patch_len // 2)
            print(f"WARNING: stride adjusted to {stride} (was > patch_len={patch_len})")

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.nhead = n_heads  # Map n_heads to nhead for transformer
        self.num_layers = num_layers

        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, dropout)

        # Calculate number of patches for positional encoding
        self.num_patches = (num_timesteps_input - patch_len) // stride + 1

        # Sanity check: ensure num_patches is positive
        if self.num_patches <= 0:
            raise ValueError(
                f"Invalid patching configuration: num_patches={self.num_patches} "
                f"(lookback={num_timesteps_input}, patch_len={patch_len}, stride={stride}). "
                f"Ensure patch_len <= lookback."
            )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 10)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.nhead,  # Use self.nhead (already set from n_heads)
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.head = FlattenHead(
            n_vars=num_features,
            nf=self.num_patches * d_model,
            target_window=num_timesteps_output,
            head_dropout=dropout
        )
        
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
        
        # Transpose to (batch_size, num_features, num_timesteps_input)
        x = x.transpose(1, 2)
        
        batch_size = x.shape[0]
        
        # Patch embedding: (batch_size, num_features, num_patches, d_model)
        x, num_patches = self.patch_embedding(x)
        
        # Process each feature independently (channel independence)
        # Reshape to process all features in batch: (batch_size * num_features, num_patches, d_model)
        x = x.reshape(batch_size * self.num_features, num_patches, self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size * num_features, num_patches, d_model)
        
        # Reshape back: (batch_size, num_features, num_patches, d_model)
        x = x.reshape(batch_size, self.num_features, num_patches, self.d_model)
        
        # Apply output head
        output = self.head(x)  # (batch_size, num_timesteps_output)
        
        return output
    
    def initialize(self):
        """
        Initialize model parameters
        """
        # Initialize patch embedding
        init.xavier_uniform_(self.patch_embedding.linear.weight)
        if self.patch_embedding.linear.bias is not None:
            init.constant_(self.patch_embedding.linear.bias, 0)
        
        # Initialize transformer encoder
        for layer in self.transformer_encoder.layers:
            # Self-attention
            for param in layer.self_attn.parameters():
                if param.dim() > 1:
                    init.xavier_uniform_(param)
                else:
                    init.constant_(param, 0)
            
            # Feedforward
            init.xavier_uniform_(layer.linear1.weight)
            init.constant_(layer.linear1.bias, 0)
            init.xavier_uniform_(layer.linear2.weight)
            init.constant_(layer.linear2.bias, 0)
            
            # Layer norms
            if hasattr(layer, 'norm1'):
                init.constant_(layer.norm1.weight, 1)
                init.constant_(layer.norm1.bias, 0)
            if hasattr(layer, 'norm2'):
                init.constant_(layer.norm2.weight, 1)
                init.constant_(layer.norm2.bias, 0)
        
        # Initialize output head
        init.xavier_uniform_(self.head.linear.weight)
        if self.head.linear.bias is not None:
            init.constant_(self.head.linear.bias, 0)
