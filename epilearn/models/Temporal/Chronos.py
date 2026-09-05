"""
Chronos: Time Series Foundation Model

Amazon's Chronos is a family of pretrained time series forecasting models based on 
language model architectures. It tokenizes time series values using scaling and 
quantization into a fixed vocabulary and trains existing transformer-based language 
model architectures on these tokenized time series.

Reference:
    Ansari et al. "Chronos: Learning the Language of Time Series" (2024)
    https://arxiv.org/abs/2403.07815

Available model sizes:
    - amazon/chronos-t5-tiny (8M parameters)
    - amazon/chronos-t5-mini (20M parameters)
    - amazon/chronos-t5-small (46M parameters)
    - amazon/chronos-t5-base (200M parameters)
    - amazon/chronos-t5-large (710M parameters)
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from .base import BaseModel


class ChronosModel(BaseModel):
    """
    Chronos Time Series Foundation Model wrapper for EpiLearn.
    
    This model uses Amazon's pretrained Chronos model for zero-shot or fine-tuned
    time series forecasting. By default, it uses the pretrained model without
    fine-tuning (zero-shot inference).
    
    Parameters
    ----------
    num_features : int
        Number of features in the input data.
    num_timesteps_input : int
        Number of input timesteps (lookback window).
    num_timesteps_output : int
        Number of output timesteps to predict (horizon).
    model_name : str, optional
        Chronos model variant to use. Default: 'amazon/chronos-t5-small'.
        Options: 'amazon/chronos-t5-tiny', 'amazon/chronos-t5-mini',
                 'amazon/chronos-t5-small', 'amazon/chronos-t5-base',
                 'amazon/chronos-t5-large'
    num_samples : int, optional
        Number of sample paths to generate for probabilistic forecasts. Default: 20.
    temperature : float, optional
        Temperature for sampling. Lower values give more deterministic outputs. Default: 1.0.
    top_k : int, optional
        Top-k sampling parameter. Default: 50.
    top_p : float, optional
        Top-p (nucleus) sampling parameter. Default: 1.0.
    device : str, optional
        Device to run the model on. Default: 'cpu'.
        
    Returns
    -------
    torch.Tensor
        Predicted values of shape (batch_size, num_timesteps_output).
    """
    
    def __init__(
        self,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        model_name='amazon/chronos-t5-small',
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        device='cpu',
        **kwargs
    ):
        super(ChronosModel, self).__init__(device=device)
        
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.model_name = model_name
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Lazy loading of the pipeline
        self._pipeline = None
        self._model_loaded = False
        self._rki_correct = bool(kwargs.get('rki_correct', False))
        self._rki_cdf = None
        self._nowcast = bool(kwargs.get('nowcast', False))

    def _load_model(self):
        """Lazy load the Chronos pipeline."""
        if not self._model_loaded:
            try:
                from chronos import ChronosPipeline
                
                # Determine torch dtype based on device
                torch_dtype = torch.float32
                if 'cuda' in str(self.device):
                    torch_dtype = torch.bfloat16  # Use bfloat16 for GPU efficiency
                
                self._pipeline = ChronosPipeline.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=torch_dtype,
                )
                self._model_loaded = True
                # print(f"Loaded Chronos model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "chronos-forecasting is required for ChronosModel. "
                    "Install it with: pip install chronos-forecasting"
                )
    
    def forward(self, x, **kwargs):
        """
        Forward pass using Chronos for prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_timesteps_input, num_features).
            Only the first feature (target column) is used for prediction.
            
        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch_size, num_timesteps_output).
        """
        import gc
        self._load_model()
        # import ipdb; ipdb.set_trace()
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size = x.shape[0]
        
        # Extract univariate context (handles nowcast -1 sentinels + optional RKI correction)
        # horizon truncation: exclude target window so forecast aligns with targets
        from .StatsModel import extract_context_with_rki
        context = extract_context_with_rki(x, self._rki_cdf, horizon=self.num_timesteps_output, nowcast=self._nowcast).cpu()
        # Process in smaller batches to avoid OOM
        max_batch_size = 32  # Limit batch size to prevent memory issues
        outputs = []
        
        for i in range(0, batch_size, max_batch_size):
            batch_context = context[i:i + max_batch_size]

            # Seed before each stochastic sampling call for reproducibility
            torch.manual_seed(42 + i)

            # Generate forecast for this batch
            forecast = self._pipeline.predict(
                inputs=batch_context,
                prediction_length=self.num_timesteps_output,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            
            # Take median of samples as point forecast
            # forecast shape: (batch_size, num_samples, prediction_length)
            batch_output = torch.median(forecast, dim=1).values
            outputs.append(batch_output)
            
            # Clean up memory after each batch
            del forecast
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        output = torch.cat(outputs, dim=0)
        return output.to(self.device)
    
    def fit(self, 
            train_input, 
            train_target, 
            train_states=None, 
            train_graph=None, 
            train_dynamic_graph=None,
            val_input=None, 
            val_target=None,
            val_states=None, 
            val_graph=None, 
            val_dynamic_graph=None,
            loss='mse', 
            epochs=1, 
            batch_size=10,
            lr=1e-3, 
            weight_decay=0,
            initialize=True, 
            verbose=False, 
            patience=10, 
            **kwargs):
        """
        Fit method for Chronos (zero-shot, no training required).
        
        Chronos is a pretrained model and performs zero-shot forecasting.
        This method simply ensures the model is loaded.
        """
        # Just load the model - no training for zero-shot
        self._load_model()

        # Learn RKI CDF for nowcast correction
        if self._rki_correct and train_input is not None and train_target is not None:
            from .StatsModel import learn_nowcast_cdf
            X = train_input.cpu().numpy() if hasattr(train_input, 'cpu') else np.asarray(train_input)
            Y = train_target.cpu().numpy() if hasattr(train_target, 'cpu') else np.asarray(train_target)
            if X.ndim == 3 and X.shape[2] > 1 and np.any(X[0] == -1):
                self._rki_cdf = learn_nowcast_cdf(X, Y)

        if verbose:
            print(f"Chronos model loaded: {self.model_name}")
            print("Using zero-shot inference (no fine-tuning)")
    
    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Make predictions using the Chronos model.
        
        Parameters
        ----------
        feature : torch.Tensor
            Input features of shape (batch_size, num_timesteps_input, num_features).
            
        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, num_timesteps_output).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(feature)
    
    def initialize(self):
        """Initialize the model (loads pretrained weights)."""
        self._load_model()


class ChronosBoltModel(BaseModel):
    """
    Chronos-Bolt: Faster variant of Chronos using encoder-only architecture.
    
    Chronos-Bolt models are more efficient variants that provide significant speedup
    while maintaining competitive forecasting accuracy.
    
    Parameters
    ----------
    num_features : int
        Number of features in the input data.
    num_timesteps_input : int
        Number of input timesteps (lookback window).
    num_timesteps_output : int
        Number of output timesteps to predict (horizon).
    model_name : str, optional
        Chronos-Bolt model variant. Default: 'amazon/chronos-bolt-small'.
        Options: 'amazon/chronos-bolt-tiny', 'amazon/chronos-bolt-mini',
                 'amazon/chronos-bolt-small', 'amazon/chronos-bolt-base'
    device : str, optional
        Device to run the model on. Default: 'cpu'.
    """
    
    def __init__(
        self,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        model_name='amazon/chronos-bolt-small',
        device='cpu',
        **kwargs
    ):
        super(ChronosBoltModel, self).__init__(device=device)
        
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.model_name = model_name
        
        self._pipeline = None
        self._model_loaded = False
        self._rki_correct = bool(kwargs.get('rki_correct', False))
        self._rki_cdf = None
        self._nowcast = bool(kwargs.get('nowcast', False))

    def _load_model(self):
        """Lazy load the Chronos-Bolt pipeline."""
        if not self._model_loaded:
            try:
                from chronos import ChronosPipeline
                
                torch_dtype = torch.float32
                if 'cuda' in str(self.device):
                    torch_dtype = torch.bfloat16
                
                self._pipeline = ChronosPipeline.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=torch_dtype,
                )
                self._model_loaded = True
                print(f"Loaded Chronos-Bolt model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "chronos-forecasting is required. "
                    "Install it with: pip install chronos-forecasting"
                )
    
    def forward(self, x, **kwargs):
        """Forward pass using Chronos-Bolt."""
        self._load_model()
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        # Extract univariate context (handles nowcast -1 sentinels + optional RKI correction)
        from .StatsModel import extract_context_with_rki
        context = extract_context_with_rki(x, self._rki_cdf, horizon=self.num_timesteps_output, nowcast=self._nowcast).cpu()

        predictions = []
        for i in range(batch_size):
            series = context[i]
            
            # Chronos-Bolt returns quantile forecasts directly
            forecast = self._pipeline.predict(
                context=series,
                prediction_length=self.num_timesteps_output,
            )
            
            # Take median (0.5 quantile) as point forecast
            if forecast.dim() == 2:
                # Multiple quantiles returned
                point_forecast = forecast[forecast.shape[0] // 2]
            else:
                point_forecast = forecast
            
            predictions.append(point_forecast)
        
        output = torch.stack(predictions, dim=0)
        return output.to(self.device)
    
    def fit(self, train_input, train_target, **kwargs):
        """Zero-shot inference - just load the model."""
        self._load_model()
        if self._rki_correct and train_input is not None and train_target is not None:
            from .StatsModel import learn_nowcast_cdf
            X = train_input.cpu().numpy() if hasattr(train_input, 'cpu') else np.asarray(train_input)
            Y = train_target.cpu().numpy() if hasattr(train_target, 'cpu') else np.asarray(train_target)
            if X.ndim == 3 and X.shape[2] > 1 and np.any(X[0] == -1):
                self._rki_cdf = learn_nowcast_cdf(X, Y)
    
    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            return self.forward(feature)
    
    def initialize(self):
        """Initialize the model."""
        self._load_model()
