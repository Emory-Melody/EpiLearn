"""
MOIRAI: Masked Encoder-based Universal Time Series Forecasting Transformer

MOIRAI (Masked encoder-based universal time series forecasting transformer) is a 
foundation model for time series forecasting that supports variable context lengths,
prediction horizons, and frequencies. It uses a masked encoder architecture with
any-variate attention for handling multivariate time series.

Reference:
    Woo et al. "Unified Training of Universal Time Series Forecasting Transformers" (2024)
    https://arxiv.org/abs/2402.02592

Available model sizes:
    - Salesforce/moirai-1.0-R-small (14M parameters)
    - Salesforce/moirai-1.0-R-base (91M parameters)  
    - Salesforce/moirai-1.0-R-large (311M parameters)
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from .base import BaseModel


class MoiraiModel(BaseModel):
    """
    MOIRAI Time Series Foundation Model wrapper for EpiLearn.
    
    This model uses Salesforce's pretrained MOIRAI model for zero-shot
    time series forecasting. MOIRAI supports multivariate time series
    and variable prediction horizons.
    
    Parameters
    ----------
    num_features : int
        Number of features in the input data.
    num_timesteps_input : int
        Number of input timesteps (lookback window).
    num_timesteps_output : int
        Number of output timesteps to predict (horizon).
    model_name : str, optional
        MOIRAI model variant to use. Default: 'Salesforce/moirai-1.0-R-small'.
        Options: 'Salesforce/moirai-1.0-R-small', 'Salesforce/moirai-1.0-R-base',
                 'Salesforce/moirai-1.0-R-large'
    num_samples : int, optional
        Number of sample paths for probabilistic forecasts. Default: 100.
    patch_size : str or int, optional
        Patch size for the model. Default: 'auto'.
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
        model_name='Salesforce/moirai-1.0-R-small',
        num_samples=100,
        patch_size='auto',
        device='cpu',
        **kwargs
    ):
        super(MoiraiModel, self).__init__(device=device)
        
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.model_name = model_name
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.freq = kwargs.get('freq', 'D')  # data frequency for GluonTS (e.g. 'D', 'W', 'M')

        # Lazy loading
        self._module = None
        self._forecast_model = None
        self._predictor = None
        self._model_loaded = False
        self._rki_correct = bool(kwargs.get('rki_correct', False))
        self._rki_cdf = None
        self._nowcast = bool(kwargs.get('nowcast', False))

    def _load_model(self):
        """Lazy load the MOIRAI model."""
        if not self._model_loaded:
            try:
                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
                
                # Load pretrained module
                self._module = MoiraiModule.from_pretrained(self.model_name)
                self._module = self._module.to(self.device)
                self._module.eval()
                
                # Create forecast model
                self._forecast_model = MoiraiForecast(
                    module=self._module,
                    prediction_length=self.num_timesteps_output,
                    context_length=self.num_timesteps_input,
                    target_dim=1,  # univariate prediction
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                    num_samples=self.num_samples,
                    patch_size=self.patch_size,
                )
                
                # Create predictor
                device_str = 'cuda' if 'cuda' in str(self.device) else 'cpu'
                self._predictor = self._forecast_model.create_predictor(
                    batch_size=32,
                    device=device_str
                )
                
                self._model_loaded = True
                print(f"Loaded MOIRAI model: {self.model_name}")
                
            except ImportError as ie:
                raise ImportError(
                    "uni2ts is required for MoiraiModel. "
                    "Install it with: pip install uni2ts"
                ) from ie
                
    def forward(self, x, **kwargs):
        """
        Forward pass using MOIRAI for prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_timesteps_input, num_features).
            
        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch_size, num_timesteps_output).
        """
        import gc
        self._load_model()
        
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Extract univariate context (handles nowcast -1 sentinels + optional RKI correction)
        from .StatsModel import extract_context_with_rki
        context = extract_context_with_rki(x, self._rki_cdf, horizon=self.num_timesteps_output, nowcast=self._nowcast).cpu().numpy()

        # Create GluonTS dataset
        from gluonts.dataset.common import ListDataset
        import pandas as pd
        
        # Process in smaller batches to avoid OOM
        max_batch_size = 32
        all_predictions = []
        
        for batch_start in range(0, batch_size, max_batch_size):
            batch_end = min(batch_start + max_batch_size, batch_size)
            batch_context = context[batch_start:batch_end]

            # Seed before each stochastic sampling call for reproducibility
            torch.manual_seed(42 + batch_start)
            np.random.seed(42 + batch_start)

            # Create list of time series for prediction
            data_list = []
            for i in range(len(batch_context)):
                data_list.append({
                    'target': batch_context[i],
                    'start': pd.Timestamp('2020-01-01')
                })
            
            test_data = ListDataset(data_list, freq=self.freq)
            
            # Make predictions for this batch
            batch_preds = []
            for forecast in self._predictor.predict(test_data):
                # Use median of samples as point forecast
                pred = forecast.median
                batch_preds.append(torch.tensor(pred, dtype=torch.float32))
            
            all_predictions.extend(batch_preds)
            
            # Clean up memory after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        output = torch.stack(all_predictions, dim=0)  # (batch_size, num_timesteps_output)
        
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
        Fit method for MOIRAI (zero-shot, no training required).
        
        MOIRAI is a pretrained model and performs zero-shot forecasting.
        """
        self._load_model()

        if self._rki_correct and train_input is not None and train_target is not None:
            from .StatsModel import learn_nowcast_cdf
            X = train_input.cpu().numpy() if hasattr(train_input, 'cpu') else np.asarray(train_input)
            Y = train_target.cpu().numpy() if hasattr(train_target, 'cpu') else np.asarray(train_target)
            if X.ndim == 3 and X.shape[2] > 1 and np.any(X[0] == -1):
                self._rki_cdf = learn_nowcast_cdf(X, Y)

        if verbose:
            print(f"MOIRAI model loaded: {self.model_name}")
            print("Using zero-shot inference (no fine-tuning)")
    
    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Make predictions using the MOIRAI model.
        
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


class MoiraiLargeModel(MoiraiModel):
    """MOIRAI Large model variant (311M parameters)."""
    
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, 
                 device='cpu', **kwargs):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            model_name='Salesforce/moirai-1.0-R-large',
            device=device,
            **kwargs
        )


class MoiraiBaseModel(MoiraiModel):
    """MOIRAI Base model variant (91M parameters)."""
    
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 device='cpu', **kwargs):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            model_name='Salesforce/moirai-1.0-R-base',
            device=device,
            **kwargs
        )
