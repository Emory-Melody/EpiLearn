"""
TimesFM: Time Series Foundation Model by Google Research

TimesFM is a decoder-only foundation model for time-series forecasting pretrained
on a large corpus of real-world and synthetic time series data.  It performs
zero-shot forecasting on unseen time series with strong out-of-the-box accuracy.

This wrapper supports **TimesFM 2.0** (the ``timesfm`` PyPI package ≤ 1.3.0) with
the ``google/timesfm-2.0-500m-pytorch`` checkpoint.

Reference:
    Das et al. "A decoder-only foundation model for time-series forecasting"
    (ICML 2024)  https://arxiv.org/abs/2310.10688

Available checkpoints (v2.0):
    - google/timesfm-2.0-500m-pytorch  (500M params, context ≤ 2048)
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from copy import deepcopy

from .base import BaseModel


class TimesFMModel(BaseModel):
    """
    TimesFM (Google) wrapper for EpiLearn — zero-shot time series forecasting.

    Parameters
    ----------
    num_features : int
        Number of features in the input data.
    num_timesteps_input : int
        Number of input timesteps (lookback window).
    num_timesteps_output : int
        Number of output timesteps to predict (horizon).
    model_name : str, optional
        HuggingFace checkpoint id.
        Default: ``'google/timesfm-2.0-500m-pytorch'``.
    freq : int, optional
        Frequency indicator: 0 = high (≤ daily), 1 = medium (weekly / monthly),
        2 = low (quarterly +).  Default: ``1`` (weekly epidemiological data).
    device : str, optional
        ``'cpu'`` or ``'cuda'``.  Default: ``'cpu'``.

    Returns
    -------
    torch.Tensor
        Point forecasts of shape ``(batch_size, num_timesteps_output)``.
    """

    def __init__(
        self,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        model_name="google/timesfm-2.0-500m-pytorch",
        freq=1,
        device="cpu",
        **kwargs,
    ):
        super(TimesFMModel, self).__init__(device=device)

        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.model_name = model_name
        self.freq = freq

        # Lazy loading
        self._tfm = None
        self._model_loaded = False
        self._rki_correct = bool(kwargs.get('rki_correct', False))
        self._rki_cdf = None
        self._nowcast = bool(kwargs.get('nowcast', False))

    # ------------------------------------------------------------------
    def _load_model(self):
        """Lazy-load the TimesFM checkpoint."""
        if self._model_loaded:
            return

        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "timesfm is required for TimesFMModel. "
                "Install it with: pip install timesfm"
            ) from exc

        backend = "gpu" if "cuda" in str(self.device) else "cpu"

        # Determine model parameters based on checkpoint
        is_v2 = "2.0" in self.model_name or "500m" in self.model_name
        num_layers = 50 if is_v2 else 20
        context_len = 2048 if is_v2 else 512
        use_pos_emb = not is_v2  # v2 disables positional embedding

        self._tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=max(self.num_timesteps_output, 128),
                num_layers=num_layers,
                context_len=context_len,
                use_positional_embedding=use_pos_emb,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_name,
            ),
        )
        self._model_loaded = True

    # ------------------------------------------------------------------
    def forward(self, x, **kwargs):
        """
        Forward pass — zero-shot forecasting with TimesFM.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, num_timesteps_input, num_features)``

        Returns
        -------
        torch.Tensor
            ``(batch, num_timesteps_output)``
        """
        self._load_model()

        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # Extract univariate context (handles nowcast -1 sentinels + optional RKI correction)
        from .StatsModel import extract_context_with_rki
        context = extract_context_with_rki(x, self._rki_cdf, horizon=self.num_timesteps_output, nowcast=self._nowcast).cpu().numpy()

        # Build list-of-arrays input expected by TimesFM
        forecast_input = [context[i] for i in range(batch_size)]
        frequency_input = [self.freq] * batch_size

        # Process in smaller batches to stay within memory
        max_batch = 256
        all_point_forecasts = []

        for start in range(0, batch_size, max_batch):
            end = min(start + max_batch, batch_size)
            batch_input = forecast_input[start:end]
            batch_freq = frequency_input[start:end]

            point_forecast, _ = self._tfm.forecast(
                batch_input,
                freq=batch_freq,
            )
            # point_forecast: numpy array of shape (n, horizon_len)
            all_point_forecasts.append(point_forecast)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        point_forecast = np.concatenate(all_point_forecasts, axis=0)

        # Trim to requested horizon (TimesFM may produce longer output)
        point_forecast = point_forecast[:, : self.num_timesteps_output]

        output = torch.tensor(point_forecast, dtype=torch.float32)
        return output.to(self.device)

    # ------------------------------------------------------------------
    def fit(
        self,
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
        loss="mse",
        epochs=1,
        batch_size=10,
        lr=1e-3,
        weight_decay=0,
        initialize=True,
        verbose=False,
        patience=10,
        **kwargs,
    ):
        """Zero-shot — just ensure the model is loaded."""
        self._load_model()
        if self._rki_correct and train_input is not None and train_target is not None:
            from .StatsModel import learn_nowcast_cdf
            X = train_input.cpu().numpy() if hasattr(train_input, 'cpu') else np.asarray(train_input)
            Y = train_target.cpu().numpy() if hasattr(train_target, 'cpu') else np.asarray(train_target)
            if X.ndim == 3 and X.shape[2] > 1 and np.any(X[0] == -1):
                self._rki_cdf = learn_nowcast_cdf(X, Y)
        if verbose:
            print(f"TimesFM model loaded: {self.model_name}")
            print("Using zero-shot inference (no fine-tuning)")

    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """Make predictions using TimesFM."""
        self.eval()
        with torch.no_grad():
            return self.forward(feature)

    def initialize(self):
        """Initialize the model (loads pretrained weights)."""
        self._load_model()
