"""
MOMENT: A Family of Open Time-series Foundation Models

MOMENT is a family of open-source foundation models for general-purpose time series
analysis pre-trained on the Time-series Pile. It supports forecasting, classification,
anomaly detection, and imputation tasks.

For zero-shot forecasting, MOMENT uses its pre-trained reconstruction head via the
``short_forecast`` method, which masks the tail of the input series and reconstructs
the missing values — no fine-tuning required.

Reference:
    Goswami et al. "MOMENT: A Family of Open Time-series Foundation Models" (ICML 2024)
    https://arxiv.org/abs/2402.03885

Available model sizes:
    - AutonLab/MOMENT-1-small
    - AutonLab/MOMENT-1-base
    - AutonLab/MOMENT-1-large  (default, 341M parameters)
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from copy import deepcopy

from .base import BaseModel

# MOMENT fixed context length
_MOMENT_SEQ_LEN = 512


class MomentModel(BaseModel):
    """
    MOMENT Time Series Foundation Model wrapper for EpiLearn.

    Uses the pre-trained reconstruction head with ``short_forecast`` for
    zero-shot time series forecasting (no fine-tuning needed).

    Parameters
    ----------
    num_features : int
        Number of features in the input data.
    num_timesteps_input : int
        Number of input timesteps (lookback window).
    num_timesteps_output : int
        Number of output timesteps to predict (horizon).
    model_name : str, optional
        HuggingFace model id. Default: ``'AutonLab/MOMENT-1-large'``.
    device : str, optional
        Device to run the model on. Default: ``'cpu'``.

    Returns
    -------
    torch.Tensor
        Predicted values of shape ``(batch_size, num_timesteps_output)``.
    """

    def __init__(
        self,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        model_name="AutonLab/MOMENT-1-large",
        device="cpu",
        **kwargs,
    ):
        super(MomentModel, self).__init__(device=device)

        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.model_name = model_name

        # Lazy loading
        self._moment_model = None
        self._model_loaded = False
        self._rki_correct = bool(kwargs.get('rki_correct', False))
        self._rki_cdf = None
        self._nowcast = bool(kwargs.get('nowcast', False))

    def _load_model(self):
        """Lazy-load the MOMENT pipeline in reconstruction mode."""
        if self._model_loaded:
            return
        try:
            from momentfm import MOMENTPipeline

            # Load in reconstruction mode so the pre-trained head is available
            # for zero-shot forecasting via short_forecast().
            self._moment_model = MOMENTPipeline.from_pretrained(
                self.model_name,
                model_kwargs={"task_name": "reconstruction"},
            )
            self._moment_model.init()

            # Move to device
            device_str = str(self.device)
            if "cuda" in device_str:
                self._moment_model = self._moment_model.to(self.device)
            self._moment_model.eval()
            self._model_loaded = True

        except ImportError as exc:
            raise ImportError(
                "momentfm is required for MomentModel. "
                "Install it with: pip install momentfm"
            ) from exc

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_input(context: torch.Tensor, seq_len: int = _MOMENT_SEQ_LEN):
        """
        Pad or truncate *context* to ``seq_len`` and build an input mask.

        Parameters
        ----------
        context : torch.Tensor
            Shape ``(batch, T)`` – univariate context values.

        Returns
        -------
        x_enc : torch.Tensor   shape ``(batch, 1, seq_len)``
        input_mask : torch.Tensor   shape ``(batch, seq_len)``
        """
        batch, T = context.shape
        if T >= seq_len:
            # Truncate – keep the most recent values
            x_enc = context[:, -seq_len:].unsqueeze(1)  # (B, 1, seq_len)
            input_mask = torch.ones(batch, seq_len, device=context.device)
        else:
            # Left-pad with zeros
            pad_len = seq_len - T
            x_enc = torch.cat(
                [
                    torch.zeros(batch, pad_len, device=context.device),
                    context,
                ],
                dim=1,
            ).unsqueeze(1)  # (B, 1, seq_len)
            input_mask = torch.cat(
                [
                    torch.zeros(batch, pad_len, device=context.device),
                    torch.ones(batch, T, device=context.device),
                ],
                dim=1,
            )
        return x_enc, input_mask

    # ------------------------------------------------------------------
    # core interface
    # ------------------------------------------------------------------
    def forward(self, x, **kwargs):
        """
        Forward pass — zero-shot forecasting via MOMENT's short_forecast.

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
        context = extract_context_with_rki(x, self._rki_cdf, horizon=self.num_timesteps_output, nowcast=self._nowcast)

        # Process in smaller batches to avoid OOM
        max_batch = 32
        outputs = []

        for start in range(0, batch_size, max_batch):
            end = min(start + max_batch, batch_size)
            batch_ctx = context[start:end]

            # Move to the model's device
            device = next(self._moment_model.parameters()).device
            batch_ctx = batch_ctx.to(device).float()

            x_enc, input_mask = self._prepare_input(batch_ctx, _MOMENT_SEQ_LEN)

            with torch.no_grad():
                out = self._moment_model.short_forecast(
                    x_enc=x_enc,
                    input_mask=input_mask,
                    forecast_horizon=self.num_timesteps_output,
                )

            # out.forecast shape: (B, n_channels=1, forecast_horizon)
            forecast = out.forecast.squeeze(1)  # (B, forecast_horizon)
            outputs.append(forecast.cpu())

            del out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = torch.cat(outputs, dim=0)
        return result.to(self.device)

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
            print(f"MOMENT model loaded: {self.model_name}")
            print("Using zero-shot inference via short_forecast (no fine-tuning)")

    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """Make predictions using the MOMENT model."""
        self.eval()
        with torch.no_grad():
            return self.forward(feature)

    def initialize(self):
        """Initialize the model (loads pretrained weights)."""
        self._load_model()


class MomentSmallModel(MomentModel):
    """MOMENT Small model variant."""

    def __init__(
        self, num_features, num_timesteps_input, num_timesteps_output,
        device="cpu", **kwargs,
    ):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            model_name="AutonLab/MOMENT-1-small",
            device=device,
            **kwargs,
        )


class MomentBaseModel(MomentModel):
    """MOMENT Base model variant."""

    def __init__(
        self, num_features, num_timesteps_input, num_timesteps_output,
        device="cpu", **kwargs,
    ):
        super().__init__(
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            model_name="AutonLab/MOMENT-1-base",
            device=device,
            **kwargs,
        )
