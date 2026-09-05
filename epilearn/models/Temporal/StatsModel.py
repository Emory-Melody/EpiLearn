import numpy as np
import torch
import warnings
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
from scipy.interpolate import PchipInterpolator


# =============================================================================
# Shared nowcast utilities (used by stats models AND foundation models)
# =============================================================================

def learn_nowcast_cdf(X, Y):
    """Estimate completion CDF from (features, targets) pairs.

    Parameters
    ----------
    X : ndarray (N, L, D)  – feature matrices with -1 sentinels
    Y : ndarray (N, W)     – target (final count) vectors

    Returns
    -------
    cdf : ndarray (D,)  – monotone CDF in [0.01, 1.0]
    """
    N, L, D = X.shape
    W = Y.shape[1]
    ratio_lists = [[] for _ in range(D)]
    for i in range(N):
        for k in range(W):
            final = Y[i, k]
            if final < 1e-8:
                continue
            row = X[i, L - W + k]
            for d in range(D):
                v = row[d]
                if v != -1 and v > 0:
                    ratio_lists[d].append(v / final)

    cdf = np.ones(D, dtype=np.float64)
    for d in range(D):
        if ratio_lists[d]:
            cdf[d] = np.median(ratio_lists[d])

    # Monotone interpolation for unmeasured delays
    measured = [d for d in range(D) if ratio_lists[d]]
    if len(measured) > 1:
        vals = np.array([cdf[d] for d in measured])
        interp = PchipInterpolator(measured, vals, extrapolate=True)
        for d in range(D):
            if not ratio_lists[d]:
                cdf[d] = float(interp(d))

    for i in range(1, D):
        cdf[i] = max(cdf[i], cdf[i - 1])
    return np.clip(cdf, 0.01, 1.0)


def extract_context_with_rki(x, rki_cdf=None, horizon=0, nowcast=False):
    """Extract univariate context from multi-feature tensor.

    Parameters
    ----------
    x : torch.Tensor (batch, time, features)
    rki_cdf : ndarray (D,) or None
    horizon : int
        If > 0 and ``nowcast=True``, truncate the last ``horizon`` rows so
        that a forecasting model's W-step-ahead output aligns with the target
        window.
    nowcast : bool
        If True, interpret -1 values as unobserved sentinels and extract the
        last valid value per row.  If False (forecasting), return the last
        feature column as the target series.

    Returns
    -------
    context : torch.Tensor (batch, time) or (batch, time - horizon)
    """
    if x.shape[2] == 1:
        return x[:, :, 0]
    if not nowcast:
        return x[:, :, -1]
    # Nowcast: extract last valid (non -1) value per row
    mask = (x != -1)
    reversed_mask = mask.flip(dims=[2])
    last_valid_idx = (x.shape[2] - 1 - reversed_mask.float().argmax(dim=2))
    context = x.gather(2, last_valid_idx.unsqueeze(2).long()).squeeze(2)
    context = context * mask.any(dim=2).float()
    if rki_cdf is not None:
        cdf_t = torch.tensor(rki_cdf, dtype=context.dtype, device=context.device)
        correction = cdf_t[last_valid_idx.long().clamp(0, len(rki_cdf) - 1)]
        context = torch.where(mask.any(dim=2), context / correction.clamp(min=0.01), context)
    # Truncate target window so W-step forecast aligns with target days
    if horizon > 0 and context.shape[1] > horizon:
        context = context[:, :-horizon]
    return context


class BaseStatsModel:
    """
    Base class for statistical time series models from statsmodels.
    Provides a unified interface compatible with the epilearn framework.
    """
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 order=(1, 0), trend='c', device='cpu', p=None, q=None,
                 rki_correct=False, **kwargs):
        # Handle Optuna-style separate p, q parameters (for VARMAX)
        # If p, q are provided separately, construct order tuple
        if p is not None and q is not None:
            order = (p, q)

        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.order = order
        self.trend = trend
        self.device = device
        self.model = None
        self.rki_correct = bool(rki_correct)
        self._rki_cdf = None
        self._nowcast = bool(kwargs.get('nowcast', False))
        # Scale correction for feature-space → target-space mismatch.
        # When features and targets are normalised with different stats,
        # per-sample models (ARIMA, etc.) fit to the feature column but
        # are evaluated against the target.  fit() learns the affine map.
        self._scale = 1.0
        self._offset = 0.0

    def _prepare_data(self, data, is_tensor=True):
        """Convert tensor to numpy if needed."""
        if is_tensor and hasattr(data, 'numpy'):
            # Move to CPU first if on CUDA
            if hasattr(data, 'is_cuda') and data.is_cuda:
                data = data.cpu()
            return data.numpy()
        return data

    def _get_order(self):
        """Get proper order tuple. Override in subclasses if needed."""
        return self.order

    def _fit_single_series(self, series):
        """Fit model to a single time series. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _fit_single_series")

    def _forecast_single_series(self, series):
        """Forecast a single time series. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _forecast_single_series")

    def fit(self, train_input, train_target, train_states=None, val_input=None, val_target=None,
            train_graph=None, train_dynamic_graph=None, val_graph=None, val_dynamic_graph=None, 
            val_states=None, epochs=1000, batch_size=10, verbose=False, patience=100, 
            lr=None, weight_decay=None, loss=None, initialize=True, **kwargs):
        """
        Fit method for statistical models.
        
        Note: Statistical models (ARIMA, VARMAX) are fitted per-sample during prediction,
        so this method is essentially a no-op. Hyperparameter tuning happens via Optuna
        using validation set performance.
        
        Parameters
        ----------
        train_input : torch.Tensor
            Shape (batch_size, num_timesteps_input, num_features) - not used
        train_target : torch.Tensor
            Shape (batch_size, num_timesteps_output) - not used
        verbose : bool
            If True, prints info message
        **kwargs : dict
            Additional parameters (ignored for compatibility)
        
        Returns
        -------
        None
        """
        # Learn RKI completion CDF if requested (for nowcast data)
        if self.rki_correct and train_input is not None and train_target is not None:
            X = self._prepare_data(train_input)
            Y = self._prepare_data(train_target)
            if X.ndim == 3 and X.shape[2] > 1 and np.any(X[0] == -1):
                self._rki_cdf = learn_nowcast_cdf(X, Y)

        # Learn scale correction: features and targets may be normalised
        # with different stats (normalize_feat vs normalize_target).  We
        # regress target[:, 0] on the last-feature-column of the last
        # input timestep to learn the affine mapping.
        if train_input is not None and train_target is not None:
            X = self._prepare_data(train_input)
            Y = self._prepare_data(train_target)
            if X.ndim == 3 and X.shape[2] >= 1 and Y.ndim == 2:
                feat_y = X[:, -1, -1]  # last timestep, last feature (= y)
                tgt_y = Y[:, 0]        # first target step
                feat_std = np.std(feat_y)
                tgt_std = np.std(tgt_y)
                if feat_std > 1e-8 and tgt_std > 1e-8:
                    self._scale = tgt_std / feat_std
                    self._offset = np.mean(tgt_y) - self._scale * np.mean(feat_y)

        # Statistical models are fitted per-sample during predict()
        if verbose:
            print(f"Statistical model initialized with order={self.order}. "
                  f"Fitting happens per-sample during prediction.")
        return None

    def predict(self, feature, graph=None, states=None, dynamic_graph=None, **kwargs):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        feature : torch.Tensor
            Shape (batch_size, num_timesteps_input, num_features)
        **kwargs : dict
            Additional parameters (ignored for compatibility)
        
        Returns
        -------
        torch.Tensor
            Shape (batch_size, num_timesteps_output)
        """
        test_input = self._prepare_data(feature)
        all_forecasts = []
        order_tuple = self._get_order()
        
        for data in test_input:
            series = self._extract_series(data)
            
            if self._is_series_valid(series, order_tuple):
                forecast = self._forecast_single_series(series)
                all_forecasts.append(forecast)
            else:
                all_forecasts.append(np.array([np.nan] * self.num_timesteps_output))
        
        # Convert to tensor and apply feature→target scale correction
        all_forecasts_array = np.array(all_forecasts)
        result = all_forecasts_array * self._scale + self._offset
        return torch.tensor(result, dtype=torch.float32)

    def _extract_series(self, data):
        """Extract univariate target series from potentially multivariate data.

        Forecasting: uses the last feature column (target variable).
        Nowcasting: extracts the last valid (non -1) value per timestep and
        truncates the target window so ARIMA forecasts W steps from history.
        """
        if self.num_features > 1 and len(data.shape) > 1:
            if self._nowcast:
                cdf = self._rki_cdf
                L = len(data)
                W = self.num_timesteps_output
                n = L - W if L > W else L
                series = np.full(n, 0.0)
                for t in range(n):
                    valid_mask = data[t] != -1
                    if np.any(valid_mask):
                        d = np.where(valid_mask)[0][-1]
                        v = data[t, d]
                        if cdf is not None and d < len(cdf):
                            v = v / cdf[d]
                        series[t] = v
                return series
            return data[:, -1]
        return data.flatten()

    def _is_series_valid(self, series, order_tuple):
        """Check if series has enough data points for the given order."""
        return series.shape[0] > max(order_tuple)

    def to(self, device):
        """Compatibility method for device placement."""
        self.device = device
        return self


class VARMAXModel(BaseStatsModel):
    """
    Vector Autoregression Moving-Average with eXogenous variables (VARMAX) Model.
    Wrapper around statsmodels VARMAX for multivariate time series.

    Parameters
    ----------
    num_features : int
        Number of features in each timestep
    num_timesteps_input : int
        Number of input timesteps
    num_timesteps_output : int
        Number of output timesteps to predict
    order : tuple
        (p, q) order of the VARMAX model
    trend : str
        Trend parameter ('c' for constant, 'n' for none)
    """
    
    def _extract_series(self, data):
        """VARMAX can handle multivariate data, so return as-is."""
        return data

    def _fit_single_series(self, series):
        """Fit VARMAX model and return forecast averaged across features.
        
        Handles numerical stability issues (LinAlgError) by falling back to
        a simpler model or returning NaN forecast.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = VARMAX(series, order=self.order, trend=self.trend)
                results = model.fit(disp=False, maxiter=20)
                forecast = results.forecast(steps=self.num_timesteps_output)
                # Average across features for univariate output
                return forecast.mean(axis=1) if len(forecast.shape) > 1 else forecast
            except np.linalg.LinAlgError:
                # Numerical stability issue - try with no trend
                try:
                    model = VARMAX(series, order=self.order, trend='n')
                    results = model.fit(disp=False, maxiter=20)
                    forecast = results.forecast(steps=self.num_timesteps_output)
                    return forecast.mean(axis=1) if len(forecast.shape) > 1 else forecast
                except (np.linalg.LinAlgError, Exception):
                    # Fall back to last value extrapolation
                    last_val = series[-1].mean() if len(series.shape) > 1 else series[-1]
                    return np.full(self.num_timesteps_output, last_val)
            except Exception:
                # Any other error - fall back to last value extrapolation
                last_val = series[-1].mean() if len(series.shape) > 1 else series[-1]
                return np.full(self.num_timesteps_output, last_val)

    def _forecast_single_series(self, series):
        """Forecast using VARMAX."""
        return self._fit_single_series(series)


class ARIMAModel(BaseStatsModel):
    """
    Autoregressive Integrated Moving Average (ARIMA) Model.
    Wrapper around statsmodels ARIMA for univariate time series.

    Parameters
    ----------
    num_features : int
        Number of features (only first feature is used for univariate ARIMA)
    num_timesteps_input : int
        Number of input timesteps
    num_timesteps_output : int
        Number of output timesteps to predict
    order : tuple
        (p, d, q) order of the ARIMA model. If only (p, d) provided, q defaults to 0
    seasonal_order : tuple, optional
        (P, D, Q, s) seasonal order of the ARIMA model. Defaults to (0, 0, 0, 0) for no seasonality.
    trend : str
        Trend parameter ('c' for constant, 'n' for none)
    """
    
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 order=(1, 0), seasonal_order=(0, 0, 0, 0), trend='c', device='cpu',
                 p=None, d=None, q=None, **kwargs):
        # Handle Optuna-style separate p, d, q parameters
        # If p, d, q are provided separately, construct order tuple
        if p is not None and d is not None:
            if q is not None:
                order = (p, d, q)
            else:
                order = (p, d)

        super().__init__(num_features, num_timesteps_input, num_timesteps_output,
                         order, trend, device, **kwargs)
        self.seasonal_order = seasonal_order

    def _get_order(self):
        """Convert 2-tuple order to 3-tuple (p, d, q) format."""
        if len(self.order) == 2:
            return (self.order[0], self.order[1], 0)
        return self.order
    
    def _get_seasonal_order(self):
        """Get seasonal order, defaulting to no seasonality if not proper format."""
        if not self.seasonal_order or len(self.seasonal_order) != 4:
            return (0, 0, 0, 0)
        # Convert to Python ints to avoid torch tensor issues
        seasonal = tuple(int(x) if hasattr(x, 'item') else int(x) for x in self.seasonal_order)
        # If sum is 0, no seasonality
        if seasonal[0] + seasonal[1] + seasonal[2] == 0:
            return (0, 0, 0, 0)
        return seasonal

    def _fit_single_series(self, series):
        """Fit ARIMA model and return forecast."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            order_tuple = self._get_order()
            seasonal_order_tuple = self._get_seasonal_order()

            # Automatically adjust trend based on integration order
            # When d > 0 or D > 0, constant trend cannot be used
            trend = self.trend
            total_integration = order_tuple[1] + seasonal_order_tuple[1]
            if total_integration > 0 and trend == 'c':
                trend = 'n'  # Use no trend when integration is present

            try:
                model = ARIMA(series, order=order_tuple, seasonal_order=seasonal_order_tuple, trend=trend)
                results = model.fit(method_kwargs={'maxiter': 50, 'disp': False})
                forecast = results.forecast(steps=self.num_timesteps_output)
                result = np.asarray(forecast)
                if np.any(np.isnan(result)):
                    # Fallback: repeat last observed value
                    return np.full(self.num_timesteps_output, series[-1])
                return result
            except Exception:
                # Fallback: repeat last observed value (naive forecast)
                return np.full(self.num_timesteps_output, series[-1])

    def _forecast_single_series(self, series):
        """Forecast using ARIMA."""
        return self._fit_single_series(series)


class SeasonalNaiveModel(BaseStatsModel):
    """
    Seasonal Naive Forecasting Model.

    A simple baseline that forecasts by repeating the values from the same
    season in the previous cycle. For example, with weekly data and annual
    seasonality (52 weeks), the forecast for week t is the actual value
    from week t-52.

    This is a strong baseline for seasonal data and often outperforms more
    complex methods when seasonality is the dominant pattern.

    Parameters
    ----------
    num_features : int
        Number of features (only last feature/target is used)
    num_timesteps_input : int
        Number of input timesteps (lookback window)
    num_timesteps_output : int
        Number of output timesteps to predict (forecast horizon)
    season_length : int
        Length of the seasonal cycle (e.g., 52 for weekly data with annual seasonality,
        12 for monthly data with annual seasonality, 7 for daily data with weekly seasonality).
        Default: 52 (annual seasonality for weekly epidemic data).
    device : str
        Device for compatibility ('cpu' or 'cuda')

    Examples
    --------
    For weekly COVID-19 data with annual patterns:
    >>> model = SeasonalNaiveModel(num_features=4, num_timesteps_input=53,
    ...                            num_timesteps_output=4, season_length=52)
    >>> # Forecast for week t uses actual value from week t-52

    Notes
    -----
    - If lookback < season_length, falls back to naive (last value) forecast
    - If season_length is not provided, uses lookback as season length (assumes 1 cycle in window)
    - Very fast to compute (no model fitting required)
    - Provides a strong baseline for seasonal data
    """

    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 season_length=None, device='cpu', **kwargs):
        # If season_length not provided, infer from lookback (assume 1 full cycle)
        if season_length is None:
            season_length = num_timesteps_input

        # Store as order for compatibility with base class
        super().__init__(num_features, num_timesteps_input, num_timesteps_output,
                         order=(season_length,), trend='n', device=device, **kwargs)
        self.season_length = int(season_length)

    def _get_order(self):
        """Return season_length as single-element tuple."""
        return (self.season_length,)

    def _is_series_valid(self, series, order_tuple):
        """Check if series has enough data points for seasonal forecast.

        Need at least season_length observations to make a seasonal forecast.
        If not available, will fall back to naive forecast (last value).
        """
        # We can still make a forecast even with less data (using naive fallback)
        return len(series) > 0

    def _fit_single_series(self, series):
        """
        Generate seasonal naive forecast.

        For each forecast step h (h=1, 2, ..., H):
        - If lookback >= season_length: forecast[h] = series[-(season_length - h + 1)]
        - Else: forecast[h] = series[-1] (naive fallback)

        Parameters
        ----------
        series : np.ndarray
            Historical time series of shape (num_timesteps_input,)

        Returns
        -------
        np.ndarray
            Forecast of shape (num_timesteps_output,)
        """
        forecast = np.zeros(self.num_timesteps_output)
        series_length = len(series)

        for h in range(self.num_timesteps_output):
            # Calculate the index of the seasonal lag
            # For h=0 (first forecast step), use value from season_length steps back
            # For h=1, use value from (season_length - 1) steps back, etc.
            seasonal_lag_idx = series_length - self.season_length + h

            if seasonal_lag_idx >= 0 and seasonal_lag_idx < series_length:
                # We have the seasonal value available
                forecast[h] = series[seasonal_lag_idx]
            else:
                # Not enough history for seasonal forecast
                # Fall back to naive method (repeat last value)
                forecast[h] = series[-1]

        return forecast

    def _forecast_single_series(self, series):
        """Forecast using seasonal naive method."""
        return self._fit_single_series(series)


# =============================================================================
# Nowcasting-specific models
# =============================================================================

class _NowcastBase(BaseStatsModel):
    """Base for nowcasting models that work directly on the reporting triangle.

    Subclasses override _nowcast_batch to produce (N, W) predictions from
    the (N, L, D) feature matrix (D = delay columns, -1 = unobserved).
    """

    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 device='cpu', **kwargs):
        super().__init__(num_features, num_timesteps_input, num_timesteps_output,
                         order=(1,), trend='n', device=device, **kwargs)

    def _is_nowcast_data(self, data):
        """Detect nowcast triangle data (has -1 sentinels and >1 feature)."""
        return self.num_features > 1 and np.any(data == -1)

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _rightmost_valid(row):
        """Return (delay_index, value) of the most mature observation, or (-1, 0)."""
        valid = np.where((row != -1) & (row >= 0))[0]
        if len(valid) == 0:
            return -1, 0.0
        d = valid[-1]
        return d, float(row[d])

    @staticmethod
    def _enforce_monotone(arr):
        """Make array non-decreasing in-place and clip to [0.01, 1]."""
        for i in range(1, len(arr)):
            arr[i] = max(arr[i], arr[i - 1])
        np.clip(arr, 0.01, 1.0, out=arr)
        return arr

    def predict(self, feature, graph=None, states=None, dynamic_graph=None, **kwargs):
        X = self._prepare_data(feature)  # (N, L, D)
        if X.ndim == 3 and self._is_nowcast_data(X[0]):
            return self._nowcast_batch(X)
        # Fallback to parent (standard forecasting)
        return super().predict(feature, graph=graph, states=states,
                               dynamic_graph=dynamic_graph, **kwargs)

    def _nowcast_batch(self, X):
        raise NotImplementedError


class RKINowcastModel(_NowcastBase):
    """RKI-style delay-adjusted nowcasting (An der Heiden & Hamouda, 2020).

    During fit(): learns the completion CDF  F(d) = E[count_at_delay_d / final_count]
    from training data where final counts (targets) are known.

    During predict(): for each target day, divides the most-mature partial
    observation by F(d) to estimate the final count.

    Parameters
    ----------
    num_features, num_timesteps_input, num_timesteps_output : int
        Standard model dimensions (set automatically by the framework).
    """

    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 device='cpu', **kwargs):
        super().__init__(num_features, num_timesteps_input, num_timesteps_output,
                         device=device, **kwargs)
        self.completion_cdf = None

    def fit(self, train_input, train_target, **kwargs):
        """Estimate completion CDF from training (features, targets) pairs."""
        X = self._prepare_data(train_input)   # (N, L, D)
        Y = self._prepare_data(train_target)  # (N, W)
        if X.ndim != 3 or not self._is_nowcast_data(X[0]):
            return  # Not nowcast data — no-op
        N, L, D = X.shape
        W = Y.shape[1]

        # Collect F(d) = partial / final for every (sample, target-day, delay)
        ratio_lists = [[] for _ in range(D)]
        for i in range(N):
            for k in range(W):
                final = Y[i, k]
                if final < 1e-8:
                    continue
                row = X[i, L - W + k]
                for d in range(D):
                    v = row[d]
                    if v != -1 and v > 0:
                        ratio_lists[d].append(v / final)

        cdf = np.ones(D, dtype=np.float64)
        for d in range(D):
            if ratio_lists[d]:
                cdf[d] = np.median(ratio_lists[d])

        # Extrapolate unmeasured high-delay F(d) via monotone interpolation
        measured = np.array([d for d in range(D) if ratio_lists[d]])
        if len(measured) > 1:
            vals = np.array([cdf[d] for d in measured])
            interp = PchipInterpolator(measured, vals, extrapolate=True)
            for d in range(D):
                if not ratio_lists[d]:
                    cdf[d] = float(interp(d))

        self.completion_cdf = self._enforce_monotone(cdf)

    def _nowcast_batch(self, X):
        N, L, D = X.shape
        W = self.num_timesteps_output
        cdf = self.completion_cdf if self.completion_cdf is not None else np.ones(D)

        preds = np.zeros((N, W))
        for i in range(N):
            prev = 0.0
            for k in range(W):
                d, v = self._rightmost_valid(X[i, L - W + k])
                if d >= 0:
                    prev = v / cdf[d]
                preds[i, k] = prev  # Fallback: carry forward if no data

        return torch.tensor(preds, dtype=torch.float32)


class NobBSModel(_NowcastBase):
    """Simplified NobBS nowcasting (McGough et al., PLOS Comp Bio, 2020).

    Estimates the delay distribution *per sample* from the observed triangle,
    then corrects partial counts and applies random-walk smoothing on the
    log-incidence curve.  No training data required — purely generative.

    Parameters
    ----------
    num_features, num_timesteps_input, num_timesteps_output : int
        Standard model dimensions.
    smoothing : float
        Exponential-smoothing weight α ∈ (0, 1]. Lower → more smoothing
        (stronger random-walk prior).  Default 0.3.
    """

    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 smoothing=0.3, device='cpu', **kwargs):
        super().__init__(num_features, num_timesteps_input, num_timesteps_output,
                         device=device, **kwargs)
        self.smoothing = float(smoothing)

    def fit(self, **kwargs):
        """No-op — NobBS estimates everything per-sample during predict()."""
        pass

    def _nowcast_batch(self, X):
        N, L, D = X.shape
        W = self.num_timesteps_output
        preds = np.zeros((N, W))
        for i in range(N):
            preds[i] = self._nowcast_single(X[i], L, D, W)
        return torch.tensor(preds, dtype=torch.float32)

    def _nowcast_single(self, X, L, D, W):
        obs = (X != -1) & (X >= 0)

        # --- Step 1: estimate delay CDF from well-observed rows -------------
        cdf = self._estimate_cdf(X, obs, L, D)

        # --- Step 2: RKI-style raw correction for every row -----------------
        raw = np.zeros(L)
        for t in range(L):
            if obs[t].any():
                d = np.where(obs[t])[0][-1]
                raw[t] = X[t, d] / cdf[d]
            elif t > 0:
                raw[t] = raw[t - 1]

        # --- Step 3: random-walk smoothing on log-scale ---------------------
        log_raw = np.log(np.maximum(raw, 1e-8))
        fwd = np.empty(L)
        fwd[0] = log_raw[0]
        a = self.smoothing
        for t in range(1, L):
            fwd[t] = a * log_raw[t] + (1 - a) * fwd[t - 1]

        return np.exp(fwd[L - W:])

    @staticmethod
    def _estimate_cdf(X, obs, L, D):
        """Estimate within-sample delay CDF from rows with many observed delays."""
        row_obs = obs.sum(axis=1)
        # Pick rows with ≥70% of delays observed (at least 3)
        thresh = max(int(D * 0.7), 3)
        good = np.where(row_obs >= thresh)[0]
        if len(good) < 3:
            good = np.where(row_obs >= 3)[0]
        if len(good) == 0:
            return np.ones(D)

        # F(d) = median( X[t,d] / X[t,d_max] ) over good rows
        cdf = np.ones(D, dtype=np.float64)
        for d in range(D):
            ratios = []
            for t in good:
                d_max = np.where(obs[t])[0][-1]
                if d <= d_max and obs[t, d] and X[t, d_max] > 0:
                    ratios.append(X[t, d] / X[t, d_max])
            if ratios:
                cdf[d] = np.median(ratios)

        for i in range(1, D):
            cdf[i] = max(cdf[i], cdf[i - 1])
        return np.clip(cdf, 0.01, 1.0)

