"""
Compartmental epidemic models (SIR, SIS, SEIR) for the epilearn benchmark.

Same pattern as StatsModel / ARIMAModel:
- fit() is a no-op (Optuna tunes lookback only).
- predict() fits ODE parameters per-sample on the lookback window, then forecasts.
"""

import numpy as np
import torch
import warnings
from abc import ABC, abstractmethod
from scipy.integrate import odeint
from scipy.optimize import minimize


class BaseCompartmentalModel(ABC):
    """
    Base class for compartmental ODE models.
    Follows the same per-sample-fitting interface as BaseStatsModel / ARIMAModel.
    """

    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 device='cpu', **kwargs):
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.device = device
        self.model = None   # compat with benchmark framework
        self.kwargs = kwargs

    # ── abstract ODE interface ───────────────────────────────────────

    @abstractmethod
    def _ode(self, y, t, params):
        """dy/dt = f(y, t; params).  Used by scipy.integrate.odeint."""
        ...

    @abstractmethod
    def _get_initial_params(self):
        """Return initial parameter guess (numpy array) from self.kwargs."""
        ...

    @abstractmethod
    def _get_bounds(self):
        """Return list of (lo, hi) bounds for each ODE parameter."""
        ...

    @abstractmethod
    def _initial_state(self, I0):
        """Return initial compartment vector given infected fraction I0."""
        ...

    @abstractmethod
    def _infected_idx(self):
        """Index of the Infected compartment in the state vector."""
        ...

    # ── fit (no-op, same as stats models) ────────────────────────────

    def fit(self, train_input, train_target, train_states=None,
            val_input=None, val_target=None,
            train_graph=None, train_dynamic_graph=None,
            val_graph=None, val_dynamic_graph=None,
            val_states=None, epochs=1000, batch_size=10,
            verbose=False, patience=100,
            lr=None, weight_decay=None, loss=None,
            initialize=True, **kwargs):
        """No-op. ODE parameters are fitted per-sample during predict()."""
        return None

    # ── predict (per-sample fitting, same pattern as ARIMA) ──────────

    def predict(self, feature, graph=None, states=None,
                dynamic_graph=None, **kwargs):
        """
        Per-sample ODE forecast.

        For each sample in the batch:
        1. Extract the target series (last feature column).
        2. Normalise to [0, 1] and fit ODE parameters via L-BFGS-B.
        3. Reconstruct compartment state at end of lookback.
        4. Simulate forward for `horizon` steps.
        5. Map forecast back to original scale.
        """
        X = (feature.detach().cpu().numpy()
             if torch.is_tensor(feature) else np.asarray(feature))

        # Handle 4D spatiotemporal shape: (B, T, N, F) → (B*N, T, F)
        if X.ndim == 4:
            b, t, n, f = X.shape
            X = X.transpose(0, 2, 1, 3).reshape(b * n, t, f)

        infections = X[:, :, -1]  # (batch, time) — target = last column
        horizon = self.num_timesteps_output
        forecasts = []

        for series in infections:
            forecast = self._forecast_single(series, horizon)
            forecasts.append(forecast)

        return torch.tensor(np.array(forecasts), dtype=torch.float32)

    def _forecast_single(self, series, horizon):
        """
        Fit ODE parameters to one lookback window, then forecast.

        1. Normalise lookback to [0, 1].
        2. Optimise ODE params to minimise MSE on the normalised lookback.
        3. Simulate the full lookback to get the final compartment state.
        4. Continue the ODE forward for `horizon` steps.
        5. Map back to original scale.
        """
        last_val = float(series[-1])
        lo, hi = float(series.min()), float(series.max())
        rng = hi - lo
        if rng < 1e-12:
            return np.full(horizon, last_val)

        normed = (series - lo) / rng                       # [0, 1]
        T = len(normed)
        I0 = float(np.clip(normed[0], 1e-4, 1.0 - 1e-4))
        idx = self._infected_idx()

        # ── per-sample parameter fitting ──
        def objective(params):
            try:
                y0 = self._initial_state(I0)
                t = np.arange(T, dtype=np.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sol = odeint(self._ode, y0, t, args=(params,))
                pred = np.clip(sol[:, idx], 0.0, 1.0)
                return float(np.mean((pred - normed) ** 2))
            except Exception:
                return 1e10

        p0 = self._get_initial_params()
        bounds = self._get_bounds()
        try:
            result = minimize(
                objective, p0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 50, 'disp': False})
            best = result.x
        except Exception:
            best = p0

        # ── reconstruct full state at end of lookback ──
        try:
            y0 = self._initial_state(I0)
            t_full = np.arange(T, dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol_full = odeint(self._ode, y0, t_full, args=(best,))
            final_state = sol_full[-1]

            # ── forecast from final state ──
            t_fc = np.arange(horizon + 1, dtype=np.float64)
            sol_fc = odeint(self._ode, final_state, t_fc, args=(best,))
            fc_norm = np.clip(sol_fc[1:, idx], 0.0, 1.0)
            forecast = fc_norm * rng + lo
        except Exception:
            forecast = np.full(horizon, last_val)

        return forecast

    def to(self, device):
        """Compatibility with framework device placement."""
        self.device = device
        return self


# ═════════════════════════════════════════════════════════════════════
# Concrete models
# ═══════════════════════════════════════════════════════════════════════

class SIRModel(BaseCompartmentalModel):
    """Susceptible → Infected → Recovered."""

    def _ode(self, y, t, params):
        S, I, R = y
        beta, gamma = params
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    def _get_initial_params(self):
        b = float(self.kwargs.get('infection_rate', 0.3))
        g = float(self.kwargs.get('recovery_rate', 0.1))
        return np.array([b, g])

    def _get_bounds(self):
        return [(0.01, 10.0), (0.01, 10.0)]

    def _initial_state(self, I0):
        return [1.0 - I0, I0, 0.0]

    def _infected_idx(self):
        return 1


class SISModel(BaseCompartmentalModel):
    """Susceptible → Infected → Susceptible (no permanent immunity)."""

    def _ode(self, y, t, params):
        S, I = y
        beta, gamma = params
        dS = -beta * S * I + gamma * I
        dI = beta * S * I - gamma * I
        return [dS, dI]

    def _get_initial_params(self):
        b = float(self.kwargs.get('infection_rate', 0.3))
        g = float(self.kwargs.get('recovery_rate', 0.1))
        return np.array([b, g])

    def _get_bounds(self):
        return [(0.01, 10.0), (0.01, 10.0)]

    def _initial_state(self, I0):
        return [1.0 - I0, I0]

    def _infected_idx(self):
        return 1


class SEIRModel(BaseCompartmentalModel):
    """Susceptible → Exposed → Infected → Recovered."""

    def _ode(self, y, t, params):
        S, E, I, R = y
        beta, sigma, gamma = params
        dS = -beta * S * I
        dE = beta * S * I - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    def _get_initial_params(self):
        b = float(self.kwargs.get('infection_rate', 0.5))
        s = float(self.kwargs.get('latency', 0.2))
        g = float(self.kwargs.get('recovery_rate', 0.1))
        return np.array([b, s, g])

    def _get_bounds(self):
        return [(0.01, 10.0), (0.01, 10.0), (0.01, 10.0)]

    def _initial_state(self, I0):
        E0 = I0 * 0.3
        return [1.0 - E0 - I0, E0, I0, 0.0]

    def _infected_idx(self):
        return 2
