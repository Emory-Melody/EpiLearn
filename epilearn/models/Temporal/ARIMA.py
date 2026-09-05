"""Backwards-compatible import path.

``VARMAXModel`` and ``ARIMAModel`` used to live in
``epilearn.models.Temporal.ARIMA``. They moved to
``epilearn.models.Temporal.StatsModel`` in 0.1.0, which also holds the
seasonal-naive and nowcasting baselines.

Deprecated -- import from ``epilearn.models.Temporal.StatsModel`` instead.
"""

from .StatsModel import VARMAXModel, ARIMAModel  # noqa: F401

__all__ = ['VARMAXModel', 'ARIMAModel']
