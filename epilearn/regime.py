"""Regime detection and epidemiological regime axes.

Beyond the quantile-threshold labels (`REGIME_LABELS`) ported from
analysis/WHEN/comprehensive_analysis.py, this module exposes two
epidemiologically grounded axes for re-stratifying benchmark errors:

  * ``Rt(series, ...)`` — local effective reproduction number, from a
    sliding log-linear fit on incidence (Wallinga–Lipsitch style); used
    for forecasting where the relevant difficulty driver is whether the
    underlying transmission parameter is locally identifiable.
  * ``reporting_completeness(triangle, delay)`` — c(t, d) =
    reported_at_delay_d(t) / final(t); used for nowcasting where the
    relevant difficulty driver is the noise-to-signal ratio of the
    partially observed signal at decision time.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import pearsonr


# Default serial interval (days) for COVID/flu-like; users can override.
DEFAULT_SERIAL_INTERVAL = 4.7


def Rt(
    series: np.ndarray,
    method: str = 'loglinear',
    window: int = 7,
    serial_interval: float = DEFAULT_SERIAL_INTERVAL,
) -> np.ndarray:
    """Local effective reproduction number from incidence.

    Computes R_t from a sliding log-linear fit on `series` (Wallinga &
    Lipsitch, 2007 approximation):

        slope_t = d log(cases) / d t  estimated over a window of `window`
                  observations centered at t,
        R_t    = exp(slope_t · serial_interval).

    This is a deterministic, scipy-only estimator — coarser than Cori's
    Bayesian R_t but free of dependencies and stable on short windows.
    Returns NaN where the window contains <3 positive observations.

    Parameters
    ----------
    series : (T,) array-like, non-negative incidence
    method : 'loglinear' (default). Reserved for future Cori-style estimator.
    window : odd integer ≥ 3, smoothing window (in time steps)
    serial_interval : float, mean serial interval in *the same time unit*
        as `series` (e.g., days for daily data, weeks for weekly data).
        Default is COVID-like (4.7 days). For weekly ILI data, pass
        ``serial_interval=4.7/7`` so R_t is unitless across resolutions.

    Returns
    -------
    rt : (T,) array of R_t values, NaN where the local fit is undefined.
    """
    if method != 'loglinear':
        raise NotImplementedError(f"Rt method='{method}' not implemented")
    series = np.asarray(series, dtype=float).ravel()
    T = len(series)
    if T < 3 or window < 3:
        return np.full(T, np.nan)
    half = window // 2
    rt = np.full(T, np.nan)
    # log(cases+1) to absorb zero-count weeks; dominant signal is positive.
    log_cases = np.log(np.maximum(series, 0.0) + 1.0)
    x = np.arange(window, dtype=float) - half
    for t in range(half, T - half):
        y = log_cases[t - half:t + half + 1]
        # Need ≥3 informative (positive) points for a stable slope
        if np.sum(np.maximum(series[t - half:t + half + 1], 0.0) > 0) < 3:
            continue
        # Closed-form OLS slope; scipy.linregress avoided to skip stat overhead
        slope = float(np.polyfit(x, y, 1)[0])
        rt[t] = np.exp(slope * serial_interval)
    return rt


def reporting_completeness(triangle: np.ndarray, delay: int) -> np.ndarray:
    """c(t, d) = reported_at_delay_d(t) / final(t) for a reporting triangle.

    Parameters
    ----------
    triangle : (T, D) array. Row t = cumulative reports of day t observed
        at delays 0..D-1. Column 0 is what was reported on day t (delay
        0); column D-1 is treated as the "final" revised count.
    delay : the delay index `d` (0-based) at which to evaluate completeness.

    Returns
    -------
    c : (T,) array in [0, 1]. NaN where final(t) ≤ 0 or `t + delay` falls
        outside the triangle.
    """
    triangle = np.asarray(triangle, dtype=float)
    if triangle.ndim != 2:
        raise ValueError("triangle must be (T, D)")
    T, D = triangle.shape
    if not 0 <= delay < D:
        raise ValueError(f"delay {delay} not in [0, {D-1}]")
    final = triangle[:, -1]
    c = triangle[:, delay] / np.where(final > 0, final, np.nan)
    return c


def Rt_band(
    rt: float,
    growing_threshold: float = 1.2,
    declining_threshold: float = 0.8,
) -> str:
    """Coarse 3-band labelling of R_t: growing / turning / declining.

    Defaults bracket the canonical R_t ≈ 1 turning-point regime where the
    SIR likelihood is locally flat and transmission is unidentifiable
    from a short observation window.
    """
    if not np.isfinite(rt):
        return 'Unknown'
    if rt >= growing_threshold:
        return 'Growing (R_t ≥ 1.2)'
    if rt <= declining_threshold:
        return 'Declining (R_t ≤ 0.8)'
    return 'Turning (0.8 < R_t < 1.2)'


REGIME_LABELS = [
    'Explosive Growth',
    'Moderate Growth',
    'Peak Plateau',
    'Turbulent',
    'Rapid Decline',
    'Gradual Decline',
    'Trough/Endemic',
]


def compute_regime_features(targets: np.ndarray) -> dict:
    """Compute regime features from a target array.

    Parameters
    ----------
    targets : (n_samples,) or (n_samples, horizon) — case counts

    Returns
    -------
    dict with scalar descriptors: level, volatility, growth_rate, etc.
    """
    if targets.ndim == 2:
        series = targets[:, 0].astype(float)
    else:
        series = targets.astype(float)
    series = series[np.isfinite(series)]
    if len(series) < 4:
        return {}

    level = float(np.mean(series))
    abs_mean = max(abs(level), 1e-8)
    level_norm = level

    volatility = float(np.std(series) / max(np.mean(np.abs(series)), 1e-8))

    skewness = float(sp_stats.skew(series))
    kurtosis = float(sp_stats.kurtosis(series))

    mid = len(series) // 2
    early_mean = np.mean(series[:mid])
    late_mean = np.mean(series[mid:])
    growth_rate = float((late_mean - early_mean) / abs_mean)

    q = max(len(series) // 4, 1)
    q1 = np.mean(series[:q])
    q2 = np.mean(series[q:2 * q])
    q3 = np.mean(series[2 * q:3 * q])
    q4 = np.mean(series[-q:])
    accel = float(((q3 + q4) - (q1 + q2)) / 2 / abs_mean - growth_rate)

    try:
        ac1 = float(pearsonr(series[:-1], series[1:])[0])
    except Exception:
        ac1 = np.nan

    q90 = np.percentile(series, 90)
    near_peak_frac = float(np.mean(series >= q90))

    return {
        'level': level,
        'level_norm': level_norm,
        'volatility': volatility,
        'growth_rate': growth_rate,
        'acceleration': accel,
        'ac1': ac1,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'near_peak_frac': near_peak_frac,
    }


def classify_regime(
    growth_rate: float,
    volatility: float,
    level: float,
    level_q15: float,
    level_q85: float,
    vol_q75: float,
) -> str:
    """Assign one of seven unified epidemic regime labels.

    Priority (first matching rule wins):
      1. Trough/Endemic  — low level AND near-flat
      2. Turbulent       — high volatility, no dominant trend
      3. Explosive Growth
      4. Moderate Growth
      5. Peak Plateau    — high level AND near-flat
      6. Rapid Decline
      7. Gradual Decline
    """
    if level < level_q15 and abs(growth_rate) < 0.20:
        return 'Trough/Endemic'
    if vol_q75 is not None and volatility > vol_q75 and abs(growth_rate) < 0.50:
        return 'Turbulent'
    if growth_rate > 0.50:
        return 'Explosive Growth'
    if growth_rate > 0.15:
        return 'Moderate Growth'
    if level > level_q85 and abs(growth_rate) <= 0.15:
        return 'Peak Plateau'
    if growth_rate < -0.50:
        return 'Rapid Decline'
    if growth_rate < -0.15:
        return 'Gradual Decline'
    if level > level_q85:
        return 'Peak Plateau'
    return 'Gradual Decline'


def assign_regimes(regime_df: pd.DataFrame) -> pd.DataFrame:
    """Compute quantile thresholds from all folds and assign labels."""
    if len(regime_df) == 0:
        return regime_df
    level_q15 = regime_df['level'].quantile(0.15)
    level_q85 = regime_df['level'].quantile(0.85)
    vol_q75 = regime_df['volatility'].quantile(0.75)

    regime_df = regime_df.copy()
    regime_df['regime'] = regime_df.apply(
        lambda r: classify_regime(
            r.get('growth_rate', 0), r.get('volatility', 0),
            r.get('level', 0), level_q15, level_q85, vol_q75,
        ), axis=1,
    )
    return regime_df


def load_regime_mapping(task: str, analysis_dir: str = 'analysis/WHEN/results') -> dict:
    """Load fold-regime mapping from WHEN analysis CSVs.

    Returns
    -------
    dict  {fold_index: regime_label}
    """
    import os
    csv_path = os.path.join(analysis_dir, f'{task}_regime_characterization.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Regime characterization not found: {csv_path}")
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        fold_idx = int(row['fold']) - 1  # CSV uses 1-based indexing
        mapping[fold_idx] = row['regime']
    return mapping
