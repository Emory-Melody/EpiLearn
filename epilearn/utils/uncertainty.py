"""
Conformal prediction and uncertainty quantification for epidemic forecasting.

Strategies
----------
1. **Static conformal** — ``pred ± q`` where ``q`` is a fixed quantile
   of calibration residuals.  Constant width, no per-sample adaptation.

2. **ACI (Adaptive Conformal Inference)** — Gibbs & Candès (2021).  Adapts
   the miscoverage rate ``α_t`` sequentially so that long-run coverage is
   controlled.  Width varies *over time* but is the same for every sample
   at a given step.

3. **Locally-weighted conformal** — a.k.a. Normalised Conformal Prediction
   (Lei et al., 2018; Papadopoulos et al., 2008).  Scales the conformal
   interval by a per-sample *difficulty estimate* ``s_i`` so that
   ``interval_i = pred_i ± q_norm × s_i``.  For epidemic data the natural
   choice is ``s_i = max(|pred_i|, floor)`` because errors scale with
   signal magnitude (heteroscedasticity).  This makes the interval width
   proportional to prediction magnitude, dramatically improving the
   σ-vs-error correlation.

4. **Locally-weighted ACI** — combines local weighting with the ACI
   α-adaptation, giving both per-sample heteroscedastic widths **and**
   long-run coverage control.

All functions operate on NumPy arrays in *original (denormalized) scale*.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

__all__ = [
    # core strategies
    "static_conformal",
    "compute_aci",
    "locally_weighted_conformal",
    "locally_weighted_aci",
    # evaluation helpers
    "winkler_score",
    "compute_uncertainty_metrics",
    "evaluate_aci_from_saved",
    # difficulty estimators
    "difficulty_from_predictions",
    "difficulty_reference_stats",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Difficulty estimators
# ═══════════════════════════════════════════════════════════════════════════════

def difficulty_from_predictions(
    predictions: np.ndarray,
    *,
    floor: float | None = None,
    floor_quantile: float = 0.1,
    ref_floor: float | None = None,
    ref_median: float | None = None,
) -> np.ndarray:
    """Per-sample difficulty score based on prediction magnitude.

    For epidemic time-series errors typically scale with the magnitude of
    the predicted count — high-incidence regions/periods have larger
    absolute errors.  This function returns ``max(mean|pred_i|, floor)``
    per sample, normalised so the median difficulty is 1.0.

    Parameters
    ----------
    predictions : ndarray, shape ``(n, ...)``
        Denormalized point predictions.  The first axis is the sample axis;
        remaining axes (e.g. horizons, nodes) are averaged.
    floor : float, optional
        Minimum difficulty value *before* normalisation.  If ``None``
        (default) it is set to ``quantile(mean|pred|, floor_quantile)``.
    floor_quantile : float
        Quantile of ``mean|pred|`` used to set the floor when *floor* is
        ``None``.  Prevents near-zero predictions from creating infinite
        normalised residuals.
    ref_floor : float, optional
        Pre-computed floor from **calibration** data.  When provided,
        overrides the floor computed from *predictions*, avoiding
        test-set leakage.
    ref_median : float, optional
        Pre-computed median difficulty from **calibration** data.  When
        provided, overrides the median computed from *predictions*,
        avoiding test-set leakage.

    Returns
    -------
    ndarray, shape ``(n,)``
        Per-sample difficulty scores, median ≈ 1.0 (w.r.t. reference).
    """
    pred = np.asarray(predictions, dtype=np.float64)
    n = pred.shape[0]
    # mean absolute prediction magnitude, averaged over horizons / nodes
    reduce = tuple(range(1, pred.ndim)) if pred.ndim > 1 else ()
    mag = np.mean(np.abs(pred), axis=reduce) if reduce else np.abs(pred).ravel()

    # Floor: prefer calibration-derived value to avoid test-batch leakage
    if ref_floor is not None:
        floor = ref_floor
    elif floor is None:
        floor = float(np.quantile(mag, floor_quantile))
    floor = max(floor, 1e-8)  # safety

    s = np.maximum(mag, floor)

    # Median normalisation: prefer calibration-derived value
    if ref_median is not None:
        median_s = ref_median
    else:
        median_s = float(np.median(s))
    if median_s > 1e-12:
        s = s / median_s          # normalise so median difficulty = 1
    return s


def difficulty_reference_stats(
    reference_data: np.ndarray,
    *,
    floor_quantile: float = 0.1,
    is_residual: bool = False,
) -> tuple[float, float]:
    """Compute difficulty normalisation constants from calibration data.

    These constants (``ref_floor``, ``ref_median``) should be passed to
    :func:`difficulty_from_predictions` when scoring *test* samples so
    that the floor and median are derived from the **calibration set**
    rather than the test batch — preventing information leakage.

    Parameters
    ----------
    reference_data : ndarray
        Either calibration predictions (shape ``(n, ...)``) or calibration
        residuals (1-D ``|pred - target|``).  Set *is_residual* accordingly.
    floor_quantile : float
        Quantile used for the floor.
    is_residual : bool
        If ``True``, *reference_data* are absolute residuals (already
        positive, 1-D).  If ``False``, per-sample mean absolute value is
        computed first.

    Returns
    -------
    (ref_floor, ref_median) : tuple[float, float]
        Values to pass to ``difficulty_from_predictions(…,
        ref_floor=ref_floor, ref_median=ref_median)``.
    """
    data = np.asarray(reference_data, dtype=np.float64)
    if is_residual:
        mag = np.abs(data).ravel()
    else:
        reduce = tuple(range(1, data.ndim)) if data.ndim > 1 else ()
        mag = (np.mean(np.abs(data), axis=reduce) if reduce
               else np.abs(data).ravel())

    ref_floor = float(np.quantile(mag, floor_quantile))
    ref_floor = max(ref_floor, 1e-8)
    s = np.maximum(mag, ref_floor)
    ref_median = float(np.median(s))
    return ref_floor, ref_median


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: Static conformal
# ═══════════════════════════════════════════════════════════════════════════════

def static_conformal(
    val_residuals: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_alpha: float = 0.1,
) -> dict:
    """Plain split-conformal prediction intervals.

    Parameters
    ----------
    val_residuals : 1-D array
        Absolute residuals from the calibration set.
    predictions, targets : arrays, shape ``(n_test, ...)``
        Test-set predictions and ground truth (denormalized).
    target_alpha : float
        Target miscoverage rate.

    Returns
    -------
    dict  — same schema as :func:`compute_aci`.
    """
    sorted_res = np.sort(np.asarray(val_residuals, dtype=np.float64).ravel())
    n_cal = len(sorted_res)
    if n_cal == 0:
        return None

    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - target_alpha)) / n_cal)
    q_idx = int(np.clip(np.ceil(q_level * n_cal) - 1, 0, n_cal - 1))
    q = sorted_res[q_idx]

    pred = np.asarray(predictions, dtype=np.float64)
    targ = np.asarray(targets, dtype=np.float64)
    lower = pred - q
    upper = pred + q

    n_test = pred.shape[0]
    cov_ps = np.all((targ >= lower) & (targ <= upper),
                     axis=tuple(range(1, targ.ndim)) if targ.ndim > 1 else ()).astype(float)

    return {
        "lower": lower,
        "upper": upper,
        "coverage": float(np.mean(cov_ps)),
        "avg_width": float(np.mean(upper - lower)),
        "winkler_score": winkler_score(lower, upper, targ, target_alpha),
        "alpha_trace": np.full(n_test, target_alpha),
        "quantile_trace": np.full(n_test, q),
        "coverage_per_sample": cov_ps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: ACI  (Gibbs & Candès 2021)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aci(
    val_residuals: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_alpha: float = 0.1,
    gamma: float = 0.005,
) -> dict | None:
    """Adaptive Conformal Inference (Gibbs & Candès, 2021).

    Processes test samples *sequentially*, adapting the miscoverage rate
    ``α_t`` based on observed coverage at each step.

    Parameters
    ----------
    val_residuals : 1-D array, shape ``(n_cal,)``
        Absolute residuals from the calibration set.
    predictions : array, shape ``(n_test,)`` or ``(n_test, horizon)``
        Point predictions for the test set (denormalized).
    targets : array, same shape as *predictions*
        Ground truth values.
    target_alpha : float
        Target miscoverage rate (0.1 → 90 % coverage).
    gamma : float
        Learning rate for α adaptation.

    Returns
    -------
    dict
        lower, upper, coverage, avg_width, winkler_score,
        alpha_trace, quantile_trace, coverage_per_sample.
    """
    sorted_res = np.sort(np.asarray(val_residuals, dtype=np.float64).ravel())
    n_cal = len(sorted_res)
    if n_cal == 0:
        return None

    pred = np.asarray(predictions, dtype=np.float64)
    targ = np.asarray(targets, dtype=np.float64)
    n_test = pred.shape[0]
    flat_pred = pred.reshape(n_test, -1)
    flat_targ = targ.reshape(n_test, -1)

    lower = np.empty_like(flat_pred)
    upper = np.empty_like(flat_pred)
    alpha_trace = np.empty(n_test)
    quantile_trace = np.empty(n_test)
    coverage_per_sample = np.empty(n_test)

    alpha_t = target_alpha
    for t in range(n_test):
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha_t)) / n_cal)
        q_idx = int(np.clip(np.ceil(q_level * n_cal) - 1, 0, n_cal - 1))
        q_t = sorted_res[q_idx]

        lower[t] = flat_pred[t] - q_t
        upper[t] = flat_pred[t] + q_t

        covered = bool(np.all((flat_targ[t] >= lower[t]) & (flat_targ[t] <= upper[t])))
        err_t = 0.0 if covered else 1.0
        coverage_per_sample[t] = 1.0 - err_t

        alpha_t = np.clip(alpha_t + gamma * (target_alpha - err_t), 0.0, 1.0)
        alpha_trace[t] = alpha_t
        quantile_trace[t] = q_t

    lower = lower.reshape(pred.shape)
    upper = upper.reshape(pred.shape)

    return {
        "lower": lower,
        "upper": upper,
        "coverage": float(np.mean(coverage_per_sample)),
        "avg_width": float(np.mean(upper - lower)),
        "winkler_score": winkler_score(lower, upper, targ, target_alpha),
        "alpha_trace": alpha_trace,
        "quantile_trace": quantile_trace,
        "coverage_per_sample": coverage_per_sample,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Locally-Weighted Conformal Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def locally_weighted_conformal(
    val_residuals: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_alpha: float = 0.1,
    *,
    val_predictions: np.ndarray | None = None,
    difficulty_fn=None,
    floor_quantile: float = 0.1,
) -> dict | None:
    """Normalised / locally-weighted split-conformal prediction.

    Produces **per-sample heteroscedastic intervals** whose width scales
    with prediction magnitude — matching the natural heteroscedasticity
    of epidemic time-series.

    When *val_predictions* are available the full Lei et al. (2018) /
    Papadopoulos et al. (2008) procedure is used:

    1. Compute difficulty ``s_cal`` for calibration samples.
    2. Normalise residuals: ``r_norm = r_i / s_cal_i``.
    3. ``q_norm`` = ``(1-α)``-quantile of ``r_norm``.
    4. ``interval_i = pred_i ± q_norm × s_test_i``.

    When *val_predictions* are **not** available (the common case for
    post-hoc analysis of saved benchmarks), a simplified procedure is
    used that is equivalent to assuming calibration difficulty ≈ 1:

    1. ``q`` = ``(1-α)``-quantile of raw ``val_residuals``.
    2. ``interval_i = pred_i ± q × s_test_i``.

    Parameters
    ----------
    val_residuals : 1-D array
        Absolute residuals from the calibration set (original scale).
    predictions, targets : arrays, shape ``(n_test, ...)``
        Test-set data (denormalized).
    target_alpha : float
        Target miscoverage rate.
    val_predictions : array, optional
        Calibration-set predictions (denormalized).  When provided the
        full normalised procedure is used.
    difficulty_fn : callable, optional
        ``f(predictions) → difficulty (n,)`` array.  Defaults to
        :func:`difficulty_from_predictions`.
    floor_quantile : float
        Passed to the default difficulty function.

    Returns
    -------
    dict  — same schema as :func:`compute_aci`, plus
        ``difficulty_test`` — per-sample difficulty score.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    targ = np.asarray(targets, dtype=np.float64)
    val_res = np.asarray(val_residuals, dtype=np.float64).ravel()
    n_test = pred.shape[0]
    n_cal = len(val_res)
    if n_cal == 0:
        return None

    # ── Compute reference stats from *calibration* data (no test leakage)
    if val_predictions is not None:
        ref_floor, ref_median = difficulty_reference_stats(
            val_predictions, floor_quantile=floor_quantile)
    else:
        # Use calibration residuals as proxy for the data scale
        ref_floor, ref_median = difficulty_reference_stats(
            val_res, floor_quantile=floor_quantile, is_residual=True)

    if difficulty_fn is None:
        difficulty_fn = lambda p: difficulty_from_predictions(
            p, floor_quantile=floor_quantile,
            ref_floor=ref_floor, ref_median=ref_median,
        )

    # ── Calibration quantile ──────────────────────────────────────────────
    if val_predictions is not None:
        # Full normalised conformal: normalise cal residuals by difficulty
        val_pred = np.asarray(val_predictions, dtype=np.float64)
        s_cal = _per_element_difficulty(val_pred, difficulty_fn)
        if len(s_cal) != len(val_res):
            s_cal = np.maximum(val_res, np.quantile(val_res, floor_quantile))
            med = np.median(s_cal)
            s_cal = s_cal / med if med > 1e-12 else s_cal
        r_norm = val_res / np.maximum(s_cal, 1e-12)
    else:
        # Simplified: use raw residuals (= assuming cal difficulty ≈ 1).
        # The raw quantile is already in the correct scale because
        # difficulty_from_predictions normalises so median(s) = 1.
        r_norm = val_res

    sorted_r = np.sort(r_norm)
    n = len(sorted_r)
    q_level = min(1.0, np.ceil((n + 1) * (1 - target_alpha)) / n)
    q_idx = int(np.clip(np.ceil(q_level * n) - 1, 0, n - 1))
    q_norm = sorted_r[q_idx]

    # ── Test: per-sample difficulty → heteroscedastic intervals ───────────
    s_test = difficulty_fn(pred)                     # (n_test,)
    s_broad = s_test.reshape((n_test,) + (1,) * (pred.ndim - 1))

    lower = pred - q_norm * s_broad
    upper = pred + q_norm * s_broad

    cov_ps = np.all(
        (targ >= lower) & (targ <= upper),
        axis=tuple(range(1, targ.ndim)) if targ.ndim > 1 else (),
    ).astype(float)

    return {
        "lower": lower,
        "upper": upper,
        "coverage": float(np.mean(cov_ps)),
        "avg_width": float(np.mean(upper - lower)),
        "winkler_score": winkler_score(lower, upper, targ, target_alpha),
        "alpha_trace": np.full(n_test, target_alpha),
        "quantile_trace": np.full(n_test, q_norm),
        "coverage_per_sample": cov_ps,
        "difficulty_test": s_test,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 4: Locally-Weighted ACI
# ═══════════════════════════════════════════════════════════════════════════════

def locally_weighted_aci(
    val_residuals: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_alpha: float = 0.1,
    gamma: float = 0.005,
    *,
    val_predictions: np.ndarray | None = None,
    difficulty_fn=None,
    floor_quantile: float = 0.1,
) -> dict | None:
    """Locally-weighted Adaptive Conformal Inference.

    Combines the per-sample difficulty scaling of
    :func:`locally_weighted_conformal` with the sequential α-adaptation of
    :func:`compute_aci`.  This gives:

    * **Per-sample** heteroscedastic widths (via difficulty scores), and
    * **Long-run** coverage control (via ACI α-adaptation).

    Parameters
    ----------
    val_residuals : 1-D array
        Absolute residuals from calibration (original scale, flattened).
    predictions, targets : arrays, shape ``(n_test, ...)``
        Test set.
    target_alpha, gamma : float
        ACI parameters.
    val_predictions : array, optional
        Calibration-set predictions.  See :func:`locally_weighted_conformal`.
    difficulty_fn, floor_quantile :
        See :func:`locally_weighted_conformal`.

    Returns
    -------
    dict  — same schema as :func:`compute_aci`, plus ``difficulty_test``.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    targ = np.asarray(targets, dtype=np.float64)
    val_res = np.asarray(val_residuals, dtype=np.float64).ravel()
    n_test = pred.shape[0]
    n_cal = len(val_res)
    if n_cal == 0:
        return None

    # ── Compute reference stats from *calibration* data (no test leakage)
    if val_predictions is not None:
        ref_floor, ref_median = difficulty_reference_stats(
            val_predictions, floor_quantile=floor_quantile)
    else:
        ref_floor, ref_median = difficulty_reference_stats(
            val_res, floor_quantile=floor_quantile, is_residual=True)

    if difficulty_fn is None:
        difficulty_fn = lambda p: difficulty_from_predictions(
            p, floor_quantile=floor_quantile,
            ref_floor=ref_floor, ref_median=ref_median,
        )

    # ── Normalised calibration residuals ──────────────────────────────────
    if val_predictions is not None:
        val_pred = np.asarray(val_predictions, dtype=np.float64)
        s_cal = _per_element_difficulty(val_pred, difficulty_fn)
        if len(s_cal) != len(val_res):
            s_cal = np.maximum(val_res, np.quantile(val_res, floor_quantile))
            med = np.median(s_cal)
            s_cal = s_cal / med if med > 1e-12 else s_cal
        r_norm = val_res / np.maximum(s_cal, 1e-12)
    else:
        r_norm = val_res

    sorted_r = np.sort(r_norm)
    n = len(sorted_r)

    # ── Test difficulty scores ────────────────────────────────────────────
    s_test = difficulty_fn(pred)                     # (n_test,)

    flat_pred = pred.reshape(n_test, -1)
    flat_targ = targ.reshape(n_test, -1)
    H = flat_pred.shape[1]               # horizons (or horizons * nodes)

    lower = np.empty_like(flat_pred)
    upper = np.empty_like(flat_pred)
    alpha_trace = np.empty(n_test)
    quantile_trace = np.empty(n_test)
    coverage_per_sample = np.empty(n_test)

    alpha_t = target_alpha
    for t in range(n_test):
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha_t)) / n)
        q_idx = int(np.clip(np.ceil(q_level * n) - 1, 0, n - 1))
        q_norm_t = sorted_r[q_idx]

        # Per-sample scaling
        s_t = s_test[t]
        half_width = q_norm_t * s_t
        lower[t] = flat_pred[t] - half_width
        upper[t] = flat_pred[t] + half_width

        covered = bool(np.all((flat_targ[t] >= lower[t]) & (flat_targ[t] <= upper[t])))
        err_t = 0.0 if covered else 1.0
        coverage_per_sample[t] = 1.0 - err_t

        alpha_t = np.clip(alpha_t + gamma * (target_alpha - err_t), 0.0, 1.0)
        alpha_trace[t] = alpha_t
        quantile_trace[t] = q_norm_t * s_t   # actual half-width for this sample

    lower = lower.reshape(pred.shape)
    upper = upper.reshape(pred.shape)

    return {
        "lower": lower,
        "upper": upper,
        "coverage": float(np.mean(coverage_per_sample)),
        "avg_width": float(np.mean(upper - lower)),
        "winkler_score": winkler_score(lower, upper, targ, target_alpha),
        "alpha_trace": alpha_trace,
        "quantile_trace": quantile_trace,
        "coverage_per_sample": coverage_per_sample,
        "difficulty_test": s_test,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def winkler_score(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> float:
    """Winkler interval score (lower is better).

    ``score = width + (2/α) × (penalty_below + penalty_above)``
    """
    width = upper - lower
    below = np.maximum(lower - targets, 0.0)
    above = np.maximum(targets - upper, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def compute_uncertainty_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_alpha: float = 0.1,
) -> dict:
    """Evaluate quality of prediction intervals.

    Returns
    -------
    dict
        coverage, avg_width, winkler_score, width_abs_corr_r,
        width_abs_corr_rho, mse_by_uncertainty_q, per_horizon_coverage.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    targ = np.asarray(targets, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)

    in_interval = (targ >= lo) & (targ <= hi)
    coverage = float(np.mean(in_interval))
    avg_width = float(np.mean(hi - lo))
    winkler = winkler_score(lo, hi, targ, target_alpha)

    # ── per-sample σ (half-width) and absolute error ─────────────────────
    reduce_axes = tuple(range(1, pred.ndim)) if pred.ndim > 1 else ()
    ps_sigma = (np.mean(hi - lo, axis=reduce_axes) / 2.0
                if reduce_axes
                else ((hi - lo) / 2.0).ravel())
    ps_abs = (np.mean(np.abs(pred - targ), axis=reduce_axes)
              if reduce_axes
              else np.abs(pred - targ).ravel())

    # Correlations: σ vs |error|
    r_val = float("nan")
    rho_val = float("nan")
    if len(ps_sigma) > 2 and np.std(ps_sigma) > 1e-12:
        r_val, _ = pearsonr(ps_sigma, ps_abs)
        r_val = float(r_val)
        rho_val, _ = spearmanr(ps_sigma, ps_abs)
        rho_val = float(rho_val)

    # ── MSE stratified by uncertainty quartile ───────────────────────────
    ps_mse = (np.mean((pred - targ) ** 2, axis=reduce_axes)
              if reduce_axes
              else ((pred - targ) ** 2).ravel())
    mse_by_q = {}
    if len(ps_sigma) >= 4 and np.std(ps_sigma) > 1e-12:
        q25, q50, q75 = np.percentile(ps_sigma, [25, 50, 75])
        bins = [
            ("Q1 (narrow)", ps_sigma <= q25),
            ("Q2", (ps_sigma > q25) & (ps_sigma <= q50)),
            ("Q3", (ps_sigma > q50) & (ps_sigma <= q75)),
            ("Q4 (wide)", ps_sigma > q75),
        ]
        for label, mask in bins:
            if np.any(mask):
                mse_by_q[label] = float(np.mean(ps_mse[mask]))

    # ── Per-horizon coverage ─────────────────────────────────────────────
    per_horizon_cov = []
    if pred.ndim == 2 and pred.shape[1] > 1:
        for h in range(pred.shape[1]):
            per_horizon_cov.append(float(np.mean(in_interval[:, h])))

    return {
        "coverage": coverage,
        "avg_width": avg_width,
        "winkler_score": winkler,
        "width_abs_corr_r": r_val,
        "width_abs_corr_rho": rho_val,
        # legacy key kept for backward compatibility
        "width_mse_corr": r_val,
        "mse_by_uncertainty_q": mse_by_q,
        "per_horizon_coverage": per_horizon_cov,
    }


def evaluate_aci_from_saved(
    results_dir: str,
    model_name: str | None = None,
    timestamp: str | None = None,
    gamma: float = 0.005,
) -> dict:
    """Run ACI + locally-weighted ACI on saved benchmark predictions.

    Always denormalizes predictions/targets to match the original-scale
    validation residuals.

    Parameters
    ----------
    results_dir : str
        Path to benchmark results directory.
    model_name : str, optional
        Single model to evaluate.  If ``None``, evaluates all models.
    timestamp : str, optional
        Specific benchmark run timestamp.
    gamma : float
        ACI learning rate.

    Returns
    -------
    dict
        ``{model_name: {'aci': {...}, 'static': {...}, ...}}``.
    """
    from pathlib import Path
    # Import load_predictions lazily to avoid circular imports
    from epilearn.benchmark import load_predictions

    results_dir = Path(results_dir)

    # Discover models
    if model_name:
        model_names = [model_name]
    else:
        if timestamp:
            mdir = results_dir / f"models_{timestamp}"
        else:
            dirs = sorted(results_dir.glob("models_*"), key=lambda p: p.name)
            mdir = dirs[-1] if dirs else None
        if mdir is None or not mdir.exists():
            return {}
        model_names = sorted(set(
            p.stem.replace("_predictions", "")
            for p in mdir.glob("*_predictions.npz")
        ))

    all_results = {}
    for mname in model_names:
        try:
            saved = load_predictions(results_dir, mname, timestamp)
        except FileNotFoundError:
            continue

        conformal_alpha = saved.get("conformal_alpha", 0.1)

        fold_aci, fold_static, fold_lw_static, fold_lw_aci, fold_traces = [], [], [], [], []
        for i, fold in enumerate(saved["folds"]):
            val_res = fold.get("val_residuals")
            if val_res is None:
                continue

            pred = fold["predictions"].astype(np.float64)
            targ = fold["targets"].astype(np.float64)
            val_res = val_res.astype(np.float64)

            # Denormalize
            if i < len(saved["process_history"]):
                st = saved["process_history"][i]
                if "target_mean" in st and "target_std" in st:
                    tstd, tmean = float(st["target_std"]), float(st["target_mean"])
                    pred = pred * tstd + tmean
                    targ = targ * tstd + tmean

            # ACI
            aci = compute_aci(val_res, pred, targ,
                              target_alpha=conformal_alpha, gamma=gamma)
            if aci is None:
                continue
            aci_uq = compute_uncertainty_metrics(
                pred, targ, aci["lower"], aci["upper"], conformal_alpha)

            # Static conformal
            static_q = fold.get("conformal_quantile")
            if static_q is None:
                static_q = float(np.quantile(val_res, 1 - conformal_alpha))
            s_lo, s_hi = pred - static_q, pred + static_q
            static_uq = compute_uncertainty_metrics(
                pred, targ, s_lo, s_hi, conformal_alpha)

            # Locally-weighted conformal (heteroscedastic intervals)
            lw_static = locally_weighted_conformal(
                val_res, pred, targ, target_alpha=conformal_alpha)
            lw_static_uq = (compute_uncertainty_metrics(
                pred, targ, lw_static["lower"], lw_static["upper"],
                conformal_alpha) if lw_static else None)

            # Locally-weighted ACI (heteroscedastic + adaptive)
            lw_aci = locally_weighted_aci(
                val_res, pred, targ,
                target_alpha=conformal_alpha, gamma=gamma)
            lw_aci_uq = (compute_uncertainty_metrics(
                pred, targ, lw_aci["lower"], lw_aci["upper"],
                conformal_alpha) if lw_aci else None)

            fold_aci.append(aci_uq)
            fold_static.append(static_uq)
            fold_lw_static.append(lw_static_uq)
            fold_lw_aci.append(lw_aci_uq)
            fold_traces.append({
                "alpha_trace": aci["alpha_trace"],
                "quantile_trace": aci["quantile_trace"],
                "coverage_per_sample": aci["coverage_per_sample"],
            })

        if not fold_aci:
            continue

        def _avg(dicts, key):
            vals = [d[key] for d in dicts
                    if d is not None and d.get(key) is not None and np.isfinite(d[key])]
            return float(np.mean(vals)) if vals else None

        def _build_strategy_dict(fold_dicts):
            return {
                "coverage": _avg(fold_dicts, "coverage"),
                "avg_width": _avg(fold_dicts, "avg_width"),
                "winkler_score": _avg(fold_dicts, "winkler_score"),
                "width_mse_corr": _avg(fold_dicts, "width_mse_corr"),
                "width_abs_corr_r": _avg(fold_dicts, "width_abs_corr_r"),
                "width_abs_corr_rho": _avg(fold_dicts, "width_abs_corr_rho"),
                "mse_by_uncertainty_q": (fold_dicts[0].get("mse_by_uncertainty_q", {})
                                         if fold_dicts and fold_dicts[0] else {}),
                "per_horizon_coverage": (fold_dicts[0].get("per_horizon_coverage", [])
                                         if fold_dicts and fold_dicts[0] else []),
            }

        all_results[mname] = {
            "conformal_alpha": conformal_alpha,
            "aci": _build_strategy_dict(fold_aci),
            "static": _build_strategy_dict(fold_static),
            "lw_static": _build_strategy_dict(fold_lw_static),
            "lw_aci": _build_strategy_dict(fold_lw_aci),
            "traces": fold_traces,
        }

    if model_name:
        return all_results.get(model_name, {})
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _per_element_difficulty(
    val_predictions: np.ndarray,
    difficulty_fn,
) -> np.ndarray:
    """Compute per-element difficulty for flattened calibration residuals.

    ``val_predictions`` may be ``(n_samples, horizon)`` or just ``(n,)``.
    The calibration residuals are already flattened to length
    ``n_samples × horizon``.  We need a matching difficulty vector of the
    same length.  Strategy: compute per-sample difficulty, then broadcast
    across horizons.
    """
    vp = np.asarray(val_predictions, dtype=np.float64)
    if vp.ndim <= 1:
        # 1-D: difficulty *is* per-element
        s = difficulty_fn(vp.reshape(-1, 1)).ravel()
        # If val_residuals are already flattened the same way, just return
        return s if len(s) == vp.size else np.repeat(s, max(1, vp.size // len(s)))

    # Multi-dim: (n_samples, horizon, ...)
    s_sample = difficulty_fn(vp)          # (n_samples,)
    n_elem_per_sample = int(np.prod(vp.shape[1:]))
    return np.repeat(s_sample, n_elem_per_sample)
