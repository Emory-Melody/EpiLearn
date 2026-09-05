"""
Epidemic-Specific Metrics for Forecasting Evaluation
=====================================================
Metrics designed for public health decision-making, emphasizing outbreak detection
and epidemic trajectory accuracy over general forecasting error.

All metrics accept an optional ``inputs`` parameter — the lookback window that the
model observed before making its prediction.  When provided, thresholds are
calibrated against the full visible time series (input + target) rather than the
test targets alone, yielding epidemiologically sounder baselines.

Key Metrics:
- Outbreak Detection Recall: Sensitivity to high-value periods (critical for early warning)
- Alert Sensitivity: Threshold-crossing detection accuracy
- Peak Underestimate Rate: Tendency to underestimate during peaks
- Rising Phase MAE: Error during epidemic growth phases
- Trend Accuracy: Directional accuracy (does the model predict the correct trend?)
"""

import numpy as np
from typing import Optional

EPSILON = 1e-8


def _extract_target_series(inputs: np.ndarray) -> np.ndarray:
    """Extract a 1-D target series from the input lookback window.

    For each sample, takes the last timestep of the last feature column.
    Handles shapes: (N, T, F), (N, T), or (N,).
    For nowcast data with -1 sentinels, takes the last valid (non -1) value.
    """
    if inputs.ndim == 1:
        return inputs
    if inputs.ndim == 2:
        # (N, T) — take last timestep
        return inputs[:, -1]
    # (N, T, F) or higher — take last timestep, last feature
    last_step = inputs[:, -1, :]  # (N, F)
    # Handle nowcast -1 sentinels: take last valid value per sample
    if np.any(last_step == -1):
        result = np.zeros(len(last_step))
        for i in range(len(last_step)):
            valid = last_step[i] != -1
            if np.any(valid):
                result[i] = last_step[i, np.where(valid)[0][-1]]
            else:
                result[i] = 0.0
        return result
    return last_step[:, -1]


def _build_baseline(target: np.ndarray, inputs: Optional[np.ndarray]) -> np.ndarray:
    """Build a baseline series for threshold calibration.

    If inputs are provided, concatenates the input target series with the
    prediction targets to calibrate thresholds against the full visible history.
    Otherwise falls back to target-only calibration.
    """
    if inputs is None:
        return target.flatten()
    input_series = _extract_target_series(inputs)
    return np.concatenate([input_series.flatten(), target.flatten()])


def compute_outbreak_recall(pred: np.ndarray, target: np.ndarray,
                            inputs: Optional[np.ndarray] = None,
                            threshold_percentile: float = 50.0,
                            threshold_std_factor: float = 0.5) -> float:
    """
    Outbreak Detection Recall: When actual values are high, how often does model predict high?

    Formula: Recall = TP / (TP + FN) where:
        - Baseline = concat(input_target_series, target) if inputs given, else target
        - High threshold tau = percentile(baseline) + factor * std(baseline)
        - TP = sum(pred > tau AND target > tau)
        - FN = sum(pred <= tau AND target > tau)

    Args:
        pred: Predictions [N,] or [N, D]
        target: Targets [N,] or [N, D]
        inputs: Optional lookback window [N, T, F] for historically-calibrated threshold
        threshold_percentile: Base percentile for threshold (default: 50 = median)
        threshold_std_factor: Std multiplier added to percentile (default: 0.5)

    Returns:
        Recall percentage (0-100), or None if insufficient high values
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    baseline = _build_baseline(target, inputs)
    threshold = np.percentile(baseline, threshold_percentile) + threshold_std_factor * np.std(baseline)

    actual_high = target_flat > threshold
    pred_high = pred_flat > threshold

    n_actual_high = np.sum(actual_high)
    if n_actual_high < 3:
        return None

    true_positives = np.sum(actual_high & pred_high)
    recall = true_positives / n_actual_high * 100

    return float(recall)


def compute_alert_sensitivity(pred: np.ndarray, target: np.ndarray,
                              inputs: Optional[np.ndarray] = None,
                              threshold_std_factor: float = 1.0) -> float:
    """
    Alert Sensitivity: When values exceed alert threshold, does model also predict above?

    Formula: Sensitivity = TP / P where:
        - Baseline = concat(input_target_series, target) if inputs given, else target
        - Alert threshold alpha = mean(baseline) + factor * std(baseline)
        - P = count(target > alpha)
        - TP = count(pred > alpha AND target > alpha)

    Args:
        pred: Predictions
        target: Targets
        inputs: Optional lookback window for historically-calibrated threshold
        threshold_std_factor: Std multiplier above mean for alert threshold

    Returns:
        Sensitivity percentage (0-100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    baseline = _build_baseline(target, inputs)
    threshold = np.mean(baseline) + threshold_std_factor * np.std(baseline)

    actual_alert = target_flat > threshold
    pred_alert = pred_flat > threshold

    n_actual_alert = np.sum(actual_alert)
    if n_actual_alert < 2:
        return None

    true_positives = np.sum(actual_alert & pred_alert)
    sensitivity = true_positives / n_actual_alert * 100

    return float(sensitivity)


def compute_peak_underestimate_rate(pred: np.ndarray, target: np.ndarray,
                                    inputs: Optional[np.ndarray] = None,
                                    top_percentile: float = 90.0) -> float:
    """
    Peak Underestimate Rate: At peak values, how often does model underestimate?

    Formula: Rate = count(pred < target | target in top 10%) / count(top 10%)

    Lower is better - underestimating peaks leads to under-prepared resources.

    Args:
        pred: Predictions
        target: Targets
        inputs: Optional lookback window (used to calibrate peak threshold)
        top_percentile: Percentile defining "peak" values (default: 90 = top 10%)

    Returns:
        Underestimate rate percentage (0-100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    baseline = _build_baseline(target, inputs)
    threshold = np.percentile(baseline, top_percentile)
    peak_indices = target_flat >= threshold

    n_peaks = np.sum(peak_indices)
    if n_peaks < 2:
        return None

    underestimates = pred_flat[peak_indices] < target_flat[peak_indices]
    rate = np.mean(underestimates) * 100

    return float(rate)


def compute_rising_phase_mae(pred: np.ndarray, target: np.ndarray,
                             inputs: Optional[np.ndarray] = None,
                             last_input: Optional[np.ndarray] = None,
                             increase_threshold: float = 0.1) -> float:
    """
    Rising Phase MAE: Error during epidemic growth (increasing) phases.

    Formula: MAE = mean(|pred - target|) for samples where target > last_input + threshold

    The last observed value is extracted from ``inputs`` (last timestep) or
    provided directly via ``last_input``.

    Args:
        pred: Predictions
        target: Targets
        inputs: Lookback window [N, T, F] — last_input extracted from last timestep
        last_input: Explicit last observed values (overrides inputs if both given)
        increase_threshold: Minimum increase to count as "rising"

    Returns:
        MAE during rising phases, or None if too few rising samples
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if last_input is None and inputs is not None:
        last_input = _extract_target_series(inputs)
    if last_input is None:
        return None

    last_flat = last_input.flatten()

    # Broadcast: for multi-horizon targets, repeat last_input per horizon step
    if len(last_flat) != len(target_flat) and target.ndim > 1:
        n_samples, horizon = target.shape[:2] if target.ndim >= 2 else (len(target), 1)
        last_flat = np.repeat(last_flat[:n_samples], horizon)

    if len(last_flat) != len(target_flat):
        return None

    actual_change = target_flat - last_flat
    rising = actual_change > increase_threshold

    n_rising = np.sum(rising)
    if n_rising < 3:
        return None

    mae = np.mean(np.abs(pred_flat[rising] - target_flat[rising]))
    return float(mae)


def compute_trend_accuracy(pred: np.ndarray, target: np.ndarray,
                           inputs: Optional[np.ndarray] = None,
                           last_input: Optional[np.ndarray] = None) -> float:
    """
    Trend Accuracy: Does the model predict the correct direction (up/down)?

    For each sample, compares the sign of (prediction - last_input) against
    (target - last_input).  Measures whether the model captures the epidemic
    trajectory direction.

    Args:
        pred: Predictions
        target: Targets
        inputs: Lookback window [N, T, F]
        last_input: Explicit last observed values (overrides inputs)

    Returns:
        Accuracy percentage (0-100), or None if insufficient data
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if last_input is None and inputs is not None:
        last_input = _extract_target_series(inputs)
    if last_input is None:
        return None

    last_flat = last_input.flatten()

    # Broadcast for multi-horizon
    if len(last_flat) != len(target_flat) and target.ndim > 1:
        n_samples = target.shape[0]
        horizon = target_flat.size // n_samples
        last_flat = np.repeat(last_flat[:n_samples], horizon)

    if len(last_flat) != len(target_flat):
        return None

    pred_direction = np.sign(pred_flat - last_flat)
    actual_direction = np.sign(target_flat - last_flat)

    # Exclude flat cases (no change)
    non_flat = actual_direction != 0
    if np.sum(non_flat) < 3:
        return None

    correct = pred_direction[non_flat] == actual_direction[non_flat]
    accuracy = np.mean(correct) * 100

    return float(accuracy)
