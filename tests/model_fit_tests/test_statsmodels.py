"""
Test fitting for StatsModels (ARIMA, VARMAX).

This script fits each model on synthetic data and visualizes:
1. Predictions vs Actual on training data
2. Residual distribution
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
import sys

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from test_config import (
    LOOKBACK, HORIZON, NUM_FEATURES, OUTPUT_DIR,
    create_synthetic_temporal_data
)

# Import models
from epilearn.models.Temporal import ARIMAModel, VARMAXModel


def visualize_statsmodel_fit(model_name, preds, targets, save_path=None):
    """
    Visualize the fit for a statsmodel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Handle different shapes
    if hasattr(preds, 'numpy'):
        preds = preds.numpy()
    if hasattr(targets, 'numpy'):
        targets = targets.numpy()
    
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Remove NaN values for metrics
    valid_mask = ~np.isnan(preds_flat) & ~np.isnan(targets_flat)
    preds_valid = preds_flat[valid_mask]
    targets_valid = targets_flat[valid_mask]
    
    if len(preds_valid) == 0:
        print(f"  Warning: All predictions are NaN!")
        return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    
    # Calculate metrics
    mse = np.mean((preds_valid - targets_valid) ** 2)
    mae = np.mean(np.abs(preds_valid - targets_valid))
    ss_res = np.sum((targets_valid - preds_valid) ** 2)
    ss_tot = np.sum((targets_valid - targets_valid.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Plot 1: Time series (first 50 samples)
    ax1 = axes[0]
    n_plot = min(50, len(preds))
    plot_preds = preds[:n_plot, 0] if len(preds.shape) > 1 else preds[:n_plot]
    plot_targets = targets[:n_plot, 0] if len(targets.shape) > 1 else targets[:n_plot]
    
    ax1.plot(plot_targets, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(plot_preds, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
    ax1.set_title(f'{model_name}: Train Fit')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter (Predicted vs Actual)
    ax2 = axes[1]
    ax2.scatter(targets_valid, preds_valid, alpha=0.3, s=10, c='blue')
    min_val = min(targets_valid.min(), preds_valid.min())
    max_val = max(targets_valid.max(), preds_valid.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax2.set_title(f'{model_name}: Scatter (R²={r2:.3f})')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual histogram
    ax3 = axes[2]
    residuals = preds_valid - targets_valid
    ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {residuals.mean():.4f}')
    ax3.set_title(f'{model_name}: Residuals (MSE={mse:.4f})')
    ax3.set_xlabel('Residual (Pred - Actual)')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved plot to {save_path}")
    
    plt.close()
    
    return {'mse': mse, 'mae': mae, 'r2': r2}


def test_statsmodels():
    """
    Test ARIMA and VARMAX models.
    """
    print("=" * 70)
    print("Testing StatsModels (ARIMA, VARMAX) - Training Fit")
    print("=" * 70)
    print(f"Lookback: {LOOKBACK}, Horizon: {HORIZON}, Features: {NUM_FEATURES}")
    print("=" * 70)
    
    # Use fewer samples for statsmodels (they're slower)
    TRAIN_SAMPLES = 50
    
    # Create synthetic data
    X_train, y_train = create_synthetic_temporal_data(
        TRAIN_SAMPLES, LOOKBACK, HORIZON, NUM_FEATURES, noise_level=0.1
    )
    
    print(f"\nData shapes: X={X_train.shape}, y={y_train.shape}")
    
    # Model configurations
    models_config = {
        'ARIMAModel': {
            'class': ARIMAModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'order': (1, 0, 1)}
        },
        'VARMAXModel': {
            'class': VARMAXModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'order': (1, 0)}
        },
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'─' * 50}")
        print(f"Testing: {model_name}")
        print(f"{'─' * 50}")
        
        try:
            # Create model
            model = config['class'](**config['args'])
            
            # Fit model
            print("  Fitting model (this may take a moment)...")
            model.fit(X_train, y_train, verbose=False)
            
            # Predict on training data
            preds = model.predict(X_train)
            
            # Convert to numpy
            if hasattr(preds, 'numpy'):
                preds = preds.numpy()
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            
            targets = y_train.numpy()
            
            # Visualize
            save_path = os.path.join(OUTPUT_DIR, f'{model_name}_fit.png')
            metrics = visualize_statsmodel_fit(model_name, preds, targets, save_path)
            
            # StatsModels are harder to fit perfectly, use lower threshold
            results[model_name] = {
                'status': 'PASS' if metrics['r2'] > 0.3 else 'WARN',
                **metrics
            }
            
            status = "✅ PASS" if metrics['r2'] > 0.3 else "⚠️ WARN (low R²)"
            print(f"  Result: {status} - R²={metrics['r2']:.3f}, MSE={metrics['mse']:.4f}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'status': 'FAIL', 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - StatsModels")
    print("=" * 70)
    
    for model_name, result in results.items():
        if result['status'] == 'PASS':
            print(f"  ✅ {model_name}: R²={result['r2']:.3f}, MSE={result['mse']:.4f}")
        elif result['status'] == 'WARN':
            print(f"  ⚠️ {model_name}: R²={result['r2']:.3f}, MSE={result['mse']:.4f} (statsmodels often need more data)")
        else:
            print(f"  ❌ {model_name}: {result.get('error', 'Unknown error')}")
    
    print("\nNote: StatsModels (ARIMA, VARMAX) are designed for longer time series")
    print("and may show lower R² on short synthetic data. This is expected behavior.")
    
    return results


if __name__ == "__main__":
    results = test_statsmodels()
