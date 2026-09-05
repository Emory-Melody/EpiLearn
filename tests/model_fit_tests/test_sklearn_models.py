"""
Test fitting for Scikit-Learn Models.
Tests: LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel, 
       RandomForestModel, GradientBoostingModel, SVRModel, KNNModel, DecisionTreeModel

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
    LOOKBACK, HORIZON, NUM_FEATURES, TRAIN_SAMPLES, OUTPUT_DIR,
    create_synthetic_temporal_data
)

# Import models
from epilearn.models.Temporal import (
    LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel,
    RandomForestModel, GradientBoostingModel, SVRModel, KNNModel, DecisionTreeModel
)


def visualize_sklearn_fit(model_name, preds, targets, save_path=None):
    """
    Visualize the fit for a sklearn model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Flatten for calculations
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Calculate metrics
    mse = np.mean((preds_flat - targets_flat) ** 2)
    mae = np.mean(np.abs(preds_flat - targets_flat))
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - targets_flat.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Plot 1: Time series (first 100 samples)
    ax1 = axes[0]
    n_plot = min(100, len(preds))
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
    ax2.scatter(targets_flat, preds_flat, alpha=0.3, s=10, c='blue')
    min_val = min(targets_flat.min(), preds_flat.min())
    max_val = max(targets_flat.max(), preds_flat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax2.set_title(f'{model_name}: Scatter (R²={r2:.3f})')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Plot 3: Residual histogram
    ax3 = axes[2]
    residuals = preds_flat - targets_flat
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


def test_sklearn_models():
    """
    Test all sklearn models.
    """
    print("=" * 70)
    print("Testing Scikit-Learn Models - Training Fit")
    print("=" * 70)
    print(f"Lookback: {LOOKBACK}, Horizon: {HORIZON}, Features: {NUM_FEATURES}")
    print(f"Train samples: {TRAIN_SAMPLES}")
    print("=" * 70)
    
    # Create synthetic data
    X_train, y_train = create_synthetic_temporal_data(
        TRAIN_SAMPLES, LOOKBACK, HORIZON, NUM_FEATURES, noise_level=0.1
    )
    
    # Reshape for sklearn: (samples, lookback * features)
    X_flat = X_train.numpy().reshape(TRAIN_SAMPLES, -1)
    y_flat = y_train.numpy()
    
    print(f"\nData shapes: X={X_flat.shape}, y={y_flat.shape}")
    
    # Model configurations
    models_config = {
        'LinearRegressionModel': {
            'class': LinearRegressionModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK, 
                     'num_timesteps_output': HORIZON}
        },
        'RidgeModel': {
            'class': RidgeModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'alpha': 1.0}
        },
        'LassoModel': {
            'class': LassoModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'alpha': 0.01}
        },
        'ElasticNetModel': {
            'class': ElasticNetModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'alpha': 0.01, 'l1_ratio': 0.5}
        },
        'RandomForestModel': {
            'class': RandomForestModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'n_estimators': 100, 'max_depth': 10}
        },
        'GradientBoostingModel': {
            'class': GradientBoostingModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'n_estimators': 100, 'max_depth': 5}
        },
        'SVRModel': {
            'class': SVRModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'C': 1.0, 'kernel': 'rbf'}
        },
        'KNNModel': {
            'class': KNNModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'n_neighbors': 5}
        },
        'DecisionTreeModel': {
            'class': DecisionTreeModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'max_depth': 10}
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
            print("  Fitting model...")
            model.fit(X_flat, y_flat)
            
            # Predict on training data
            preds = model.predict(X_flat)
            
            # Convert to numpy if needed
            if hasattr(preds, 'numpy'):
                preds = preds.numpy()
            if hasattr(preds, 'cpu'):
                preds = preds.cpu().numpy()
            
            # Visualize
            save_path = os.path.join(OUTPUT_DIR, f'{model_name}_fit.png')
            metrics = visualize_sklearn_fit(model_name, preds, y_flat, save_path)
            
            results[model_name] = {
                'status': 'PASS' if metrics['r2'] > 0.5 else 'WARN',
                **metrics
            }
            
            status = "✅ PASS" if metrics['r2'] > 0.5 else "⚠️ WARN (low R²)"
            print(f"  Result: {status} - R²={metrics['r2']:.3f}, MSE={metrics['mse']:.4f}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'status': 'FAIL', 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Scikit-Learn Models")
    print("=" * 70)
    
    for model_name, result in results.items():
        if result['status'] == 'PASS':
            print(f"  ✅ {model_name}: R²={result['r2']:.3f}, MSE={result['mse']:.4f}")
        elif result['status'] == 'WARN':
            print(f"  ⚠️ {model_name}: R²={result['r2']:.3f}, MSE={result['mse']:.4f} (needs tuning)")
        else:
            print(f"  ❌ {model_name}: {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    results = test_sklearn_models()
