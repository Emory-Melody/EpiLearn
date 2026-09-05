"""
Test fitting for Temporal Neural Network Models.
Tests: GRUModel, LSTMModel, CNNModel, MLPModel, DlinearModel, PatchTSTModel

This script trains each model on synthetic data and visualizes:
1. Training loss curve
2. Predictions vs Actual on training data
3. Residual distribution
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
import sys

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from test_config import (
    DEVICE, LOOKBACK, HORIZON, NUM_FEATURES, TRAIN_SAMPLES,
    EPOCHS, BATCH_SIZE, LR, PATIENCE, OUTPUT_DIR,
    create_synthetic_temporal_data
)

# Import models
from epilearn.models.Temporal import (
    GRUModel, LSTMModel, CNNModel, MLPModel, DlinearModel, PatchTSTModel
)


def train_temporal_model(model, X_train, y_train, epochs=EPOCHS, lr=LR, 
                         batch_size=BATCH_SIZE, patience=PATIENCE, device=DEVICE):
    """
    Train a temporal model and return training history.
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'loss': [], 'train_preds': None, 'train_targets': None}
    best_loss = float('inf')
    patience_counter = 0
    
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            output = model(X_batch)
            
            # Handle output shape
            if len(output.shape) == 3:
                output = output.squeeze(-1)
            if output.shape != y_batch.shape:
                output = output[:, :y_batch.shape[1]]
            
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Get final predictions on training data
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train)
        if len(train_preds.shape) == 3:
            train_preds = train_preds.squeeze(-1)
        if train_preds.shape[1] != y_train.shape[1]:
            train_preds = train_preds[:, :y_train.shape[1]]
    
    history['train_preds'] = train_preds.cpu().numpy()
    history['train_targets'] = y_train.cpu().numpy()
    
    return history


def visualize_fit(model_name, history, save_path=None):
    """
    Visualize the training fit for a model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    ax1.plot(history['loss'], 'b-', linewidth=1.5)
    ax1.set_title(f'{model_name}: Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    final_loss = history['loss'][-1]
    ax1.axhline(y=final_loss, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_loss:.4f}')
    ax1.legend()
    
    # Plot 2: Predictions vs Actual (first 100 samples, first timestep)
    ax2 = axes[1]
    preds = history['train_preds'][:100, 0] if len(history['train_preds'].shape) > 1 else history['train_preds'][:100]
    targets = history['train_targets'][:100, 0] if len(history['train_targets'].shape) > 1 else history['train_targets'][:100]
    
    ax2.plot(targets, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax2.plot(preds, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
    ax2.set_title(f'{model_name}: Train Fit (First 100 samples)')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter (Predicted vs Actual)
    ax3 = axes[2]
    preds_flat = history['train_preds'].flatten()
    targets_flat = history['train_targets'].flatten()
    
    ax3.scatter(targets_flat, preds_flat, alpha=0.3, s=10, c='blue')
    min_val = min(targets_flat.min(), preds_flat.min())
    max_val = max(targets_flat.max(), preds_flat.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    # Calculate R² and MSE
    mse = np.mean((preds_flat - targets_flat) ** 2)
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - targets_flat.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    ax3.set_title(f'{model_name}: Scatter (R²={r2:.3f}, MSE={mse:.4f})')
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Predicted')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved plot to {save_path}")
    
    plt.close()
    
    return {'mse': mse, 'r2': r2, 'final_loss': final_loss}


def test_temporal_nn_models():
    """
    Test all temporal neural network models.
    """
    print("=" * 70)
    print("Testing Temporal Neural Network Models - Training Fit")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Lookback: {LOOKBACK}, Horizon: {HORIZON}, Features: {NUM_FEATURES}")
    print(f"Train samples: {TRAIN_SAMPLES}, Epochs: {EPOCHS}")
    print("=" * 70)
    
    # Create synthetic data
    X_train, y_train = create_synthetic_temporal_data(
        TRAIN_SAMPLES, LOOKBACK, HORIZON, NUM_FEATURES, noise_level=0.1
    )
    print(f"\nData shapes: X={X_train.shape}, y={y_train.shape}")
    
    # Model configurations
    models_config = {
        'GRUModel': {
            'class': GRUModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK, 
                     'num_timesteps_output': HORIZON, 'nhids': 64, 'dropout': 0.1}
        },
        'LSTMModel': {
            'class': LSTMModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'nhid': 64, 'dropout': 0.1}
        },
        'CNNModel': {
            'class': CNNModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'conv1_hid': 16, 'conv2_hid': 32, 
                     'linear_hid': 64, 'dropout': 0.1}
        },
        'MLPModel': {
            'class': MLPModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'nhids': 64, 'dropout': 0.1}
        },
        'DlinearModel': {
            'class': DlinearModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'moving_avg_window': 7}
        },
        'PatchTSTModel': {
            'class': PatchTSTModel,
            'args': {'num_features': NUM_FEATURES, 'num_timesteps_input': LOOKBACK,
                     'num_timesteps_output': HORIZON, 'd_model': 32, 'n_heads': 4, 
                     'n_layers': 2, 'patch_len': 4}
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
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")
            
            # Train
            history = train_temporal_model(model, X_train.clone(), y_train.clone())
            
            # Visualize
            save_path = os.path.join(OUTPUT_DIR, f'{model_name}_fit.png')
            metrics = visualize_fit(model_name, history, save_path)
            
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
    print("SUMMARY - Temporal Neural Network Models")
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
    results = test_temporal_nn_models()
