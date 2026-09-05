"""
Test fitting for Spatiotemporal Models.
Tests: STGCN, DCRNN, GraphWaveNet, EpiGNN, ColaGNN, ATMGNN

This script trains each model on synthetic graph data and visualizes:
1. Training loss curve
2. Predictions vs Actual on training data (per node)
3. Scatter plot with R²
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
    DEVICE, LOOKBACK, HORIZON, NUM_NODES, NUM_FEATURES, TRAIN_SAMPLES,
    EPOCHS, BATCH_SIZE, LR, PATIENCE, OUTPUT_DIR,
    create_synthetic_graph_data
)

# Import models
from epilearn.models.SpatialTemporal import (
    STGCN, DCRNN, GraphWaveNet, EpiGNN, ColaGNN, ATMGNN
)


def train_spatiotemporal_model(model, X_train, y_train, adj, 
                                epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, 
                                patience=PATIENCE, device=DEVICE):
    """
    Train a spatiotemporal model and return training history.
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    adj = adj.to(device)
    
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
            
            # Forward pass with graph
            output = model(X_batch, adj, None, None)
            
            # Handle output shape: (batch, nodes, horizon) or (batch, horizon, nodes)
            if len(output.shape) == 3:
                # Check if we need to transpose
                if output.shape[1] == y_batch.shape[2] and output.shape[2] == y_batch.shape[1]:
                    # (batch, nodes, horizon) -> compare with (batch, horizon, nodes)
                    output = output.permute(0, 2, 1)
            
            # Ensure shapes match
            if output.shape != y_batch.shape:
                # Try to match shapes
                if len(output.shape) == 3 and len(y_batch.shape) == 3:
                    # Transpose if needed
                    if output.shape[1:] == y_batch.shape[1:][::-1]:
                        output = output.permute(0, 2, 1)
            
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
        train_preds = model(X_train, adj, None, None)
        
        # Handle shape
        if len(train_preds.shape) == 3:
            if train_preds.shape[1] == y_train.shape[2] and train_preds.shape[2] == y_train.shape[1]:
                train_preds = train_preds.permute(0, 2, 1)
    
    history['train_preds'] = train_preds.cpu().numpy()
    history['train_targets'] = y_train.cpu().numpy()
    
    return history


def visualize_spatiotemporal_fit(model_name, history, n_nodes_to_show=3, save_path=None):
    """
    Visualize the training fit for a spatiotemporal model.
    """
    preds = history['train_preds']
    targets = history['train_targets']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Training Loss (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history['loss'], 'b-', linewidth=1.5)
    ax1.set_title(f'{model_name}: Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    # Use log scale only if all losses are positive
    if min(history['loss']) > 1e-10:
        ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    final_loss = history['loss'][-1]
    ax1.axhline(y=final_loss, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_loss:.4f}')
    ax1.legend()
    
    # Plot 2: Overall Scatter (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    ax2.scatter(targets_flat, preds_flat, alpha=0.2, s=5, c='blue')
    min_val = min(targets_flat.min(), preds_flat.min())
    max_val = max(targets_flat.max(), preds_flat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    mse = np.mean((preds_flat - targets_flat) ** 2)
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - targets_flat.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    ax2.set_title(f'{model_name}: Overall Scatter (R²={r2:.3f})')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual distribution (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    residuals = preds_flat - targets_flat
    ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_title(f'{model_name}: Residuals (MSE={mse:.4f})')
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Per-node time series (bottom row)
    # Shape should be (samples, horizon, nodes) or (samples, nodes, horizon)
    if len(preds.shape) == 3:
        # Determine which dimension is nodes
        if preds.shape[2] > preds.shape[1]:  # (samples, horizon, nodes)
            n_nodes = preds.shape[2]
            for i, node_idx in enumerate(range(min(n_nodes_to_show, n_nodes))):
                ax = fig.add_subplot(2, 3, 4 + i)
                n_plot = min(50, preds.shape[0])
                
                # (samples, horizon, nodes) -> take first timestep, specific node
                node_preds = preds[:n_plot, 0, node_idx]
                node_targets = targets[:n_plot, 0, node_idx]
                
                ax.plot(node_targets, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
                ax.plot(node_preds, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
                ax.set_title(f'Node {node_idx}: Train Fit')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
        else:  # (samples, nodes, horizon)
            n_nodes = preds.shape[1]
            for i, node_idx in enumerate(range(min(n_nodes_to_show, n_nodes))):
                ax = fig.add_subplot(2, 3, 4 + i)
                n_plot = min(50, preds.shape[0])
                
                # (samples, nodes, horizon) -> take specific node, first timestep
                node_preds = preds[:n_plot, node_idx, 0]
                node_targets = targets[:n_plot, node_idx, 0]
                
                ax.plot(node_targets, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
                ax.plot(node_preds, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
                ax.set_title(f'Node {node_idx}: Train Fit')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved plot to {save_path}")
    
    plt.close()
    
    return {'mse': mse, 'r2': r2, 'final_loss': final_loss}


def test_spatiotemporal_models():
    """
    Test all spatiotemporal models.
    """
    print("=" * 70)
    print("Testing Spatiotemporal Models - Training Fit")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Lookback: {LOOKBACK}, Horizon: {HORIZON}")
    print(f"Nodes: {NUM_NODES}, Features: {NUM_FEATURES}")
    print(f"Train samples: {TRAIN_SAMPLES}, Epochs: {EPOCHS}")
    print("=" * 70)
    
    # Create synthetic graph data
    X_train, y_train, adj = create_synthetic_graph_data(
        TRAIN_SAMPLES, LOOKBACK, HORIZON, NUM_NODES, NUM_FEATURES, noise_level=0.1
    )
    
    print(f"\nData shapes: X={X_train.shape}, y={y_train.shape}, adj={adj.shape}")
    
    # Model configurations
    models_config = {
        'STGCN': {
            'class': STGCN,
            'args': {
                'num_nodes': NUM_NODES,
                'num_features': NUM_FEATURES,
                'num_timesteps_input': LOOKBACK,
                'num_timesteps_output': HORIZON,
                'nhid': 32
            }
        },
        'DCRNN': {
            'class': DCRNN,
            'args': {
                'num_nodes': NUM_NODES,
                'num_features': NUM_FEATURES,
                'num_timesteps_input': LOOKBACK,
                'num_timesteps_output': HORIZON,
                'rnn_units': 32,
                'num_rnn_layers': 1,
                'device': DEVICE
            }
        },
        'GraphWaveNet': {
            'class': GraphWaveNet,
            'args': {
                'num_nodes': NUM_NODES,
                'num_features': NUM_FEATURES,
                'num_timesteps_input': LOOKBACK,
                'num_timesteps_output': HORIZON,
                'residual_channels': 16,
                'dilation_channels': 16,
                'skip_channels': 32,
                'end_channels': 64,
                'blocks': 2,
                'nlayers': 1,
                'device': DEVICE
            }
        },
        'EpiGNN': {
            'class': EpiGNN,
            'args': {
                'num_nodes': NUM_NODES,
                'num_features': NUM_FEATURES,
                'num_timesteps_input': LOOKBACK,
                'num_timesteps_output': HORIZON,
                'hidR': 32,
                'hidA': 32,
                'dropout': 0.1,
                'device': DEVICE
            }
        },
        'ColaGNN': {
            'class': ColaGNN,
            'args': {
                'num_nodes': NUM_NODES,
                'num_features': NUM_FEATURES,
                'num_timesteps_input': LOOKBACK,
                'num_timesteps_output': HORIZON,
                'nhid': 32,
                'dropout': 0.1,
                'device': DEVICE
            }
        },
        'ATMGNN': {
            'class': ATMGNN,
            'args': {
                'num_nodes': NUM_NODES,
                'num_features': NUM_FEATURES,
                'num_timesteps_input': LOOKBACK,
                'num_timesteps_output': HORIZON,
                'nhid': 32,
                'device': DEVICE
            }
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
            history = train_spatiotemporal_model(
                model, X_train.clone(), y_train.clone(), adj.clone()
            )
            
            # Visualize
            save_path = os.path.join(OUTPUT_DIR, f'{model_name}_fit.png')
            metrics = visualize_spatiotemporal_fit(model_name, history, save_path=save_path)
            
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
    print("SUMMARY - Spatiotemporal Models")
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
    results = test_spatiotemporal_models()
