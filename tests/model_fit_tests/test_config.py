"""
Configuration for model fitting tests.
"""
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data configuration
LOOKBACK = 14
HORIZON = 4
NUM_NODES = 10
NUM_FEATURES = 3
TRAIN_SAMPLES = 200
TEST_SAMPLES = 50

# Training configuration
EPOCHS = 50  # Fewer epochs for faster testing
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 15

# Paths
# Plots and the JSON report land here. Override with EPILEARN_TEST_OUTPUT_DIR to
# keep the working tree clean (e.g. EPILEARN_TEST_OUTPUT_DIR=/tmp/epilearn_tests).
OUTPUT_DIR = os.environ.get(
    'EPILEARN_TEST_OUTPUT_DIR',
    os.path.join(os.path.dirname(__file__), 'outputs'),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_synthetic_temporal_data(n_samples, lookback, horizon, n_features, noise_level=0.1):
    """
    Create synthetic temporal data with a learnable pattern.
    The target is a combination of the input features to ensure it's predictable.
    """
    # Create input features with temporal patterns
    t = np.linspace(0, 4 * np.pi, n_samples + lookback + horizon)
    
    # Generate features with different patterns
    features = np.zeros((len(t), n_features))
    features[:, 0] = np.sin(t) + noise_level * np.random.randn(len(t))
    if n_features > 1:
        features[:, 1] = np.cos(t * 0.5) + noise_level * np.random.randn(len(t))
    if n_features > 2:
        features[:, 2] = np.sin(t * 2) * 0.5 + noise_level * np.random.randn(len(t))
    
    # Target is a weighted combination of features (easy to learn)
    target = 0.5 * features[:, 0] + 0.3 * features[:, 1] + 0.2 * features[:, 2]
    target += noise_level * np.random.randn(len(t))
    
    # Create sliding windows
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        X_list.append(features[i:i+lookback])
        y_list.append(target[i+lookback:i+lookback+horizon])
    
    X = np.array(X_list)  # (n_samples, lookback, n_features)
    y = np.array(y_list)  # (n_samples, horizon)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_synthetic_graph_data(n_samples, lookback, horizon, n_nodes, n_features, noise_level=0.1):
    """
    Create synthetic spatiotemporal data with graph structure.
    """
    # Create adjacency matrix (random graph with some structure)
    adj = np.random.rand(n_nodes, n_nodes) * 0.3
    adj = (adj + adj.T) / 2  # Symmetric
    np.fill_diagonal(adj, 1.0)
    adj = (adj > 0.5).astype(float)  # Threshold to create edges
    np.fill_diagonal(adj, 1.0)
    
    # Create temporal patterns for each node
    t = np.linspace(0, 4 * np.pi, n_samples + lookback + horizon)
    
    # Features: (time, nodes, features)
    all_features = np.zeros((len(t), n_nodes, n_features))
    all_targets = np.zeros((len(t), n_nodes))
    
    for node in range(n_nodes):
        phase = node * 0.3  # Different phase per node
        all_features[:, node, 0] = np.sin(t + phase) + noise_level * np.random.randn(len(t))
        if n_features > 1:
            all_features[:, node, 1] = np.cos(t * 0.5 + phase) + noise_level * np.random.randn(len(t))
        if n_features > 2:
            all_features[:, node, 2] = np.sin(t * 2 + phase) * 0.5 + noise_level * np.random.randn(len(t))
        
        # Target includes neighbor influence
        neighbor_effect = 0
        for neighbor in range(n_nodes):
            if adj[node, neighbor] > 0 and neighbor != node:
                neighbor_effect += 0.1 * np.sin(t + neighbor * 0.3)
        
        all_targets[:, node] = (
            0.5 * all_features[:, node, 0] + 
            0.3 * all_features[:, node, 1] + 
            0.1 * neighbor_effect +
            noise_level * np.random.randn(len(t))
        )
    
    # Create sliding windows: (samples, lookback, nodes, features)
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        X_list.append(all_features[i:i+lookback])
        y_list.append(all_targets[i+lookback:i+lookback+horizon])
    
    X = np.array(X_list)  # (n_samples, lookback, nodes, features)
    y = np.array(y_list)  # (n_samples, horizon, nodes)
    
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(adj, dtype=torch.float32)
    )
