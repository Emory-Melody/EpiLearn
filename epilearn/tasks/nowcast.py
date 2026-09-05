"""
Nowcasting Task for Epidemiological Data.

Standard formulation:
- At time T, we observe reports up to a minimum delay d_min
- For times in [T-d_min-W+1, T-d_min], we have incomplete reports (window of size W)
- Goal: Predict final values for this incomplete window

This task extends BaseTask and supports ALL models from epilearn/models by simply
changing the model prototype, just like the Forecast task in temporal.ipynb.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from .base import BaseTask
from ..data import Dataset


class NowcastTask(BaseTask):
    """
    Nowcasting task for epidemiological reporting triangles.
    
    Extends BaseTask to use the same model interface as Forecast.
    Supports ALL models from epilearn/models:
    - Temporal: GRU, LSTM, MLP, Transformer, etc.
    - Statistical: ARIMA, etc.
    
    Data format:
    - Features: (batch, lookback, n_regions, n_delays)
      - At each time step, input the n_delays vector
      - Unobserved values are marked as -1
      - n_delays = max_delay - min_delay + 1
    - Targets: (batch, n_regions, horizon)
      - Final counts for the last horizon days
    
    Usage:
        from epilearn.tasks import Nowcast
        from epilearn.models.Temporal import GRUModel
        
        # Create task with any model
        task = NowcastTask(prototype=GRUModel, lookback=30, horizon=7, 
                          min_delay=3, max_delay=30)
        
        # Load triangle data and create dataset
        data = task.load_triangle('data.npz')
        dataset = task.create_dataset(data['triangle'], data['final_counts'])
        
        # Run training with rolling evaluation (uses base.py's rolling_train)
        results = task.rolling_train(
            dataset=dataset,
            train_size=500,
            test_size=100,
            val_size=100,
        )
    """
    
    def __init__(
        self,
        prototype=None,
        model=None,
        lookback: int = 30,
        horizon: int = 7,
        min_delay: int = 3,
        max_delay: Optional[int] = None,
        device: str = 'cpu',
    ):
        """
        Initialize NowcastTask.
        
        Args:
            prototype: Model class (e.g., GRUModel, LSTMModel, MLPModel)
            model: Pre-initialized model (optional)
            lookback: Number of past days to use as features (L)
            horizon: Number of days to nowcast (W)
            min_delay: Minimum reporting delay (first delay column to use)
            max_delay: Maximum reporting delay (last delay column to use).
                      If None, uses all delays from the triangle.
            device: 'cpu' or 'cuda'
        """
        super().__init__(prototype, model, None, lookback, horizon, 0, device)
        self.min_delay = min_delay
        self.max_delay = max_delay  # Will be set properly in create_dataset
        self.L = lookback
        self.W = horizon
        self.n_delays = None  # Number of delay columns used
        self._delays_range = None  # (start_idx, end_idx) in triangle
        self._extract_univariate = False  # Set True to collapse triangle to univariate

        # Store latest_obs for naive baseline (populated in create_dataset)
        self._latest_obs = None
    
    @staticmethod
    def load_triangle(path: str) -> Dict[str, np.ndarray]:
        """Load reporting triangle from .npz file."""
        data = np.load(path)
        return {
            'triangle': data['triangle'],
            'final_counts': data['final_counts'],
            'delays': data['delays'],
            'time_values': data['time_values'],
        }
    
    def _build_sample(
        self,
        triangle: np.ndarray,
        final_counts: np.ndarray,
        T: int,
        delay_start_idx: int,
        delay_end_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build one training sample at time T.
        
        Args:
            triangle: Full triangle (n_days, total_delays)
            final_counts: Final counts per day
            T: Current time point
            delay_start_idx: Start index in delay columns
            delay_end_idx: End index in delay columns (exclusive)
        
        Returns:
            X: (L, n_delays) feature matrix with -1 for unobserved
            y_final: (W,) ground truth final counts
            latest_obs: (W,) latest available observation for each day (naive baseline)
        """
        n_days = triangle.shape[0]
        n_delays = delay_end_idx - delay_start_idx
        latest_day = T - self.min_delay
        
        # Build feature matrix: L days of history
        # Each row is n_delays vector, -1 for unobserved
        X = np.full((self.L, n_delays), -1.0, dtype=np.float32)
        for i in range(self.L):
            day = latest_day - self.L + 1 + i
            if day < 0 or day >= n_days:
                continue
            # max_d is the maximum delay index available at time T for this day
            max_d_available = T - day - self.min_delay
            for d_idx in range(n_delays):
                actual_d = delay_start_idx + d_idx  # Actual delay index in triangle
                if actual_d <= max_d_available:
                    val = triangle[day, actual_d]
                    X[i, d_idx] = val if not np.isnan(val) else -1.0
        
        # Build targets: W days in the incomplete window
        y_final = np.zeros(self.W, dtype=np.float32)
        latest_obs = np.zeros(self.W, dtype=np.float32)
        
        for k in range(self.W):
            day = latest_day - self.W + 1 + k
            if day < 0 or day >= n_days:
                continue
            y_final[k] = final_counts[day]
            # Find latest observation for naive baseline
            max_d_available = T - day - self.min_delay
            for actual_d in range(min(max_d_available, delay_end_idx - 1), delay_start_idx - 1, -1):
                if actual_d >= 0:
                    val = triangle[day, actual_d]
                    if not np.isnan(val):
                        latest_obs[k] = val
                        break
        
        return X, y_final, latest_obs
    
    def create_dataset(
        self,
        triangle: np.ndarray,
        final_counts: np.ndarray,
        delays: Optional[np.ndarray] = None,
    ) -> Dataset:
        """
        Convert reporting triangle to Dataset for use with rolling_train.
        
        Args:
            triangle: (n_days, total_delays) reporting triangle
            final_counts: (n_days,) final values for each day
            delays: Optional array of delay values (e.g., [3, 4, 5, ..., 60]).
                   Used to map min_delay/max_delay to column indices.
        
        Returns:
            Dataset with:
            - x: (n_samples, lookback, n_regions=1, n_delays) - features
            - y: (n_samples, n_regions=1, horizon) - targets
            
        Note:
            - n_delays = max_delay - min_delay + 1 (or all columns if max_delay not set)
            - Unobserved values in features are marked as -1
            - Models should handle -1 appropriately (mask or replace)
        """
        n_days, total_delays = triangle.shape
        
        # Determine delay column range
        if delays is not None:
            # Map delay values to column indices
            delay_values = np.array(delays)
            delay_start_idx = 0
            delay_end_idx = total_delays
            
            # Find start index (first column with delay >= min_delay)
            if self.min_delay is not None:
                matches = np.where(delay_values >= self.min_delay)[0]
                if len(matches) > 0:
                    delay_start_idx = matches[0]
            
            # Find end index (last column with delay <= max_delay)
            if self.max_delay is not None:
                matches = np.where(delay_values <= self.max_delay)[0]
                if len(matches) > 0:
                    delay_end_idx = matches[-1] + 1
        else:
            # No delay values provided - assume columns represent delays starting at min_delay
            # E.g., if min_delay=3 and triangle has 58 columns, delays are [3, 4, 5, ..., 60]
            # So column index = delay_value - min_delay
            delay_start_idx = 0  # First column corresponds to min_delay
            if self.max_delay is not None:
                # max_delay -> column index = max_delay - min_delay
                delay_end_idx = min(self.max_delay - self.min_delay + 1, total_delays)
            else:
                delay_end_idx = total_delays
        
        self.n_delays = delay_end_idx - delay_start_idx
        self._delays_range = (delay_start_idx, delay_end_idx)
        
        # Update max_delay if not set
        if self.max_delay is None:
            self.max_delay = total_delays + self.min_delay - 1
        
        start_T = self.min_delay + self.L + self.W
        
        X_list, y_list, latest_list = [], [], []
        for T in range(start_T, n_days + self.min_delay):
            X, y_final, latest = self._build_sample(
                triangle, final_counts, T, delay_start_idx, delay_end_idx
            )
            X_list.append(X)
            y_list.append(y_final)
            latest_list.append(latest)
        
        X = np.array(X_list)  # (n_samples, L, n_delays)
        y = np.array(y_list)  # (n_samples, W)
        latest = np.array(latest_list)  # (n_samples, W)
        
        # Store latest_obs for naive baseline computation
        self._latest_obs = latest
        
        features = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
        
        # Shape: (n_samples, lookback, n_regions=1, n_delays)
        features = features.unsqueeze(2)  # (n_samples, L, 1, n_delays)
        
        # Shape: (n_samples, n_regions=1, horizon)
        targets = targets.unsqueeze(1)    # (n_samples, 1, W)
        
        # Create Dataset using epilearn.data.Dataset
        dataset = Dataset(
            x=features,
            y=targets,
            timestamps=list(range(len(X))),
        )
        
        return dataset
    
    def _generate_split(self, ds, lookback=None, interval=None):
        """
        Generate split dictionary from Dataset.

        Overrides BaseTask._generate_split to handle nowcast-specific shapes.

        Input: ds.x shape (T, lookback, n_regions, n_delays)
        Output features: (T, lookback, n_delays) for temporal models

        Args:
            lookback: If provided and < features dim 1, crop to last lookback rows.
            interval: Ignored (kept for API compatibility with base class).
        """
        features = ds.x  # (T, lookback, n_regions=1, n_delays)
        targets = ds.y   # (T, n_regions=1, horizon)

        # Apply lookback cropping if requested (Optuna lookback tuning)
        if lookback is not None and lookback < features.shape[1]:
            features = features[:, -lookback:, :, :]

        # Squeeze region dimension for single-region nowcasting
        # (T, lookback, 1, n_delays) -> (T, lookback, n_delays)
        if features.shape[2] == 1:
            features = features.squeeze(2)

        # (T, 1, horizon) -> (T, horizon)
        if len(targets.shape) == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        # Extract univariate series from triangle: (T, L, 28) -> (T, L, 1)
        # Takes the last valid (non -1) value per row, same extraction as
        # ARIMA/foundation but WITHOUT truncating the target window — NNs
        # benefit from seeing partial observations in the target window
        # (correlated with final counts), unlike ARIMA which extrapolates
        # from history only.
        if self._extract_univariate and features.shape[-1] > 1:
            mask = (features != -1)
            reversed_mask = mask.flip(dims=[2])
            last_valid_idx = (features.shape[2] - 1
                              - reversed_mask.float().argmax(dim=2))
            features = features.gather(
                2, last_valid_idx.unsqueeze(2).long()
            )  # (T, L, 1)
            # Zero out rows with no valid data at all
            features = features * mask.any(dim=2, keepdim=True).float()

        return {
            'features': features.float().to(self.device),
            'targets': targets.float().to(self.device),
            'graph': ds.graph,
            'dynamic_graph': ds.dynamic_graph,
            'states': ds.states,
        }
    
    def _init_model_from_split(self, train_split, model_args=None):
        """
        Initialize model from train split.
        
        Features shape: (batch, lookback, n_delays)
        - num_timesteps_input = lookback (sequence length)
        - num_features = n_delays (input at each time step)
        """
        if model_args is None:
            model_args = {}
        
        features = train_split['features']
        targets = train_split['targets']
        
        # Features: (batch, lookback, n_delays)
        num_timesteps_input = features.shape[1]  # lookback
        num_features = features.shape[2]         # n_delays
        num_timesteps_output = targets.shape[-1] # horizon
        
        base_inputs = {
            'num_features': num_features,
            'num_timesteps_input': num_timesteps_input,
            'num_timesteps_output': num_timesteps_output,
            'device': self.device,
            'nowcast': True,
        }
        base_inputs.update(model_args)
        
        self.model = self.prototype(**base_inputs)
        self.model = self.model.to(self.device)
    
    def compute_naive_baseline(self, dataset: Dataset) -> Dict:
        """
        Compute naive baseline metrics for comparison.
        
        The naive baseline uses the latest available observation as the prediction.
        """
        if self._latest_obs is None:
            return {'naive_mae': None, 'message': 'No latest_obs available'}
        
        targets = dataset.y.squeeze(1).numpy()  # (T, W)
        latest = self._latest_obs
        
        mask = (targets != 0) & (latest != 0)
        if mask.sum() > 0:
            naive_mae = np.abs(latest[mask] - targets[mask]).mean()
            return {
                'naive_mae': naive_mae,
                'n_samples': mask.sum(),
            }
        return {'naive_mae': None, 'message': 'No valid samples'}


# Alias for compatibility
Nowcast = NowcastTask
