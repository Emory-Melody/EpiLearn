import torch
import numpy as np
import optuna
from optuna.samplers import TPESampler
import time
import psutil
import os
import sys
from copy import deepcopy

from ..utils import metrics  

def _flush_print(*args, **kwargs):
    """Print and immediately flush stdout for multiprocessing compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


def _is_callable_or_module(obj):
    """Check if object is a callable loss function or nn.Module."""
    import torch.nn as nn
    return callable(obj) or isinstance(obj, nn.Module)


def _sample_optuna_params(trial, params_dict, prefix='', object_registry=None):
    """
    Sample hyperparameters from a parameter dictionary using Optuna.
    
    This is a shared helper function used by rolling_train.
    
    Args:
        trial: Optuna trial object
        params_dict: Dictionary of parameter ranges. Values can be:
            - Single value: Used as-is
            - List with 1 element: Used as fixed value  
            - List with 2 elements: Treated as [min, max] range (if numeric)
            - List with 3+ elements: Treated as categorical choices
            - Callable/nn.Module: Custom loss functions are sampled by index
        prefix: Prefix for parameter names (e.g., 'model_')
        object_registry: Optional dict to store non-hashable objects (mutated in place)
    
    Returns:
        Dictionary of sampled parameter values
    """
    if object_registry is None:
        object_registry = {}
    sampled_params = {}
    
    for param_name, param_values in params_dict.items():
        full_name = f'{prefix}{param_name}' if prefix else param_name
        
        # Handle non-list values (single fixed values)
        if not isinstance(param_values, list):
            # For callables/modules, use directly without Optuna tracking
            if _is_callable_or_module(param_values):
                sampled_params[param_name] = param_values
            else:
                sampled_params[param_name] = trial.suggest_categorical(full_name, [param_values])
            continue
        
        # Handle empty lists (skip)
        if len(param_values) == 0:
            continue
        
        # Check if list contains callables/modules (e.g., custom loss functions)
        if any(_is_callable_or_module(v) for v in param_values):
            # Store objects in registry, sample by index
            registry_key = full_name
            object_registry[registry_key] = param_values
            idx = trial.suggest_int(f'{full_name}_idx', 0, len(param_values) - 1)
            sampled_params[param_name] = param_values[idx]
            continue
        
        # Single value - register with Optuna for tracking
        if len(param_values) == 1:
            sampled_params[param_name] = trial.suggest_categorical(full_name, param_values)
            continue
        
        # Two values - check if numeric range or categorical
        if len(param_values) == 2:
            if isinstance(param_values[0], int) and isinstance(param_values[1], int):
                sampled_params[param_name] = trial.suggest_int(full_name, param_values[0], param_values[1])
            elif isinstance(param_values[0], (int, float)) and isinstance(param_values[1], (int, float)):
                # Check if log scale is appropriate (e.g., learning rate)
                if param_name == 'lr' or (param_values[1] / param_values[0] > 100 if param_values[0] > 0 else False):
                    sampled_params[param_name] = trial.suggest_float(full_name, param_values[0], param_values[1], log=True)
                else:
                    sampled_params[param_name] = trial.suggest_float(full_name, param_values[0], param_values[1])
            else:
                # Categorical (e.g., strings)
                sampled_params[param_name] = trial.suggest_categorical(full_name, param_values)
        else:
            # 3+ values - treat as categorical
            if all(isinstance(v, (int, float)) for v in param_values):
                sampled_params[param_name] = trial.suggest_categorical(full_name, param_values)
            else:
                # Handle non-primitive types
                categorical_values = []
                for v in param_values:
                    if isinstance(v, (list, tuple)):
                        categorical_values.append(str(tuple(v)))
                    else:
                        categorical_values.append(v)
                
                sampled_value = trial.suggest_categorical(full_name, categorical_values)
                
                # Convert string back to tuple if needed
                if isinstance(sampled_value, str) and sampled_value.startswith('('):
                    sampled_params[param_name] = eval(sampled_value)
                else:
                    sampled_params[param_name] = sampled_value
    
    return sampled_params


class BaseTask:
    def __init__(self, prototype, model, dataset, lookback, horizon, ahead, device='cpu'):
        self.model = model
        self.prototype = prototype
        self.dataset = dataset
        self.device = device
        self.lookback = lookback
        self.horizon = horizon
        self.ahead = ahead

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods for rolling_train
    # ─────────────────────────────────────────────────────────────────────────
    
    def _generate_split(self, ds, lookback=None, interval=None):
        """Generate sliding window dataset from a Dataset object."""
        lookback = lookback or self.lookback
        ahead = getattr(self, 'ahead', 0)
        return ds.generate_dataset(
            X=ds.x, Y=ds.y,
            states=ds.states,
            dynamic_adj=ds.dynamic_graph,
            adj=ds.graph,
            lookback_window_size=lookback,
            horizon_size=self.horizon,
            interval=interval,
            ahead=ahead
        )

    def _run_optuna_tuning(self, train_ds, val_ds, max_lookback, tuning_lookback,
                          model_args, optimizer_params, optuna_model_args,
                          lr, batch_size, weight_decay, epochs, patience,
                          train_loss, val_loss, n_trials,
                          collect_trial_details=True):
        """Run Optuna hyperparameter tuning and return best params.
        
        Args:
            collect_trial_details: If True, collect per-trial, per-node metrics
                for building recommendation models later.
                
        Returns:
            Dictionary with:
                - All best hyperparameters
                - 'optuna_trials': List of per-trial results (if collect_trial_details=True)
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Prepare data sources
        if tuning_lookback:
            optuna_train_ds, optuna_val_ds = deepcopy(train_ds), deepcopy(val_ds)
        else:
            optuna_train_split = deepcopy(self._generate_split(train_ds))
            optuna_val_split = deepcopy(self._generate_split(val_ds))
        
        # Shared registry to track callable objects (loss functions, etc.)
        object_registry = {}
        
        # Collector for detailed trial results (for recommendation model)
        trial_details = []
        
        # Get number of nodes from dataset for per-node metrics
        n_nodes = train_ds.n_regions if hasattr(train_ds, 'n_regions') else 1
        
        def objective(trial):
            # Sample hyperparameters (pass object_registry to track callables)
            opt_params = _sample_optuna_params(trial, optimizer_params, object_registry=object_registry)
            trial_model_args = _sample_optuna_params(trial, optuna_model_args, prefix='model_', object_registry=object_registry)
            trial_lookback = trial_model_args.pop('lookback', self.lookback)
            
            # Validate transformer constraint: d_model must be divisible by n_heads
            d_model = trial_model_args.get('d_model')
            n_heads = trial_model_args.get('n_heads')
            if d_model is not None and n_heads is not None and d_model % n_heads != 0:
                raise optuna.TrialPruned(f"Invalid config: d_model={d_model} not divisible by n_heads={n_heads}")
            
            # Generate or copy splits
            if tuning_lookback:
                trial_train = self._generate_split(optuna_train_ds, trial_lookback, interval=max_lookback - trial_lookback)
                trial_val = self._generate_split(optuna_val_ds, trial_lookback, interval=max_lookback - trial_lookback)
            else:
                trial_train = deepcopy(optuna_train_split)
                trial_val = deepcopy(optuna_val_split)

            # Initialize and train model
            merged_args = {**model_args, **trial_model_args}
            self._init_model_from_split(trial_train, merged_args)
            # import ipdb; ipdb.set_trace()
            try:
                result = self.train_model(
                    train_split=trial_train,
                    val_split=trial_val,
                    test_split=deepcopy(trial_val),
                    train_loss=opt_params.get('train_loss', train_loss),
                    val_loss=opt_params.get('val_loss', val_loss),
                    epochs=opt_params.get('epochs', epochs),
                    batch_size=opt_params.get('batch_size', batch_size),
                    lr=opt_params.get('lr', lr),
                    weight_decay=opt_params.get('weight_decay', weight_decay),
                    patience=patience,
                    verbose=False,
                    model_args=merged_args,
                    initialize=False,
                )
                if result is None:
                    return float('inf')
                
                # Compute aggregate val loss
                trial_val_loss = opt_params.get('val_loss', val_loss)
                loss_fn = metrics.get_loss(trial_val_loss)
                val_loss_value = loss_fn(result['predictions'], result['targets'])
                if hasattr(val_loss_value, 'item'):
                    val_loss_value = val_loss_value.item()
                
                # Collect detailed per-node metrics for recommendation model
                if collect_trial_details:
                    trial_info = self._collect_trial_node_metrics(
                        trial_number=trial.number,
                        hyperparams={**opt_params, **trial_model_args, 'lookback': trial_lookback},
                        predictions=result['predictions'],
                        targets=result['targets'],
                        train_targets=None,  # Not used anymore
                        n_nodes=n_nodes,
                        val_loss_value=val_loss_value,
                        train_start_idx=train_ds.timestamps[0] if hasattr(train_ds, 'timestamps') else None,
                        train_end_idx=train_ds.timestamps[-1] if hasattr(train_ds, 'timestamps') else None,
                        val_start_idx=val_ds.timestamps[0] if hasattr(val_ds, 'timestamps') and val_ds else None,
                        val_end_idx=val_ds.timestamps[-1] if hasattr(val_ds, 'timestamps') and val_ds else None,
                    )
                    trial_details.append(trial_info)
                
                return val_loss_value if np.isfinite(val_loss_value) else float('inf')
            except Exception as e:
                _flush_print(f"    Trial failed: {e}")
                return float('inf')
        
        # Run study (seeded sampler for reproducibility)
        _flush_print(f"  Running Optuna ({n_trials} trials)...")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        _flush_print(f"  Best val loss: {study.best_value:.4f}")
        _flush_print(f"  Best params: {study.best_params}")
        
        # Reconstruct best_params with actual objects from registry
        best_params = dict(study.best_params)
        for registry_key, objects in object_registry.items():
            idx_key = f'{registry_key}_idx'
            if idx_key in best_params:
                idx = best_params.pop(idx_key)
                param_name = registry_key.split('_', 1)[-1] if registry_key.startswith('model_') else registry_key
                best_params[param_name] = objects[idx]
        
        # Attach trial details for recommendation model
        best_params['_optuna_trials'] = trial_details
        
        return best_params

    def _collect_trial_node_metrics(self, trial_number, hyperparams, predictions, targets,
                                     train_targets, n_nodes, val_loss_value,
                                     train_start_idx=None, train_end_idx=None,
                                     val_start_idx=None, val_end_idx=None):
        """Collect per-node metrics for recommendation model.
        
        This captures data needed to build a recommendation model that learns to
        predict which hyperparameters work best for time series with certain characteristics.
        
        Instead of computing time series statistics, we save the node ID and the
        time range indices so users can reference the original data later.
        
        Args:
            trial_number: Optuna trial number
            hyperparams: Dict of hyperparameters used in this trial
            predictions: Model predictions [B, N, H, F] or similar shape
            targets: Ground truth [B, N, H, F] or similar shape
            train_targets: Training targets (unused, kept for compatibility)
            n_nodes: Number of nodes/regions (expected from dataset)
            val_loss_value: Aggregate validation loss for this trial
            train_start_idx: Start index of training time series in original data
            train_end_idx: End index of training time series in original data
            val_start_idx: Start index of validation time series in original data
            val_end_idx: End index of validation time series in original data
            
        Returns:
            Dict with trial info and list of per-node results
        """
        import torch
        
        # Convert to numpy for easier manipulation
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(train_targets, torch.Tensor):
            train_targets = train_targets.detach().cpu().numpy()
        
        preds = np.array(predictions)
        targs = np.array(targets)
        train_targs = np.array(train_targets) if train_targets is not None else None
        
        # Handle temporal model reshaping: predictions may be (samples*nodes, horizon)
        # We need to detect and reshape back to (samples, nodes, horizon)
        if preds.ndim == 2 and n_nodes > 1:
            total_samples, horizon = preds.shape
            if total_samples % n_nodes == 0:
                n_samples = total_samples // n_nodes
                # Reshape from (samples*nodes, horizon) to (samples, nodes, horizon)
                preds = preds.reshape(n_samples, n_nodes, horizon)
                targs = targs.reshape(n_samples, n_nodes, horizon)
        
        # Now normalize to [samples, nodes, horizon, features]
        if preds.ndim == 2:  # [samples, horizon] - single node case
            preds = preds[:, None, :, None]
            targs = targs[:, None, :, None]
        elif preds.ndim == 3:  # [samples, nodes, horizon]
            preds = preds[:, :, :, None]
            targs = targs[:, :, :, None]
        elif preds.ndim == 4:  # Already [samples, nodes, horizon, features]
            pass
        
        n_samples, n_nodes_actual, horizon, n_features = preds.shape
        
        # Serialize hyperparams (convert non-serializable to strings)
        serializable_hyperparams = {}
        for k, v in hyperparams.items():
            if callable(v):
                serializable_hyperparams[k] = v.__name__ if hasattr(v, '__name__') else str(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                serializable_hyperparams[k] = v
            else:
                serializable_hyperparams[k] = str(v)
        
        # Collect per-node results
        node_results = []
        for node_idx in range(n_nodes_actual):
            # Get node-level predictions and targets
            node_preds = preds[:, node_idx]  # [samples, horizon, features]
            node_targs = targs[:, node_idx]  # [samples, horizon, features]
            
            # Flatten for metric computation
            node_preds_flat = node_preds.flatten()
            node_targs_flat = node_targs.flatten()
            
            # Compute node-level metrics
            residuals = node_preds_flat - node_targs_flat
            node_mse = float(np.mean(residuals ** 2))
            node_mae = float(np.mean(np.abs(residuals)))
            node_rmse = float(np.sqrt(node_mse))
            
            # Compute MAPE safely (avoid division by zero)
            mask = np.abs(node_targs_flat) > 1e-8
            if mask.sum() > 0:
                node_mape = float(np.mean(np.abs(residuals[mask] / node_targs_flat[mask])))
            else:
                node_mape = np.nan
            
            node_result = {
                'node_idx': node_idx,
                'mse': node_mse,
                'mae': node_mae,
                'rmse': node_rmse,
                'mape': node_mape,
                # Time range for referencing original data
                'train_start_idx': train_start_idx,
                'train_end_idx': train_end_idx,
                'val_start_idx': val_start_idx,
                'val_end_idx': val_end_idx,
            }
            node_results.append(node_result)
        
        return {
            'trial_number': trial_number,
            'hyperparams': serializable_hyperparams,
            'val_loss_aggregate': val_loss_value,
            'n_nodes': n_nodes_actual,
            'n_samples': n_samples,
            'horizon': horizon,
            'n_features': n_features,
            'node_results': node_results,
        }

    def _compute_fold_metrics(self, preds, targets, metric_names=None, residual_fn=None):
        """Compute specified metrics from predictions and targets.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
            metric_names: List of metric names to compute (default: ['mse', 'mae', 'rmse'])
            residual_fn: Optional custom residual function (preds, targets) -> residuals
                        Used for custom loss functions where preds may be a dict.
                        
        Returns:
            Dictionary of metric names to values
        """
        if metric_names is None:
            metric_names = ['mse', 'mae', 'rmse']
        
        return metrics.compute_metrics(preds, targets, metric_names, residual_fn)
    
    def _compute_residuals(self, preds, targets, residual_type=None):
        """Compute residuals for conformal prediction.
        
        Args:
            preds: Model predictions (tensor or dict)
            targets: Ground truth targets
            residual_type: How to compute residuals:
                - None or 'abs': Standard absolute residuals ``|preds - targets|``
                - callable: Custom function (preds, targets) -> residuals
                - 'mean': Use preds['mean'] if preds is dict
        
        Returns:
            Absolute residuals tensor
        """
        if callable(residual_type):
            return torch.abs(residual_type(preds, targets))
        elif residual_type == 'mean' and isinstance(preds, dict):
            return torch.abs(preds['mean'] - targets)
        elif isinstance(preds, dict):
            # Default: use 'mean' key if available
            return torch.abs(preds.get('mean', preds.get('prediction', list(preds.values())[0])) - targets)
        else:
            return torch.abs(preds - targets)

    def _compute_fold_conformal(self, val_split, test_preds, test_targets, 
                                 conformal_alpha=0.1, residual_type=None,
                                 process_history=None):
        """Compute conformal prediction intervals for a single fold.
        
        Uses validation set predictions to calibrate the conformal quantile,
        then applies it to the test predictions for uncertainty estimation.
        
        Args:
            val_split: Validation data split (features, targets, etc.)
            test_preds: Test set predictions (should be in original scale if denormalized)
            test_targets: Test set targets (should be in original scale if denormalized)
            conformal_alpha: Significance level (default 0.1 for 90% coverage)
            residual_type: How to compute residuals (None, callable, or 'mean')
            process_history: Optional dict with 'target_mean', 'target_std' for inverse normalization
                           of validation predictions (to match denormalized test scale)
            
        Returns:
            Dictionary with conformal prediction results:
                - conformal_quantile: Calibrated quantile from validation set
                - prediction_lower/upper: Confidence intervals for test predictions
                - coverage: Empirical coverage on test set
                - val_residuals: Validation set residuals used for calibration
        """
        # Get validation predictions using the trained model
        val_preds = self.model.predict(
            feature=val_split['features'],
            graph=val_split['graph'],
            states=val_split['states'],
            dynamic_graph=val_split['dynamic_graph']
        )
        
        # Normalize output format
        if isinstance(val_preds, (tuple, list)):
            val_preds = val_preds[0]
        if not isinstance(val_preds, dict):
            val_targets = val_split['targets']
            if len(val_targets.shape) > len(val_preds.shape):
                val_preds = val_preds.view(val_targets.shape[0], val_targets.shape[1], -1)
            val_preds = val_preds.detach().cpu()
        
        val_targets = val_split['targets'].detach().cpu()
        
        # Keep validation predictions/targets in the SAME scale as the test data
        # (normalized). The conformal quantile must be in the same units as
        # test_preds/test_targets for intervals and coverage to be correct.
        
        # Compute residuals on validation set
        val_residuals = self._compute_residuals(val_preds, val_targets, residual_type).flatten()
        
        # Compute conformal quantile from validation residuals
        n = len(val_residuals)
        if n == 0:
            return {
                'conformal_quantile': float('inf'),
                'prediction_lower': test_preds,
                'prediction_upper': test_preds,
                'coverage': 0.0,
                'val_residuals': val_residuals,
            }
        
        # Conformal quantile formula: ceil((n+1)*(1-alpha)) / n
        q_level = np.ceil((n + 1) * (1 - conformal_alpha)) / n
        conformal_quantile = torch.quantile(val_residuals, min(q_level, 1.0)).item()
        
        # Compute prediction intervals for test set
        if isinstance(test_preds, dict):
            # For dict predictions (e.g., {'mean': ..., 'std': ...})
            pred_values = test_preds.get('mean', test_preds.get('prediction', list(test_preds.values())[0]))
        else:
            pred_values = test_preds
            
        prediction_lower = pred_values - conformal_quantile
        prediction_upper = pred_values + conformal_quantile
        
        # Compute empirical coverage on test set
        within = ((test_targets >= prediction_lower) & (test_targets <= prediction_upper)).float()
        coverage = torch.mean(within).item()
        
        # Compute mean prediction interval width (PIW)
        # Width = upper - lower = 2 * conformal_quantile (symmetric intervals)
        interval_width = torch.mean(prediction_upper - prediction_lower).item()
        
        return {
            'conformal_quantile': conformal_quantile,
            'prediction_lower': prediction_lower,
            'prediction_upper': prediction_upper,
            'coverage': coverage,
            'interval_width': interval_width,
            'val_residuals': val_residuals,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main rolling_train method
    # ─────────────────────────────────────────────────────────────────────────

    def rolling_train(
        self,
        dataset,
        train_size: int,
        test_size: int,
        val_size: int = 0,
        step_size: int = None,
        expanding: bool = True,
        max_folds: int = None,
        train_loss='mse',
        val_loss='mse',
        epochs=100,
        batch_size=32,
        lr=1e-3,
        weight_decay=0,
        patience=50,
        verbose=False,
        model_args={},
        conformal_alpha=0.1,
        regions=None,
        device=None,
        use_optuna=False,
        n_trials=10,
        optimizer_params=None,
        optuna_model_args=None,
        residual_type=None,
        report_metrics=None,
    ):
        """
        Rolling/walk-forward evaluation for time series forecasting.
        
        Trains and evaluates using a rolling window approach to avoid data leakage.
        Optionally uses Optuna for per-fold hyperparameter tuning.
        
        For each fold:
        1. Train model on training window
        2. Compute predictions on validation set → calibrate conformal quantile
        3. Compute predictions on test set → apply conformal intervals
        
        This gives per-fold uncertainty estimation using the validation data that
        is temporally closest to the test data.
        
        Args:
            dataset: Dataset object with timestamps
            train_size: Initial training window size
            test_size: Test window size for each fold
            val_size: Validation window size (required for conformal prediction)
            step_size: Timesteps to advance each fold (default = test_size)
            expanding: If True, training window expands; if False, slides
            max_folds: Maximum number of folds (None = all possible)
            train_loss, val_loss: Loss functions (str like 'mse' or nn.Module)
            epochs, batch_size, lr, weight_decay, patience: Training params
            verbose: Print training progress
            model_args: Fixed model arguments
            conformal_alpha: Significance level for conformal prediction (default 0.1 = 90% coverage)
            regions: Optional list of regions
            device: Training device
            use_optuna: Enable Optuna tuning per fold
            n_trials: Optuna trials per fold
            optimizer_params: Optimizer hyperparameter ranges (can include custom loss)
            optuna_model_args: Model hyperparameter ranges (can include 'lookback')
            residual_type: How to compute residuals for conformal prediction:
                - None: Standard ``|preds - targets|``
                - callable: Custom function (preds, targets) -> residuals
            report_metrics: List of metric names to compute and report (default: ['mse', 'mae', 'rmse'])
                Available metrics: 'mse', 'mae', 'rmse', 'mape', 'r2', 'acc' or custom callables
            
        Returns:
            Dictionary with:
                - fold_results: Per-fold metrics + conformal results
                - aggregate_metrics: Mean/std of specified metrics + coverage
                - conformal_intervals: Per-fold prediction intervals
                - all_predictions/all_targets: Concatenated predictions and targets
        """
        if device is not None:
            self.device = device
        step_size = step_size or test_size
        
        # Set default report metrics
        if report_metrics is None:
            report_metrics = ['mse', 'mae', 'rmse']
        
        # Setup Optuna defaults and lookback tuning
        max_lookback, tuning_lookback = self.lookback, False
        if use_optuna:
            # Use explicit None check: empty {} from config means "no optimizer tuning needed"
            # (e.g., sklearn/statsmodels), while None means "use defaults"
            if optimizer_params is None:
                optimizer_params = {
                    'lr': [1e-4, 1e-2], 'batch_size': [batch_size],
                    'weight_decay': [0, weight_decay] if weight_decay > 0 else [0, 0.1],
                    'epochs': [epochs], 'train_loss': [train_loss], 'val_loss': [val_loss],
                }
            optuna_model_args = optuna_model_args or model_args or {}
            if 'lookback' in optuna_model_args:
                lookback_vals = optuna_model_args['lookback']
                if isinstance(lookback_vals, list) and lookback_vals:
                    max_lookback = max(lookback_vals)
                    tuning_lookback = True
                    _flush_print(f"Lookback tuning enabled: {lookback_vals}")
        
        # Validate window sizes
        ahead = getattr(self, 'ahead', 0)
        min_window = max_lookback + abs(self.horizon) + ahead
        if test_size < min_window:
            _flush_print(f"WARNING: test_size adjusted to {min_window}")
            test_size = min_window
        if val_size > 0 and val_size < min_window:
            _flush_print(f"WARNING: val_size adjusted to {min_window}")
            val_size = min_window
        
        # Print setup info
        _flush_print(f"\n{'='*60}\nRolling Evaluation Setup\n{'='*60}")
        _flush_print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}, Step: {step_size}")
        if use_optuna:
            _flush_print(f"Optuna: {n_trials} trials/fold")
        
        # Initialize collectors
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024
        fold_results, all_predictions, all_targets = [], [], []
        all_conformal_intervals = []  # Per-fold conformal intervals

        # Rolling evaluation loop
        for fold_idx, (train_ds, val_ds, test_ds) in enumerate(dataset.rolling_splits(
            train_size=train_size, test_size=test_size, val_size=val_size,
            step_size=step_size, expanding=expanding, regions=regions
        ), start=1):
            if max_folds and fold_idx > max_folds:
                break
            
            _flush_print(f"\n--- Fold {fold_idx} ---")
            _flush_print(f"Train: {train_ds.n_timesteps} [{train_ds.timestamps[0]} - {train_ds.timestamps[-1]}]")
            if val_ds:
                _flush_print(f"Val:   {val_ds.n_timesteps} [{val_ds.timestamps[0]} - {val_ds.timestamps[-1]}]")
            _flush_print(f"Test:  {test_ds.n_timesteps} [{test_ds.timestamps[0]} - {test_ds.timestamps[-1]}]")
            
            # Apply transforms (compute stats on train, apply to all)
            stats = {}  # Initialize stats to empty dict
            if train_ds.transforms:
                train_ds.apply_transforms(inplace=True)
                stats = train_ds.get_process_history()
                val_ds = self._apply_transforms_with_stats(val_ds, stats) if val_ds else None
                test_ds = self._apply_transforms_with_stats(test_ds, stats)
            
            # Generate splits
            train_split = self._generate_split(train_ds)
            if val_ds and val_ds.n_timesteps >= self.lookback + self.horizon:
                val_split = self._generate_split(val_ds)
            else:
                # Validation set too short; use train split but flag that
                # conformal calibration will be optimistic (residuals from
                # training data are smaller than out-of-sample residuals).
                val_split = self._generate_split(train_ds) if train_ds else None
                _flush_print(f"  WARNING: val_ds too short ({val_ds.n_timesteps if val_ds else 0} < "
                             f"{self.lookback + self.horizon}); conformal calibrated on train data")
            test_split = self._generate_split(test_ds)
            
            # Skip if insufficient samples
            if train_split['features'].shape[0] < batch_size or test_split['features'].shape[0] == 0:
                _flush_print(f"  Skipping fold {fold_idx}: insufficient samples")
                continue
            
            # Determine final hyperparameters
            best_params = None
            if use_optuna:
                best_params = self._run_optuna_tuning(
                    train_ds, val_ds, max_lookback, tuning_lookback,
                    model_args, optimizer_params, optuna_model_args,
                    lr, batch_size, weight_decay, epochs, patience,
                    train_loss, val_loss, n_trials
                )
                
                # Extract final params
                final_lr = best_params.get('lr', lr)
                final_batch_size = best_params.get('batch_size', batch_size)
                final_weight_decay = best_params.get('weight_decay', weight_decay)
                final_epochs = best_params.get('epochs', epochs)
                final_train_loss = best_params.get('train_loss', train_loss)
                final_val_loss = best_params.get('val_loss', val_loss)
                final_lookback = best_params.get('model_lookback', self.lookback)
                
                # Regenerate splits with alignment so all models predict the SAME
                # time windows regardless of lookback. interval = max_lookback - lb
                # shifts the start so that target[0] aligns across all lookback sizes.
                if tuning_lookback:
                    if final_lookback != self.lookback:
                        _flush_print(f"  Using best lookback: {final_lookback}")
                        self.lookback = final_lookback
                    alignment_interval = max_lookback - final_lookback
                    train_split = self._generate_split(train_ds, final_lookback, interval=alignment_interval)
                    val_split = self._generate_split(val_ds, final_lookback, interval=alignment_interval)
                    test_split = self._generate_split(test_ds, final_lookback, interval=alignment_interval)
                
                # Extract model args from best_params (keys have 'model_' prefix)
                final_model_args = {**model_args, **{
                    k: best_params[f'model_{k}']
                    for k in optuna_model_args
                    if f'model_{k}' in best_params
                }}
            else:
                final_lr, final_batch_size, final_weight_decay = lr, batch_size, weight_decay
                final_epochs, final_train_loss, final_val_loss = epochs, train_loss, val_loss
                final_model_args = model_args
            # Train and evaluate
            self._init_model_from_split(train_split, final_model_args)
            try:
                result = self.train_model(
                    train_split=train_split, val_split=val_split, test_split=test_split,
                    train_loss=final_train_loss, val_loss=final_val_loss,
                    epochs=final_epochs, batch_size=final_batch_size,
                    lr=final_lr, weight_decay=final_weight_decay,
                    patience=patience, verbose=verbose,
                    model_args=final_model_args, initialize=False,
                )
                
                if result is None:
                    continue

                # Keep predictions and targets in normalized scale by default
                # This ensures metrics are computed on the same scale as training
                # NOTE: Conformal prediction will still use denormalized scale internally
                preds_for_eval = result['predictions']
                targets_for_eval = result['targets']

                # REMOVED: Automatic denormalization for metrics
                # Metrics will now be computed on normalized scale
                # if stats and 'target_mean' in stats and 'target_std' in stats:
                #     target_mean = stats['target_mean']
                #     target_std = stats['target_std']
                #     preds_for_eval = self._inverse_normalize_target(preds_for_eval, target_mean, target_std)
                #     targets_for_eval = self._inverse_normalize_target(targets_for_eval, target_mean, target_std)
                
                # Compute per-fold conformal quantile from validation predictions
                fold_conformal = self._compute_fold_conformal(
                    val_split=val_split, 
                    test_preds=preds_for_eval,
                    test_targets=targets_for_eval,
                    conformal_alpha=conformal_alpha,
                    residual_type=residual_type,
                    process_history=stats,
                )
                
                # Use custom residual function if provided
                residual_fn = residual_type if callable(residual_type) else None
                fold_metrics = self._compute_fold_metrics(preds_for_eval, targets_for_eval, report_metrics, residual_fn)
                
                # Check for invalid metrics (use first metric as primary check)
                primary_metric = report_metrics[0] if isinstance(report_metrics[0], str) else getattr(report_metrics[0], '__name__', 'metric')
                if primary_metric in fold_metrics and not np.isfinite(fold_metrics[primary_metric]):
                    _flush_print(f"  WARNING: Fold {fold_idx} inf/nan metrics - skipping")
                    continue
                
                fold_result = {
                    'fold': fold_idx,
                    'train_timestamps': (train_ds.timestamps[0], train_ds.timestamps[-1]),
                    'test_timestamps': (test_ds.timestamps[0], test_ds.timestamps[-1]),
                    **fold_metrics,
                    **fold_conformal,  # Include conformal results
                    'n_train': train_split['features'].shape[0],
                    'n_test': test_split['features'].shape[0],
                    'test_split': test_split,
                    'process_history': stats,  # Store normalization stats for inverse transform
                }
                if best_params:
                    # Extract optuna trials (node-level results) if available
                    optuna_trials = best_params.pop('_optuna_trials', [])
                    fold_result['best_params'] = best_params
                    fold_result['optuna_trials'] = optuna_trials
                fold_results.append(fold_result)
                
                all_predictions.append(preds_for_eval)
                all_targets.append(targets_for_eval)
                all_conformal_intervals.append({
                    'lower': fold_conformal['prediction_lower'],
                    'upper': fold_conformal['prediction_upper'],
                    'quantile': fold_conformal['conformal_quantile'],
                })
                
                # Print per-fold metrics
                metric_strs = [f"{k.upper()}: {v:.4f}" for k, v in fold_metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)]
                _flush_print(f"  Test {', '.join(metric_strs)}")
                _flush_print(f"  Conformal quantile: {fold_conformal['conformal_quantile']:.4f}, Coverage: {fold_conformal['coverage']:.1%}, Interval Width: {fold_conformal.get('interval_width', 0):.4f}")
                
            except Exception as e:
                _flush_print(f"  Fold {fold_idx} failed: {e}")
                continue
            # import ipdb; ipdb.set_trace()
        # Aggregate results
        _flush_print(f"\n{'='*60}\nRolling Evaluation Complete ({len(fold_results)} folds)\n{'='*60}")
        
        if not fold_results:
            _flush_print("WARNING: No successful folds!")
            return {'fold_results': [], 'aggregate_metrics': None}
        
        # Aggregate metrics dynamically based on report_metrics
        aggregate_metrics = {'n_folds': len(fold_results)}
        
        for metric_name in report_metrics:
            # Get string key for the metric
            key = metric_name if isinstance(metric_name, str) else getattr(metric_name, '__name__', 'custom_metric')
            
            # Collect values from all folds
            values = [r.get(key) for r in fold_results if key in r and np.isfinite(r.get(key, float('nan')))]
            
            if values:
                aggregate_metrics[f'{key}_mean'] = np.mean(values)
                aggregate_metrics[f'{key}_std'] = np.std(values)
        
        # Print aggregated metrics
        for metric_name in report_metrics:
            key = metric_name if isinstance(metric_name, str) else getattr(metric_name, '__name__', 'custom_metric')
            if f'{key}_mean' in aggregate_metrics:
                _flush_print(f"{key.upper()}: {aggregate_metrics[f'{key}_mean']:.4f} ± {aggregate_metrics[f'{key}_std']:.4f}")
        
        # Aggregate conformal prediction metrics across folds
        if all_conformal_intervals:
            quantiles = [ci['quantile'] for ci in all_conformal_intervals]
            coverages = [r['coverage'] for r in fold_results if 'coverage' in r]
            interval_widths = [r['interval_width'] for r in fold_results if 'interval_width' in r and r['interval_width'] is not None]
            aggregate_metrics['conformal_quantile_mean'] = np.mean(quantiles)
            aggregate_metrics['conformal_quantile_std'] = np.std(quantiles)
            aggregate_metrics['coverage_mean'] = np.mean(coverages)
            aggregate_metrics['coverage_std'] = np.std(coverages)
            if interval_widths:
                aggregate_metrics['interval_width_mean'] = np.mean(interval_widths)
                aggregate_metrics['interval_width_std'] = np.std(interval_widths)
            _flush_print(f"\nConformal ({(1-conformal_alpha)*100:.0f}% target):")
            _flush_print(f"  Quantile: {aggregate_metrics['conformal_quantile_mean']:.4f} ± {aggregate_metrics['conformal_quantile_std']:.4f}")
            _flush_print(f"  Coverage: {aggregate_metrics['coverage_mean']:.1%} ± {aggregate_metrics['coverage_std']:.1%}")
            if interval_widths:
                _flush_print(f"  Interval Width: {aggregate_metrics['interval_width_mean']:.4f} ± {aggregate_metrics['interval_width_std']:.4f}")
        
        # Runtime stats
        total_duration = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        _flush_print(f"\nTotal time: {total_duration:.2f}s, Memory: {end_memory - start_memory:+.2f} MB")
        
        self.rolling_results = {
            'fold_results': fold_results,
            'aggregate_metrics': aggregate_metrics,
            'conformal_alpha': conformal_alpha,
            'all_predictions': all_predictions,
            'all_targets': all_targets,
            'conformal_intervals': all_conformal_intervals,  # Per-fold intervals
            'runtime': total_duration,
        }
        return self.rolling_results
    
    def _inverse_normalize_target(self, data, target_mean, target_std):
        """
        Inverse-transform target data from normalized to original scale.
        
        Handles scalar and per-node normalization stats, and various tensor shapes
        (2D samples×horizon, 3D samples×nodes×horizon, etc.).
        
        Args:
            data: Tensor of predictions or targets in normalized scale
            target_mean: Scalar or array of per-node means
            target_std: Scalar or array of per-node stds
            
        Returns:
            Tensor in original scale
        """
        if data is None:
            return data
        
        # Convert to tensors
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        if isinstance(target_std, (int, float)):
            return data * target_std + target_mean

        if not isinstance(target_std, torch.Tensor):
            target_std = torch.FloatTensor(target_std)
        if not isinstance(target_mean, torch.Tensor):
            target_mean = torch.FloatTensor(target_mean)

        # Check if using scalar (global) or array (per-node) normalization
        if target_mean.numel() == 1:
            # Scalar normalization - broadcast automatically
            return data * target_std.item() + target_mean.item()
        else:
            # Array normalization (per-node) - legacy behavior
            n_nodes = len(target_mean)

            if len(data.shape) > 2 and data.shape[-2] == n_nodes:
                # Shape (samples, nodes, horizon) — per-node denorm
                mean = target_mean.unsqueeze(-1)
                std = target_std.unsqueeze(-1)
                return data * std + mean
            elif len(data.shape) == 2 and data.shape[-1] == n_nodes:
                # Shape (samples, nodes)
                return data * target_std + target_mean
            else:
                # Flattened or incompatible shape — use global average stats
                global_mean = target_mean.mean().item()
                global_std = target_std.mean().item()
                return data * global_std + global_mean

    def _init_model_from_split(self, train_split, model_args={}):
        """Initialize model based on train split shape."""
        if train_split['graph'] is not None or train_split.get('dynamic_graph') is not None:
            graph = train_split['graph']
            if graph is not None:
                if len(graph.shape) == 3:
                    num_nodes = graph.shape[1]
                else:
                    num_nodes = graph.shape[0]
            else:
                num_nodes = train_split['dynamic_graph'].shape[2]
                
            if len(train_split['features'].shape) > 3:
                base_inputs = {
                    "num_nodes": num_nodes,
                    "num_features": train_split['features'].shape[3],
                    "num_timesteps_input": train_split['features'].shape[1],
                    "num_timesteps_output": abs(self.horizon),
                    "device": self.device
                }
            else:
                base_inputs = {
                    "num_nodes": num_nodes,
                    "num_features": train_split['features'].shape[-1],
                    "num_classes": abs(self.horizon),
                    "device": self.device
                }
        else:
            # Use shape[-1] for features: handles both 3D (S, L, F) and
            # 4D (S, L, N, F) temporal data before reshape
            base_inputs = {
                "num_features": train_split['features'].shape[-1],
                "num_timesteps_input": train_split['features'].shape[1],
                "num_timesteps_output": abs(self.horizon),
                "device": self.device
            }

        base_inputs.update(model_args)
        self.model = self.prototype(**base_inputs)
        self.model = self.model.to(self.device)

    def _apply_transforms_with_stats(self, dataset, process_history):
        """
        Apply normalization to a dataset using pre-computed statistics.

        This is used to apply the same normalization (computed on training data)
        to validation and test data, preventing data leakage.

        Parameters
        ----------
        dataset : Dataset
            The dataset to normalize
        process_history : dict
            Dictionary containing normalization statistics:
            - 'feat_mean', 'feat_std': Feature normalization stats (scalars or arrays)
            - 'target_mean', 'target_std': Target normalization stats (scalars or arrays)

        Returns
        -------
        Dataset
            The normalized dataset
        """
        import torch

        # Apply feature normalization
        if 'feat_mean' in process_history and 'feat_std' in process_history:
            feat_mean = process_history['feat_mean']
            feat_std = process_history['feat_std']

            if not isinstance(feat_mean, torch.Tensor):
                feat_mean = torch.tensor(feat_mean)
            if not isinstance(feat_std, torch.Tensor):
                feat_std = torch.tensor(feat_std)

            if dataset.x is not None:
                x = dataset.x
                # Handle both scalar and array normalization statistics
                if feat_mean.numel() == 1:
                    # Scalar normalization - broadcast automatically
                    x = (x - feat_mean.item()) / feat_std.item()
                else:
                    # Per-feature normalization - unsqueeze as needed
                    if len(x.shape) == 2:
                        x = (x - feat_mean) / feat_std
                    elif len(x.shape) == 3:
                        x = (x - feat_mean.unsqueeze(0).unsqueeze(0)) / feat_std.unsqueeze(0).unsqueeze(0)
                    elif len(x.shape) == 4:
                        x = (x - feat_mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)) / feat_std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                x[torch.isnan(x)] = 0
                dataset.x = x

        # Apply target normalization
        if 'target_mean' in process_history and 'target_std' in process_history:
            target_mean = process_history['target_mean']
            target_std = process_history['target_std']

            if not isinstance(target_mean, torch.Tensor):
                target_mean = torch.tensor(target_mean)
            if not isinstance(target_std, torch.Tensor):
                target_std = torch.tensor(target_std)

            if dataset.y is not None:
                y = dataset.y
                # Handle both scalar and array normalization statistics
                if target_mean.numel() == 1:
                    # Scalar normalization - broadcast automatically
                    y = (y - target_mean.item()) / target_std.item()
                else:
                    # Per-feature normalization - unsqueeze as needed
                    if len(y.shape) == 2:
                        y = (y - target_mean) / target_std
                    elif len(y.shape) == 3:
                        y = (y - target_mean.unsqueeze(0).unsqueeze(0)) / target_std.unsqueeze(0).unsqueeze(0)
                    elif len(y.shape) == 4:
                        y = (y - target_mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)) / target_std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                y[torch.isnan(y)] = 0
                dataset.y = y

        dataset.process_history = process_history
        return dataset

    def _reshape_4d_to_temporal(self, split, lookback_dim, features):
        """Reshape 4D spatiotemporal data to 3D temporal format."""
        if split is None or len(split['features'].shape) != 4:
            return split
        split['features'] = split['features'].permute(0, 2, 1, 3).reshape(-1, lookback_dim, features)
        split['targets'] = split['targets'].reshape(-1, split['targets'].shape[-1])
        if split['states'] is not None:
            split['states'] = split['states'].permute(0, 2, 1, 3).reshape(-1, lookback_dim, -1)
        return split

    def _get_num_nodes(self, train_split):
        """Extract number of nodes from graph or dynamic graph."""
        graph = train_split['graph']
        if train_split['dynamic_graph'] is not None:
            return train_split['dynamic_graph'].shape[2]
        if graph is None:
            return None
        return graph.shape[1] if len(graph.shape) == 3 else graph.shape[0]

    def _build_model_inputs(self, train_split, model_args):
        """Build model initialization parameters based on data shape."""
        features = train_split['features']
        has_graph = train_split['graph'] is not None or train_split['dynamic_graph'] is not None
        
        # Use actual data shape for num_timesteps_input, not self.lookback
        # This is critical when Optuna tunes lookback - the data shape reflects
        # the actual lookback used, while self.lookback may be the default/max
        # Shape is (batch, time, ...) for both temporal and spatiotemporal
        actual_lookback = features.shape[1]
        
        if has_graph:
            num_nodes = self._get_num_nodes(train_split)
            if len(features.shape) > 3:  # Spatiotemporal
                return {
                    "num_nodes": num_nodes,
                    "num_features": features.shape[3],
                    "num_timesteps_input": actual_lookback,
                    "num_timesteps_output": abs(self.horizon),
                    "device": self.device,
                    **model_args
                }
            else:  # Spatial
                return {
                    "num_nodes": num_nodes,
                    "num_features": features.shape[-1],
                    "num_classes": abs(self.horizon),
                    "device": self.device,
                    **model_args
                }
        else:  # Temporal
            # Use shape[-1] for features: handles both 3D (S, L, F) and
            # 4D (S, L, N, F) temporal data before reshape
            return {
                "num_features": features.shape[-1],
                "num_timesteps_input": actual_lookback,
                "num_timesteps_output": abs(self.horizon),
                "device": self.device,
                **model_args
            }

    def train_model(self, train_split=None, val_split=None, test_split=None,
                    train_loss='mse', val_loss='mse', epochs=1000, batch_size=10,
                    lr=1e-3, weight_decay=0, initialize=True, verbose=False,
                    patience=100, device=None, pretrained=None, model_args={}):
        """
        Train the model and evaluate on test set.
        
        Args:
            train_split, val_split, test_split: Data splits from generate_dataset()
            train_loss, val_loss: Loss function names
            epochs, batch_size, lr, weight_decay, patience: Training hyperparameters
            initialize: Whether to initialize model weights
            verbose: Print training progress
            device: Training device
            pretrained: Pre-trained model to use instead of initializing
            model_args: Additional model arguments
            
        Returns:
            Dict with loss, predictions, and targets
        """
        if device is not None:
            self.device = device
        if self.prototype is None:
            raise RuntimeError("Model prototype not set. Please load a model first.")
        if None in (train_split, val_split, test_split):
            raise RuntimeError("Missing splits. Provide train_split, val_split, and test_split.")
        # Reshape 4D temporal data BEFORE model init (always needed for correct shapes)
        if len(train_split['features'].shape) == 4 and train_split['graph'] is None:
            samples, lookback_dim, nodes, features = train_split['features'].shape
            for split in (train_split, val_split, test_split):
                self._reshape_4d_to_temporal(split, lookback_dim, features)

        # Initialize or use pretrained model
        if pretrained is not None:
            self.model = pretrained
        elif not initialize and self.model is not None:
            # Skip re-creation when initialize=False and model already exists
            # (model was set up by _init_model_from_split before this call)
            pass
        else:
            base_inputs = self._build_model_inputs(train_split, model_args)
            self.model = self.prototype(**base_inputs)

        self.model = self.model.to(self.device)


        try:
            # Train
            self.model.fit(
                train_input=train_split['features'],
                train_target=train_split['targets'],
                train_states=train_split['states'],
                train_graph=train_split['graph'],
                train_dynamic_graph=train_split['dynamic_graph'],
                val_input=val_split['features'],
                val_target=val_split['targets'],
                val_states=val_split['states'],
                val_graph=val_split['graph'],
                val_dynamic_graph=val_split['dynamic_graph'],
                verbose=verbose, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, epochs=epochs,
                loss=train_loss, initialize=initialize, patience=patience
            )

            # Predict
            out = self.model.predict(
                feature=test_split['features'],
                graph=test_split['graph'],
                states=test_split['states'],
                dynamic_graph=test_split['dynamic_graph']
            )
            # import ipdb; ipdb.set_trace()
            # Normalize output format
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, dict):
                target = test_split['targets']
                if len(target.shape) > len(out.shape):
                    out = out.view(target.shape[0], target.shape[1], -1)
                preds = out.detach().cpu()
            else:
                preds = out
            
            targets = test_split['targets'].detach().cpu()
            loss_fn = metrics.get_loss(val_loss)
            test_loss = loss_fn(preds, targets).item()

            return {"loss": test_loss, "val_loss": test_loss, "predictions": preds, "targets": targets}

        except Exception as e:
            _flush_print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    


    # def _compute_conformal_intervals_general(self, preds, targets, abs_residuals, conformal_quantile=None, 
    #                                         dimension_names=None):
    #     """
    #     Compute conformal prediction intervals and uncertainty quantification (General version).
        
    #     This function is general-purpose and works for various prediction tasks including
    #     time series forecasting, classification, regression, spatial prediction, etc.
        
    #     Args:
    #         preds: Predictions tensor of any shape (samples, ...)
    #         targets: Ground truth tensor matching preds shape
    #         abs_residuals: Absolute residuals tensor matching preds shape
    #         conformal_quantile: Conformal quantile value for base prediction intervals
    #         dimension_names: Optional list of dimension names (e.g., ['samples', 'time', 'features'])
    #                        If None, dimensions are named generically as 'dim_0', 'dim_1', etc.
            
    #     Returns:
    #         Dictionary with conformal prediction results including:
    #             - multi_level_quantiles: Quantiles at different confidence levels
    #             - dimension_quantiles: Per-dimension quantiles (e.g., per time step, per feature)
    #             - adaptive_quantiles: Sample-specific adaptive intervals
    #             - coverage metrics
    #     """
    #     if conformal_quantile is None:
    #         print("\n--- Conformal Prediction ---")
    #         print("Warning: No conformal quantile available. Run rolling_train first or provide conformal_quantile.")
    #         return {}
        
    #     results = {}
        
    #     # Get per-fold conformal intervals if available
    #     conformal_alpha = 0.1
    #     calib_residuals = None
        
    #     if hasattr(self, 'rolling_results'):
    #         conformal_alpha = self.rolling_results.get('conformal_alpha', 0.1)
    #         # Collect validation residuals from all folds for multi-level analysis
    #         fold_results = self.rolling_results.get('fold_results', [])
    #         val_residuals_list = [r.get('val_residuals') for r in fold_results if 'val_residuals' in r]
    #         if val_residuals_list:
    #             calib_residuals = torch.cat(val_residuals_list, dim=0)
    #     elif hasattr(self, 'cv_results') and 'calibration_residuals' in self.cv_results:
    #         calib_residuals = self.cv_results['calibration_residuals']
    #         conformal_alpha = self.cv_results.get('conformal_alpha', 0.1)
        
    #     if calib_residuals is None:
    #         # Use provided residuals if no stored ones
    #         calib_residuals = abs_residuals.flatten()
        
    #     # --- Multi-level Quantiles ---
    #     # Compute quantiles at different confidence levels
    #     quantile_levels = [0.5, 0.75, 0.9, 0.95, 0.99]
    #     sample_quantiles = {}

    #     # Subsample calibration residuals if too large (same as done earlier)
    #     max_samples = 100000
    #     if len(calib_residuals) > max_samples:
    #         indices = torch.randperm(len(calib_residuals))[:max_samples]
    #         calib_residuals_sampled = calib_residuals[indices]
    #     else:
    #         calib_residuals_sampled = calib_residuals

    #     for q_level in quantile_levels:
    #         q_value = np.ceil((len(calib_residuals_sampled) + 1) * (1 - (1 - q_level))) / len(calib_residuals_sampled)
    #         q = torch.quantile(calib_residuals_sampled, q_value)
    #         sample_quantiles[f'quantile_{int(q_level*100)}'] = q.item()
    #     results['multi_level_quantiles'] = sample_quantiles
        
    #     # --- Dimension-wise Analysis ---
    #     # Analyze residuals along each dimension (e.g., time steps, features, spatial locations)
    #     pred_shape = preds.shape
        
    #     if len(pred_shape) > 1:
    #         # Set default dimension names if not provided
    #         if dimension_names is None:
    #             dimension_names = [f'dim_{i}' for i in range(len(pred_shape))]
            
    #         # Analyze each non-sample dimension
    #         for dim_idx in range(1, len(pred_shape)):
    #             dim_name = dimension_names[dim_idx] if dim_idx < len(dimension_names) else f'dim_{dim_idx}'
                
    #             # Reshape residuals to isolate this dimension
    #             # Shape: (n_samples, dim_size)
    #             reshaped_residuals = abs_residuals.reshape(pred_shape[0], -1, pred_shape[dim_idx])
    #             reshaped_residuals = reshaped_residuals.mean(dim=1) if reshaped_residuals.dim() > 2 else reshaped_residuals
                
    #             if reshaped_residuals.dim() == 2 and reshaped_residuals.shape[1] == pred_shape[dim_idx]:
    #                 dim_quantiles = []
    #                 dim_stds = []
    #                 dim_means = []
                    
    #                 for i in range(pred_shape[dim_idx]):
    #                     dim_residuals = reshaped_residuals[:, i]
    #                     q_level = np.ceil((len(dim_residuals) + 1) * (1 - conformal_alpha)) / len(dim_residuals)
    #                     dim_q = torch.quantile(dim_residuals, q_level)
    #                     dim_quantiles.append(dim_q.item())
    #                     dim_stds.append(torch.std(dim_residuals).item())
    #                     dim_means.append(torch.mean(dim_residuals).item())
                    
    #                 results[f'{dim_name}_quantiles'] = dim_quantiles
    #                 results[f'{dim_name}_stds'] = dim_stds
    #                 results[f'{dim_name}_means'] = dim_means
        
    #     # --- Adaptive Sample-wise Quantiles ---
    #     # Compute adaptive intervals based on per-sample uncertainty
    #     if len(pred_shape) > 1:
    #         # Calculate uncertainty for each sample (average residual across all dimensions)
    #         sample_uncertainties = abs_residuals.reshape(pred_shape[0], -1).mean(dim=1)
    #         mean_uncertainty = torch.mean(sample_uncertainties)
            
    #         # Scale conformal quantile based on relative uncertainty
    #         # Clamp to avoid extreme values
    #         relative_uncertainties = sample_uncertainties / (mean_uncertainty + 1e-8)
    #         adaptive_quantiles = conformal_quantile * torch.clamp(relative_uncertainties, 0.5, 2.0)
            
    #         # Reshape adaptive quantiles to match prediction shape
    #         adaptive_shape = [pred_shape[0]] + [1] * (len(pred_shape) - 1)
    #         adaptive_quantiles_reshaped = adaptive_quantiles.reshape(*adaptive_shape)
            
    #         # Compute adaptive prediction intervals
    #         adaptive_lower = preds - adaptive_quantiles_reshaped
    #         adaptive_upper = preds + adaptive_quantiles_reshaped
            
    #         # Calculate coverage
    #         adaptive_within = ((targets >= adaptive_lower) & (targets <= adaptive_upper)).float()
    #         adaptive_coverage = torch.mean(adaptive_within)
            
    #         results.update({
    #             'sample_uncertainties': sample_uncertainties,
    #             'adaptive_quantiles': adaptive_quantiles,
    #             'adaptive_lower': adaptive_lower,
    #             'adaptive_upper': adaptive_upper,
    #             'adaptive_coverage': adaptive_coverage.item(),
    #         })
        
    #     # --- Basic Prediction Intervals ---
    #     # Standard conformal intervals (constant width)
    #     results['prediction_lower'] = preds - conformal_quantile
    #     results['prediction_upper'] = preds + conformal_quantile
    #     basic_within = ((targets >= results['prediction_lower']) & 
    #                    (targets <= results['prediction_upper'])).float()
    #     results['basic_coverage'] = torch.mean(basic_within).item()
        
    #     return results
    
    # def _compute_conformal_intervals_timeseries(self, preds, targets, abs_residuals, conformal_quantile=None,
    #                                            alpha=0.1, decay_factor=0.95, coverage_learning_rate=0.01):
    #     """
    #     Compute adaptive conformal prediction intervals for time series under distribution shift.
        
    #     Implements Algorithm 15: Adaptive Conformal Inference Under Distribution Shift.
    #     This method adapts prediction intervals over time using exponentially decaying weights
    #     to handle non-stationarity and distribution shifts in time series data.
        
    #     Args:
    #         preds: Predictions tensor of shape (n_samples, horizon, ...) 
    #         targets: Ground truth tensor matching preds shape
    #         abs_residuals: Absolute residuals tensor matching preds shape
    #         conformal_quantile: Initial conformal quantile from calibration (if available)
    #         alpha: Miscoverage rate (default 0.1 for 90% coverage)
    #         decay_factor: Decay factor λ for exponential weights (default 0.95)
    #         coverage_learning_rate: Learning rate γ for adaptive coverage (default 0.01)
            
    #     Returns:
    #         Dictionary with adaptive conformal prediction results including:
    #             - adaptive_lower: Lower prediction bounds
    #             - adaptive_upper: Upper prediction bounds
    #             - adaptive_quantiles: Time-varying quantiles
    #             - coverage_per_step: Coverage at each time step
    #             - effective_alpha: Adaptively adjusted alpha values
    #     """
    #     if conformal_quantile is None:
    #         print("\n--- Conformal Prediction ---")
    #         print("Warning: No conformal quantile available. Run rolling_train first or provide conformal_quantile.")
    #         return {}
        
    #     results = {}
        
    #     # Get calibration residuals from validation data
    #     calib_residuals = None
    #     if hasattr(self, 'rolling_results'):
    #         fold_results = self.rolling_results.get('fold_results', [])
    #         val_residuals_list = [r.get('val_residuals') for r in fold_results if 'val_residuals' in r]
    #         if val_residuals_list:
    #             calib_residuals = torch.cat(val_residuals_list, dim=0)
    #     elif hasattr(self, 'cv_results') and 'calibration_residuals' in self.cv_results:
    #         calib_residuals = self.cv_results['calibration_residuals']
        
    #     if calib_residuals is None:
    #         # Fall back to provided residuals
    #         calib_residuals = abs_residuals.flatten()
        
    #     n_cal = len(calib_residuals)
        
    #     # Step 2: Initialize for adaptive procedure
    #     n_test = preds.shape[0]
    #     pred_shape = preds.shape[1:] if len(preds.shape) > 1 else (1,)
        
    #     # Initialize adaptive quantiles and alpha
    #     adaptive_quantiles = torch.zeros(n_test, *pred_shape)
    #     adaptive_lower = torch.zeros_like(preds)
    #     adaptive_upper = torch.zeros_like(preds)
    #     effective_alphas = torch.zeros(n_test)
        
    #     # Step 3: Compute exponentially decaying weights
    #     weights_lower = torch.zeros(n_cal)
    #     weights_upper = torch.zeros(n_cal)
    #     for t in range(n_cal):
    #         weights_lower[t] = decay_factor ** (n_cal - t - 1)
    #         weights_upper[t] = decay_factor ** (n_cal - t - 1)
        
    #     # Normalize weights
    #     weights_lower = weights_lower / weights_lower.sum()
    #     weights_upper = weights_upper / weights_upper.sum()
        
    #     # Step 4: Adaptive prediction intervals for test points
    #     current_alpha = alpha
        
    #     for t in range(n_test):
    #         # Use prediction as both lower and upper bound estimates (no quantile model)
    #         pred_lower = preds[t]
    #         pred_upper = preds[t]
            
    #         # Compute weighted quantiles from calibration residuals
    #         # Sort residuals and compute cumulative weights
    #         sorted_residuals, sort_indices = torch.sort(calib_residuals)
    #         cum_weights_lower = torch.cumsum(weights_lower[sort_indices], dim=0)
    #         cum_weights_upper = torch.cumsum(weights_upper[sort_indices], dim=0)
            
    #         # Find quantile corresponding to (1 - alpha)
    #         q_lower_idx = torch.searchsorted(cum_weights_lower, 1 - current_alpha)
    #         q_upper_idx = torch.searchsorted(cum_weights_upper, 1 - current_alpha)
            
    #         q_lower_idx = min(q_lower_idx, len(sorted_residuals) - 1)
    #         q_upper_idx = min(q_upper_idx, len(sorted_residuals) - 1)
            
    #         quantile_lower = sorted_residuals[q_lower_idx]
    #         quantile_upper = sorted_residuals[q_upper_idx]
            
    #         # Average the two quantiles
    #         adaptive_quantile = (quantile_lower + quantile_upper) / 2
            
    #         # Store quantile and compute intervals
    #         adaptive_quantiles[t] = adaptive_quantile
    #         adaptive_lower[t] = pred_lower - adaptive_quantile
    #         adaptive_upper[t] = pred_upper + adaptive_quantile
    #         effective_alphas[t] = current_alpha
            
    #         # Step 5: Update alpha adaptively based on coverage
    #         # Check if target is within interval
    #         if t < len(targets):
    #             within_interval = ((targets[t] >= adaptive_lower[t]) & 
    #                               (targets[t] <= adaptive_upper[t])).float()
    #             coverage_error = within_interval.mean() - (1 - alpha)
                
    #             # Update alpha with learning rate
    #             current_alpha = current_alpha + coverage_learning_rate * coverage_error.item()
    #             current_alpha = max(0.01, min(0.5, current_alpha))  # Clamp alpha
        
    #     # Compute overall coverage metrics
    #     within = ((targets >= adaptive_lower) & (targets <= adaptive_upper)).float()
    #     overall_coverage = torch.mean(within).item()
        
    #     # Compute coverage per time step (if horizon > 1)
    #     coverage_per_step = None
    #     if len(pred_shape) > 0 and pred_shape[0] > 1:
    #         coverage_per_step = []
    #         for h in range(pred_shape[0]):
    #             step_within = within[:, h].mean().item() if len(within.shape) > 1 else within.mean().item()
    #             coverage_per_step.append(step_within)
        
    #     results.update({
    #         'method': 'adaptive_conformal_timeseries',
    #         'adaptive_lower': adaptive_lower,
    #         'adaptive_upper': adaptive_upper,
    #         'adaptive_quantiles': adaptive_quantiles,
    #         'effective_alphas': effective_alphas,
    #         'overall_coverage': overall_coverage,
    #         'coverage_per_step': coverage_per_step,
    #         'target_coverage': 1 - alpha,
    #         'decay_factor': decay_factor,
    #         'learning_rate': coverage_learning_rate,
    #     })
        
    #     return results


    # def _compute_conformal_intervals(self, preds, targets, abs_residuals, conformal_quantile=None, 
    #                                 dimension_names=None, method='general', **kwargs):
    #     """
    #     Compute conformal prediction intervals with multiple methods.
        
    #     Args:
    #         preds: Predictions tensor
    #         targets: Ground truth tensor
    #         abs_residuals: Absolute residuals tensor
    #         conformal_quantile: Conformal quantile from calibration
    #         dimension_names: Optional dimension names for general method
    #         method: 'general' or 'timeseries' (default: 'general')
    #         **kwargs: Additional arguments for specific methods
    #             For 'timeseries': alpha, decay_factor, coverage_learning_rate
        
    #     Returns:
    #         Dictionary with conformal prediction results
    #     """
    #     if method == 'timeseries':
    #         return self._compute_conformal_intervals_timeseries(
    #             preds, targets, abs_residuals, conformal_quantile, **kwargs
    #         )
    #     else:
    #         return self._compute_conformal_intervals_general(
    #             preds, targets, abs_residuals, conformal_quantile, dimension_names
    #         )
