import torch
import numpy as np

from ..visualize import plot_series
from ..utils import utils, metrics  
from .base import BaseTask

class Forecast(BaseTask):
    """
    The Forecast class extends the BaseTask class, focusing on the training and evaluation of forecast models for 
    time-series prediction tasks. It includes functionalities specific to handling time-series data, especially 
    in settings that involve spatial-temporal dynamics. The class supports model initialization, training, evaluation, 
    and preprocessing, facilitating the application of various neural network architectures and configurations.
    """
    def __init__(self, prototype = None, model = None, dataset = None, lookback = None, horizon = None, ahead=0, device = 'cpu'):
        super().__init__(prototype, model, dataset, lookback, horizon, ahead, device)
        self.feat_mean = 0
        self.feat_std = 1
        self.device = device

    def evaluate_model(self,
                    model=None,
                    dataset=None,
                    process_history=None,
                    use_conformal=True,
                    conformal_quantile=None,
                    inverse_normalize=False,
                    residue_func=None
                    ):
        """
        Evaluate the trained model and compute metrics with adaptive conformal prediction intervals.
        
        Args:
            model: Model to evaluate (uses self.model if None)
            dataset: Dataset dictionary with 'features', 'targets', 'graph', etc.
            process_history: Dict with 'target_mean', 'target_std' for inverse normalization
            use_conformal: Whether to compute conformal prediction intervals
            conformal_quantile: Conformal quantile (uses self.conformal_quantile if None)
            inverse_normalize: Whether to inverse normalize predictions/targets
            residue_func: Custom residual function for misaligned predictions
            
        Returns:
            Dictionary containing:
            - Point metrics: mse, mae, rmse, mape, r2
            - Residual statistics: residual_mean, residual_std
            - Raw outputs: predictions, targets, residuals
            - Conformal results: adaptive_lower, adaptive_upper, coverage stats (if use_conformal=True)
        """
        if model is None:
            if not hasattr(self, "model"):
                raise RuntimeError("model not exists, please load model first!")
            model = self.model

        targets = dataset['targets'].to(self.device)
        features = dataset['features'].to(self.device)
        graph = dataset['graph'].to(self.device) if dataset['graph'] is not None else None
        dynamic_graph = dataset['dynamic_graph'].to(self.device) if dataset['dynamic_graph'] is not None else None
        states = dataset['states'].to(self.device) if dataset['states'] is not None else None
        with torch.no_grad():
            out = model.predict(feature=features, 
                                    graph=graph, 
                                    states=states, 
                                    dynamic_graph=dynamic_graph
                                    )
            
        if type(out) is tuple:
            out = out[0]

        preds = out.detach().cpu()
        targets = targets.detach().cpu()

        pre_preds = preds
        pre_targets = targets
        
        # Calculate absolute residuals for conformal prediction
        # Use residue_func first when provided (handles misaligned preds/targets)
        if residue_func is not None:
            pre_abs_residuals = residue_func(preds, targets)
        else:
            try:
                pre_abs_residuals = torch.abs(pre_preds - pre_targets)
            except Exception as e:
                print(f"Error computing residuals: {e}; Consider using custom residue_func.")
                pre_abs_residuals = None

        if inverse_normalize:
            # Get node indices if available (for flattened temporal data)
            node_indices = dataset.get('node_indices')
            # Apply inverse normalization at the end for final outputs
            preds = self.inverse_norm(preds, process_history['target_mean'], process_history['target_std'], node_indices)
            targets = self.inverse_norm(targets, process_history['target_mean'], process_history['target_std'], node_indices)
            
        # Calculate residuals for metrics
        # Use residue_func if provided (for potentially misaligned data)
        if residue_func is not None:
            # If inverse_normalize was applied, use residue_func on normalized data
            if inverse_normalize:
                abs_residuals = residue_func(preds, targets)
            else:
                # Reuse pre_abs_residuals if no normalization change
                abs_residuals = pre_abs_residuals
            residuals = abs_residuals  # For residue_func, residuals are absolute by nature
            squared_residuals = abs_residuals ** 2
        else:
            # Standard calculation when data is aligned
            try:
                residuals = preds - targets
                abs_residuals = torch.abs(residuals)
                squared_residuals = residuals ** 2
            except Exception as e:
                print(f"Error computing residuals for metrics: {e}; Consider using custom residue_func.")
                raise
        
        # Basic metrics
        mse = torch.mean(squared_residuals)
        mae = torch.mean(abs_residuals)
        rmse = torch.sqrt(mse)
        
        # MAPE calculation (skip if using residue_func as data may be misaligned)
        if residue_func is not None:
            mape = torch.tensor(float('nan'))  # MAPE not meaningful for misaligned data
        else:
            mape = torch.mean(torch.abs((targets - preds) / (targets + 1e-8))) * 100
        
        # R-squared
        ss_res = torch.sum(squared_residuals)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Additional metrics
        median_ae = torch.median(abs_residuals)
        max_error = torch.max(abs_residuals)
        residual_mean = torch.mean(residuals)
        residual_std = torch.std(residuals)
        residual_median = torch.median(residuals)
        
        # Conformal prediction and uncertainty quantification
        conformal_results = {}
        if use_conformal and conformal_quantile is not None:
            raise NotImplementedError(
                "Passing an explicit conformal_quantile to evaluate_model is not "
                "supported in this release. Use rolling_train(), which calibrates a "
                "conformal quantile per fold and returns it in "
                "result['fold_results'][i]['conformal_quantile'], or call the "
                "strategies in epilearn.utils.uncertainty (static_conformal, "
                "compute_aci, locally_weighted_conformal) directly on saved "
                "residuals. Leave conformal_quantile=None to evaluate without it."
            )
            # For time series forecasting, use dimension names for better interpretability
            dimension_names = ['samples', 'time_steps']
            conformal_results = self._compute_conformal_intervals(
                pre_preds, 
                pre_targets, 
                pre_abs_residuals, 
                conformal_quantile=conformal_quantile,
                dimension_names=dimension_names
            )

            if inverse_normalize and conformal_results and 'adaptive_lower' in conformal_results:
                conformal_results['adaptive_lower'] = self.inverse_norm(
                    conformal_results['adaptive_lower'], 
                    process_history['target_mean'], 
                    process_history['target_std']
                )
                conformal_results['adaptive_upper'] = self.inverse_norm(
                    conformal_results['adaptive_upper'], 
                    process_history['target_mean'], 
                    process_history['target_std']
                )

        # Print evaluation summary
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"\n--- Point Metrics ---")
        print(f"MSE:              {mse.item():.6f}")
        print(f"MAE:              {mae.item():.6f}")
        print(f"RMSE:             {rmse.item():.6f}")
        if not torch.isnan(mape):
            print(f"MAPE:             {mape.item():.2f}%")
        print(f"R²:               {r2.item():.6f}")
        print(f"Median AE:        {median_ae.item():.6f}")
        print(f"Max Error:        {max_error.item():.6f}")
        print(f"\n--- Residual Statistics ---")
        print(f"Mean:             {residual_mean.item():.6f}")
        print(f"Std Dev:          {residual_std.item():.6f}")
        
        if use_conformal and conformal_results:
            print(f"\n--- Conformal Prediction ---")
            if 'coverage' in conformal_results:
                print(f"Coverage:         {conformal_results['coverage']*100:.1f}%")
            if 'adaptive_quantiles' in conformal_results:
                aq = conformal_results['adaptive_quantiles']
                print(f"Adaptive Quantile: mean={aq.mean().item():.4f}, min={aq.min().item():.4f}, max={aq.max().item():.4f}")
        print(f"{'='*60}")
        
        # Compile results
        results = {
            'mse': mse.item(),
            'mae': mae.item(),
            'rmse': rmse.item(),
            'mape': mape.item() if not torch.isnan(mape) else None,
            'r2': r2.item(),
            'median_ae': median_ae.item(),
            'max_error': max_error.item(),
            'residual_mean': residual_mean.item(),
            'residual_std': residual_std.item(),
            'predictions': preds,
            'targets': targets,
            'residuals': residuals,
        }
        
        # Add conformal results
        if conformal_results:
            results.update(conformal_results)
        
        return results
    

    def inverse_norm(self, data, mean, std, node_indices=None):
        """
        Apply inverse normalization to data.
        
        Handles multiple cases:
        - Global normalization (scalar mean/std)
        - Per-node normalization with data that still has node dimension
        - Per-node normalization with flattened data and node_indices
        - Per-node normalization with flattened data without node_indices (uses global average)
        
        Args:
            data: Tensor to denormalize
            mean: Either scalar or array of means (one per node)
            std: Either scalar or array of stds (one per node)
            node_indices: Optional tensor of node indices for flattened data
        
        Returns:
            Denormalized tensor
        """
        if isinstance(std, (int, float)):
            return data * std + mean

        # Convert to tensor if needed
        if not isinstance(std, torch.Tensor):
            std = torch.FloatTensor(std)
        if not isinstance(mean, torch.Tensor):
            mean = torch.FloatTensor(mean)

        # Check if using scalar (global) or array (per-node) normalization
        if mean.numel() == 1:
            # Scalar normalization - broadcast automatically
            return data * std.item() + mean.item()
        else:
            # Array normalization (per-node) - legacy behavior
            n_nodes = len(mean)

            if len(data.shape) > 2:
                # Original shape preserved: (samples, nodes, horizon) or (samples, lookback, nodes, features)
                std = std.unsqueeze(-1)
                mean = mean.unsqueeze(-1)
                return data * std + mean
            elif len(data.shape) == 2 and data.shape[-1] == n_nodes:
                # Shape is (samples, nodes) - can apply per-node
                return data * std + mean
            elif node_indices is not None:
                # Flattened data with node indices: (samples*nodes, horizon) or (samples*nodes,)
                # Use per-node statistics indexed by node_indices
                node_mean = mean[node_indices]  # Shape: (samples*nodes,)
                node_std = std[node_indices]    # Shape: (samples*nodes,)

                # Expand to match data shape if needed
                if len(data.shape) == 2:
                    node_mean = node_mean.unsqueeze(-1)  # (samples*nodes, 1)
                    node_std = node_std.unsqueeze(-1)    # (samples*nodes, 1)

                return data * node_std + node_mean
            else:
                # Flattened data without node indices: (samples*nodes, horizon) or (samples, horizon)
                # Per-node normalization cannot be correctly applied because we don't know
                # which samples correspond to which nodes.
                # Use the global average of per-node statistics as approximation.
                global_mean = mean.mean().item()
                global_std = std.mean().item()
                return data * global_std + global_mean
    
    def plot_preds(self, eval_results, n_show=None, figsize=(15, 7), 
                   save_path=None, backend='matplotlib', 
                   region_idx=0, horizon_idx=-1, interactive=False):
        """
        Plot predictions with adaptive uncertainty intervals in a clean white style.
        
        Args:
            eval_results: Dictionary returned from evaluate_model containing predictions, targets, and uncertainty estimates
            n_show: Number of time samples to display (None plots all samples, default: None)
            figsize: Figure size as (width, height) tuple for matplotlib (default: (15, 7))
            save_path: Optional path to save the figure (e.g., 'plot.png' or 'plot.html')
            backend: Plotting backend - 'matplotlib' or 'plotly' (default: 'matplotlib')
            region_idx: Index of region to plot (default: 0)
            horizon_idx: Index of horizon step to plot (default: -1, last step)
            interactive: Whether to make plotly plots interactive (default: False)
            
        Returns:
            For matplotlib: fig, ax
            For plotly: fig
        """
        if backend == 'plotly':
            return self._plot_with_plotly(eval_results, n_show, save_path, region_idx, horizon_idx, interactive)
        else:
            return self._plot_with_matplotlib(eval_results, n_show, figsize, save_path, region_idx, horizon_idx)
    
    def _plot_with_matplotlib(self, eval_results, n_show, figsize, save_path, region_idx, horizon_idx):
        """Internal method for matplotlib plotting."""
        import matplotlib.pyplot as plt
        
        # Extract predictions and targets from evaluation results
        # Shape: (time, regions, horizon)
        preds = eval_results['predictions'].numpy()
        targets = eval_results['targets'].numpy()
        
        print(f"Data shape: {preds.shape} (time, regions, horizon)")
        print(f"Plotting region {region_idx}, horizon step {horizon_idx}")
        
        # Extract specific region and horizon
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, axis=1)
            targets = targets.reshape(preds.shape)
        preds_plot = preds[:, region_idx, horizon_idx]
        targets_plot = targets[:, region_idx, horizon_idx]
        
        # Determine number of samples to show (all by default)
        if n_show is None:
            n_show = len(preds_plot)
        else:
            n_show = min(n_show, len(preds_plot))
        
        x = np.arange(n_show)
        
        print(f"Plotting {n_show} time samples")
        
        # Set white background style
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Extract adaptive uncertainty intervals
        if 'adaptive_lower' in eval_results:
            adaptive_lower = eval_results['adaptive_lower'].numpy()
            adaptive_upper = eval_results['adaptive_upper'].numpy()
            
            # Extract for specific region and horizon
            if len(adaptive_lower.shape) == 2:
                adaptive_lower = np.expand_dims(adaptive_lower, axis=1)
                adaptive_upper = np.expand_dims(adaptive_upper, axis=1)
            adaptive_lower_plot = adaptive_lower[:, region_idx, horizon_idx]
            adaptive_upper_plot = adaptive_upper[:, region_idx, horizon_idx]
            
            # Plot adaptive interval
            ax.fill_between(x, adaptive_lower_plot[:n_show], adaptive_upper_plot[:n_show], 
                             alpha=0.25, color='#3498db', label='Adaptive Prediction Interval', zorder=1)
        
        # Plot true values and predictions on top
        ax.plot(x, targets_plot[:n_show], 'o-', label='True Values', color='#2c3e50', 
                 alpha=0.9, markersize=7, linewidth=2.5, zorder=3)
        ax.plot(x, preds_plot[:n_show], 's-', label='Predictions', color='#e74c3c', 
                 alpha=0.8, markersize=6, linewidth=2, zorder=2)
        
        ax.set_xlabel('Time Index', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Value', fontsize=13, fontweight='bold', color='#2c3e50')
        
        title = f'Predictions with Adaptive Uncertainty (Region {region_idx}, Horizon {horizon_idx})'
        ax.set_title(title, fontsize=15, fontweight='bold', color='#2c3e50', pad=20)
        
        # Customize legend
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, 
                  loc='best', edgecolor='#bdc3c7', facecolor='white')
        
        # Customize grid
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#95a5a6')
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_edgecolor('#bdc3c7')
            spine.set_linewidth(1.2)
        
        # Customize ticks
        ax.tick_params(colors='#2c3e50', labelsize=10)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nFigure saved to: {save_path}")
        
        # Calculate and print coverage statistics
        if 'adaptive_lower' in eval_results:
            within_adaptive = np.sum((targets_plot[:n_show] >= adaptive_lower_plot[:n_show]) & 
                                    (targets_plot[:n_show] <= adaptive_upper_plot[:n_show]))
            coverage_pct = within_adaptive / n_show * 100
            print(f"\n{'='*60}")
            print(f"Coverage Statistics")
            print(f"{'='*60}")
            print(f"Adaptive Coverage: {within_adaptive}/{n_show} = {coverage_pct:.2f}%")
            print(f"{'='*60}")
        
        return fig, ax
    
    def _plot_with_plotly(self, eval_results, n_show, save_path, region_idx, horizon_idx, interactive):
        """Internal method for plotly plotting."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for interactive plotting. Install with: pip install plotly")
        
        # Extract predictions and targets from evaluation results
        # Shape: (time, regions, horizon)
        preds = eval_results['predictions'].numpy()
        targets = eval_results['targets'].numpy()
        
        print(f"Data shape: {preds.shape} (time, regions, horizon)")
        print(f"Plotting region {region_idx}, horizon step {horizon_idx}")
        
        # Extract specific region and horizon
        preds_plot = preds[:, region_idx, horizon_idx]
        targets_plot = targets[:, region_idx, horizon_idx]
        
        # Determine number of samples to show (all by default)
        if n_show is None:
            n_show = len(preds_plot)
        else:
            n_show = min(n_show, len(preds_plot))
        
        x = np.arange(n_show)
        
        print(f"Plotting {n_show} time samples")
        
        # Create figure
        fig = go.Figure()
        
        # Extract adaptive uncertainty intervals
        if 'adaptive_lower' in eval_results:
            adaptive_lower = eval_results['adaptive_lower'].numpy()
            adaptive_upper = eval_results['adaptive_upper'].numpy()
            
            # Extract for specific region and horizon
            adaptive_lower_plot = adaptive_lower[:, region_idx, horizon_idx]
            adaptive_upper_plot = adaptive_upper[:, region_idx, horizon_idx]
            
            # Add adaptive interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([adaptive_upper_plot[:n_show], adaptive_lower_plot[:n_show][::-1]]),
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.25)',
                line=dict(color='rgba(52, 152, 219, 0)'),
                name='Adaptive Prediction Interval',
                hoverinfo='skip',
                showlegend=True
            ))
        
        # Add true values
        fig.add_trace(go.Scatter(
            x=x,
            y=targets_plot[:n_show],
            mode='lines+markers',
            name='True Values',
            line=dict(color='#2c3e50', width=2.5),
            marker=dict(size=7, symbol='circle'),
            hovertemplate='<b>True Value</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=x,
            y=preds_plot[:n_show],
            mode='lines+markers',
            name='Predictions',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=6, symbol='square'),
            hovertemplate='<b>Prediction</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ))
        
        # Update layout
        title = f'Predictions with Adaptive Uncertainty (Region {region_idx}, Horizon {horizon_idx})'
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='#2c3e50', family='Arial, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Time Index',
                titlefont=dict(size=14, color='#2c3e50'),
                gridcolor='rgba(149, 165, 166, 0.25)',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title='Value',
                titlefont=dict(size=14, color='#2c3e50'),
                gridcolor='rgba(149, 165, 166, 0.25)',
                showgrid=True,
                zeroline=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified' if interactive else 'closest',
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                font=dict(size=11)
            ),
            width=1200,
            height=600
        )
        
        # Save figure if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
                print(f"\nInteractive figure saved to: {save_path}")
            else:
                fig.write_image(save_path, width=1200, height=600)
                print(f"\nFigure saved to: {save_path}")
        
        if interactive:
            fig.show()
        
        # Calculate and print coverage statistics
        if 'adaptive_lower' in eval_results:
            within_adaptive = np.sum((targets_plot[:n_show] >= adaptive_lower_plot[:n_show]) & 
                                    (targets_plot[:n_show] <= adaptive_upper_plot[:n_show]))
            coverage_pct = within_adaptive / n_show * 100
            print(f"\n{'='*60}")
            print(f"Coverage Statistics")
            print(f"{'='*60}")
            print(f"Adaptive Coverage: {within_adaptive}/{n_show} = {coverage_pct:.2f}%")
            print(f"{'='*60}")
        
        return fig


