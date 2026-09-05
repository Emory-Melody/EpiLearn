import torch
import numpy as np
import optuna
from optuna.samplers import TPESampler

from ..visualize import plot_series
from ..utils import utils, metrics  
from .base import BaseTask

class Detection(BaseTask):
    """
    The Detection class extends the BaseTask class, focusing on the training and evaluation of source detection models
    for spatial-temporal prediction tasks. It predicts the source node given future spatial-temporal observations.
    The class supports model initialization, training, evaluation with conformal prediction, and comprehensive metrics
    for classification tasks including accuracy, precision, recall, and F1 score.
    """
    def __init__(self, prototype = None, model = None, dataset = None, lookback = None, horizon = None, ahead=0, device = 'cpu'):
        super().__init__(prototype, model, dataset, lookback, horizon, ahead, device)
        self.device = device


    def evaluate_model(self,
                    model=None,
                    dataset=None,
                    process_history=None,
                    use_conformal=True,
                    conformal_quantile=None,
                    n_bootstrap=100,
                    compute_bootstrap_ci=True,
                    ):
        """
        Comprehensive evaluation of the trained detection model with uncertainty quantification.
        
        Args:
            model: Model to evaluate (uses self.model if None)
            dataset: Dataset dict with 'features', 'targets', 'graph', etc.
            process_history: Processing history (not used for detection, kept for API consistency)
            use_conformal: Whether to compute conformal prediction intervals
            conformal_quantile: Conformal quantile (uses self.conformal_quantile if None)
            n_bootstrap: Number of bootstrap samples for uncertainty estimation
            compute_bootstrap_ci: Whether to compute bootstrap confidence intervals
            
        Returns:
            Dictionary containing comprehensive evaluation metrics and uncertainty estimates
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

        # Get class probabilities and predictions
        probs = torch.softmax(out, dim=-1).detach().cpu()  # Shape: (batch, num_nodes, num_classes)
        preds = probs.argmax(dim=-1)  # Shape: (batch, num_nodes)
        targets = targets.detach().cpu()

        # Calculate classification metrics
        # Flatten for overall metrics
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Accuracy
        correct = (preds_flat == targets_flat).float()
        accuracy = torch.mean(correct)
        
        # Per-class metrics
        num_classes = probs.shape[-1]
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for c in range(num_classes):
            # True positives, false positives, false negatives
            tp = ((preds_flat == c) & (targets_flat == c)).float().sum()
            fp = ((preds_flat == c) & (targets_flat != c)).float().sum()
            fn = ((preds_flat != c) & (targets_flat == c)).float().sum()
            
            # Precision, recall, F1
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            precision_per_class.append(precision.item())
            recall_per_class.append(recall.item())
            f1_per_class.append(f1.item())
        
        # Macro-averaged metrics
        macro_precision = np.mean(precision_per_class)
        macro_recall = np.mean(recall_per_class)
        macro_f1 = np.mean(f1_per_class)
        
        # Confidence-based metrics
        max_probs = probs.max(dim=-1)[0]  # Maximum probability for each prediction
        mean_confidence = torch.mean(max_probs)
        
        # Bootstrap confidence intervals
        bootstrap_results = {}
        if compute_bootstrap_ci and n_bootstrap > 0:
            print(f"\n--- Bootstrap Confidence Intervals (n={n_bootstrap}) ---")
            print("Computing bootstrap estimates...")
            
            bootstrap_results = self._compute_bootstrap_ci_detection(
                preds_flat, 
                targets_flat,  
                n_bootstrap=n_bootstrap,
                num_classes=num_classes
            )
            print(f"Accuracy 95% CI:  [{bootstrap_results['accuracy_ci'][0]:.4f}, {bootstrap_results['accuracy_ci'][1]:.4f}]")
            print(f"F1 Score 95% CI:  [{bootstrap_results['f1_ci'][0]:.4f}, {bootstrap_results['f1_ci'][1]:.4f}]")
        
        # Print evaluation summary
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE MODEL EVALUATION - SOURCE DETECTION")
        print(f"{'='*60}")
        print(f"\n--- Classification Metrics ---")
        print(f"Accuracy:         {accuracy.item():.4f}")
        print(f"Macro Precision:  {macro_precision:.4f}")
        print(f"Macro Recall:     {macro_recall:.4f}")
        print(f"Macro F1:         {macro_f1:.4f}")
        print(f"Mean Confidence:  {mean_confidence.item():.4f}")
        
        print(f"\n--- Per-Class Metrics ---")
        for c in range(num_classes):
            print(f"Class {c} - Precision: {precision_per_class[c]:.4f}, "
                  f"Recall: {recall_per_class[c]:.4f}, F1: {f1_per_class[c]:.4f}")
        
        print(f"\n{'='*60}")
        
        # Compile results
        results = {
            'accuracy': accuracy.item(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'mean_confidence': mean_confidence.item(),
            'predictions': preds,
            'targets': targets,
            'probabilities': probs,
        }
        
        # Add bootstrap results
        if bootstrap_results:
            results.update({
                'bootstrap_accuracy_ci': bootstrap_results['accuracy_ci'],
                'bootstrap_precision_ci': bootstrap_results['precision_ci'],
                'bootstrap_recall_ci': bootstrap_results['recall_ci'],
                'bootstrap_f1_ci': bootstrap_results['f1_ci'],
                'bootstrap_accuracies': bootstrap_results['accuracies'],
                'bootstrap_precisions': bootstrap_results['precisions'],
                'bootstrap_recalls': bootstrap_results['recalls'],
                'bootstrap_f1s': bootstrap_results['f1s'],
            })
        
        return results

    def _compute_bootstrap_ci_detection(self, preds, targets, n_bootstrap=100, num_classes=2):
        """
        Compute bootstrap confidence intervals for detection evaluation metrics.
        
        Args:
            preds: Predictions tensor (flattened)
            targets: Targets tensor (flattened)
            n_bootstrap: Number of bootstrap samples
            num_classes: Number of classes
            
        Returns:
            Dictionary with bootstrap results
        """
        n_samples = len(preds)
        bootstrap_accuracies = []
        bootstrap_precisions = []
        bootstrap_recalls = []
        bootstrap_f1s = []
        
        for _ in range(n_bootstrap):
            indices = torch.randint(0, n_samples, (n_samples,))
            preds_boot = preds[indices]
            targets_boot = targets[indices]
            
            # Accuracy
            correct = (preds_boot == targets_boot).float()
            accuracy_boot = torch.mean(correct)
            bootstrap_accuracies.append(accuracy_boot.item())
            
            # Macro-averaged metrics
            precision_list = []
            recall_list = []
            f1_list = []
            
            for c in range(num_classes):
                tp = ((preds_boot == c) & (targets_boot == c)).float().sum()
                fp = ((preds_boot == c) & (targets_boot != c)).float().sum()
                fn = ((preds_boot != c) & (targets_boot == c)).float().sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                precision_list.append(precision.item())
                recall_list.append(recall.item())
                f1_list.append(f1.item())
            
            bootstrap_precisions.append(np.mean(precision_list))
            bootstrap_recalls.append(np.mean(recall_list))
            bootstrap_f1s.append(np.mean(f1_list))
        
        # Calculate 95% confidence intervals
        return {
            'accuracy_ci': (np.percentile(bootstrap_accuracies, 2.5), np.percentile(bootstrap_accuracies, 97.5)),
            'precision_ci': (np.percentile(bootstrap_precisions, 2.5), np.percentile(bootstrap_precisions, 97.5)),
            'recall_ci': (np.percentile(bootstrap_recalls, 2.5), np.percentile(bootstrap_recalls, 97.5)),
            'f1_ci': (np.percentile(bootstrap_f1s, 2.5), np.percentile(bootstrap_f1s, 97.5)),
            'accuracies': bootstrap_accuracies,
            'precisions': bootstrap_precisions,
            'recalls': bootstrap_recalls,
            'f1s': bootstrap_f1s,
        }
    
    def plot_preds(self, eval_results, n_show=None, figsize=(15, 7), 
                   save_path=None, backend='matplotlib', 
                   sample_idx=0, interactive=False):
        """
        Plot source detection predictions with confidence scores.
        
        Args:
            eval_results: Dictionary returned from evaluate_model containing predictions, targets, and probabilities
            n_show: Number of nodes to display (None plots all nodes, default: None)
            figsize: Figure size as (width, height) tuple for matplotlib (default: (15, 7))
            save_path: Optional path to save the figure (e.g., 'plot.png' or 'plot.html')
            backend: Plotting backend - 'matplotlib' or 'plotly' (default: 'matplotlib')
            sample_idx: Index of sample to plot (default: 0)
            interactive: Whether to make plotly plots interactive (default: False)
            
        Returns:
            For matplotlib: fig, ax
            For plotly: fig
        """
        if backend == 'plotly':
            return self._plot_with_plotly(eval_results, n_show, save_path, sample_idx, interactive)
        else:
            return self._plot_with_matplotlib(eval_results, n_show, figsize, save_path, sample_idx)
    
    def _plot_with_matplotlib(self, eval_results, n_show, figsize, save_path, sample_idx):
        """Internal method for matplotlib plotting of detection results."""
        import matplotlib.pyplot as plt
        
        # Extract predictions and targets from evaluation results
        # Shape: (num_samples, num_nodes)
        preds = eval_results['predictions'].numpy()
        targets = eval_results['targets'].numpy()
        probs = eval_results['probabilities'].numpy()  # Shape: (num_samples, num_nodes, num_classes)
        
        print(f"Data shape: {preds.shape} (samples, nodes)")
        print(f"Plotting sample {sample_idx}")
        
        # Extract specific sample
        preds_plot = preds[sample_idx]
        targets_plot = targets[sample_idx]
        probs_plot = probs[sample_idx]  # Shape: (num_nodes, num_classes)
        
        # Get confidence scores (probability of predicted class)
        confidence_scores = probs_plot[np.arange(len(preds_plot)), preds_plot]
        
        # Determine number of nodes to show (all by default)
        if n_show is None:
            n_show = len(preds_plot)
        else:
            n_show = min(n_show, len(preds_plot))
        
        x = np.arange(n_show)
        
        print(f"Plotting {n_show} nodes")
        
        # Set white background style
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, facecolor='white', sharex=True)
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        
        # Plot 1: Predictions vs True Labels
        ax1.plot(x, targets_plot[:n_show], 'o-', label='True Labels', color='#2c3e50', 
                 alpha=0.9, markersize=8, linewidth=2.5, zorder=3)
        ax1.plot(x, preds_plot[:n_show], 's-', label='Predictions', color='#e74c3c', 
                 alpha=0.8, markersize=7, linewidth=2, zorder=2)
        
        # Highlight correct/incorrect predictions
        correct_mask = (preds_plot[:n_show] == targets_plot[:n_show])
        incorrect_idx = x[~correct_mask]
        if len(incorrect_idx) > 0:
            ax1.scatter(incorrect_idx, preds_plot[:n_show][~correct_mask], 
                       s=200, facecolors='none', edgecolors='red', linewidths=2.5, 
                       zorder=4, label='Incorrect')
        
        ax1.set_ylabel('Class Label', fontsize=13, fontweight='bold', color='#2c3e50')
        ax1.set_title(f'Source Detection Predictions (Sample {sample_idx})', 
                     fontsize=15, fontweight='bold', color='#2c3e50', pad=20)
        ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, 
                  loc='best', edgecolor='#bdc3c7', facecolor='white')
        ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#95a5a6')
        
        # Plot 2: Confidence Scores
        colors = ['#27ae60' if c else '#e74c3c' for c in correct_mask]
        bars = ax2.bar(x, confidence_scores[:n_show], color=colors, alpha=0.7, edgecolor='#2c3e50', linewidth=1.2)
        
        ax2.axhline(y=0.5, color='#95a5a6', linestyle='--', linewidth=2, label='Decision Threshold', alpha=0.7)
        ax2.set_xlabel('Node Index', fontsize=13, fontweight='bold', color='#2c3e50')
        ax2.set_ylabel('Confidence Score', fontsize=13, fontweight='bold', color='#2c3e50')
        ax2.set_title('Prediction Confidence Scores', fontsize=15, fontweight='bold', color='#2c3e50', pad=20)
        ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, 
                  loc='best', edgecolor='#bdc3c7', facecolor='white')
        ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#95a5a6', axis='y')
        
        # Customize spines for both plots
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor('#bdc3c7')
                spine.set_linewidth(1.2)
            ax.tick_params(colors='#2c3e50', labelsize=10)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nFigure saved to: {save_path}")
        
        # Calculate and print accuracy statistics
        accuracy = correct_mask.sum() / n_show * 100
        mean_confidence = confidence_scores[:n_show].mean()
        print(f"\n{'='*60}")
        print(f"Visualization Statistics")
        print(f"{'='*60}")
        print(f"Accuracy (shown nodes): {accuracy:.2f}%")
        print(f"Mean Confidence: {mean_confidence:.4f}")
        print(f"Correct Predictions: {correct_mask.sum()}/{n_show}")
        print(f"{'='*60}")
        
        return fig, (ax1, ax2)
    
    def _plot_with_plotly(self, eval_results, n_show, save_path, sample_idx, interactive):
        """Internal method for plotly plotting of detection results."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly is required for interactive plotting. Install with: pip install plotly")
        
        # Extract predictions and targets from evaluation results
        preds = eval_results['predictions'].numpy()
        targets = eval_results['targets'].numpy()
        probs = eval_results['probabilities'].numpy()
        
        print(f"Data shape: {preds.shape} (samples, nodes)")
        print(f"Plotting sample {sample_idx}")
        
        # Extract specific sample
        preds_plot = preds[sample_idx]
        targets_plot = targets[sample_idx]
        probs_plot = probs[sample_idx]
        
        # Get confidence scores
        confidence_scores = probs_plot[np.arange(len(preds_plot)), preds_plot]
        
        # Determine number of nodes to show
        if n_show is None:
            n_show = len(preds_plot)
        else:
            n_show = min(n_show, len(preds_plot))
        
        x = np.arange(n_show)
        
        print(f"Plotting {n_show} nodes")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Source Detection Predictions (Sample {sample_idx})', 
                          'Prediction Confidence Scores'),
            vertical_spacing=0.15
        )
        
        # Plot 1: Predictions vs True Labels
        fig.add_trace(go.Scatter(
            x=x, y=targets_plot[:n_show],
            mode='lines+markers',
            name='True Labels',
            line=dict(color='#2c3e50', width=2.5),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>True Label</b><br>Node: %{x}<br>Class: %{y}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=x, y=preds_plot[:n_show],
            mode='lines+markers',
            name='Predictions',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=7, symbol='square'),
            hovertemplate='<b>Prediction</b><br>Node: %{x}<br>Class: %{y}<extra></extra>'
        ), row=1, col=1)
        
        # Highlight incorrect predictions
        correct_mask = (preds_plot[:n_show] == targets_plot[:n_show])
        incorrect_idx = x[~correct_mask]
        if len(incorrect_idx) > 0:
            fig.add_trace(go.Scatter(
                x=incorrect_idx, y=preds_plot[:n_show][~correct_mask],
                mode='markers',
                name='Incorrect',
                marker=dict(size=15, symbol='circle-open', color='red', line=dict(width=2.5)),
                hovertemplate='<b>Incorrect</b><br>Node: %{x}<br>Predicted: %{y}<extra></extra>'
            ), row=1, col=1)
        
        # Plot 2: Confidence Scores
        colors = ['#27ae60' if c else '#e74c3c' for c in correct_mask]
        fig.add_trace(go.Bar(
            x=x, y=confidence_scores[:n_show],
            marker=dict(color=colors, line=dict(color='#2c3e50', width=1.2)),
            name='Confidence',
            hovertemplate='<b>Confidence</b><br>Node: %{x}<br>Score: %{y:.4f}<extra></extra>'
        ), row=2, col=1)
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="#95a5a6", line_width=2,
                     annotation_text="Decision Threshold", row=2, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Node Index", row=2, col=1, 
                        titlefont=dict(size=14, color='#2c3e50'),
                        gridcolor='rgba(149, 165, 166, 0.25)')
        fig.update_yaxes(title_text="Class Label", row=1, col=1,
                        titlefont=dict(size=14, color='#2c3e50'),
                        gridcolor='rgba(149, 165, 166, 0.25)')
        fig.update_yaxes(title_text="Confidence Score", row=2, col=1,
                        titlefont=dict(size=14, color='#2c3e50'),
                        gridcolor='rgba(149, 165, 166, 0.25)')
        
        fig.update_layout(
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
            height=900
        )
        
        # Save figure if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
                print(f"\nInteractive figure saved to: {save_path}")
            else:
                fig.write_image(save_path, width=1200, height=900)
                print(f"\nFigure saved to: {save_path}")
        
        if interactive:
            fig.show()
        
        # Calculate and print statistics
        accuracy = correct_mask.sum() / n_show * 100
        mean_confidence = confidence_scores[:n_show].mean()
        print(f"\n{'='*60}")
        print(f"Visualization Statistics")
        print(f"{'='*60}")
        print(f"Accuracy (shown nodes): {accuracy:.2f}%")
        print(f"Mean Confidence: {mean_confidence:.4f}")
        print(f"Correct Predictions: {correct_mask.sum()}/{n_show}")
        print(f"{'='*60}")
        
        return fig

