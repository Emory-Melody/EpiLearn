"""
Scenario Modeling Task for Epidemiological Data.

Scenario modeling answers "what if" questions by:
- Training models on historical epidemic trajectories with various interventions
- Predicting future outcomes under different intervention scenarios
- Measuring the treatment effect of interventions compared to no-intervention baseline

Key differences from forecasting:
- Input X: (L, n_scenarios, N+4) - shared history with scenario-specific intervention features
- Output Y: (H, n_scenarios, N) - divergent futures per scenario
- Metrics: Treatment effect metrics comparing intervention scenarios to baseline

This task extends BaseTask and supports temporal models from epilearn/models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from .base import BaseTask
from ..data import Dataset
from ..utils.compartmental_models import SEIRVIModel


class ScenarioTask(BaseTask):
    """
    Scenario Modeling task for epidemiological counterfactual analysis.
    
    Extends BaseTask to support multi-scenario predictions with shared history.
    
    Data format:
    - Features X: (batch, lookback, n_scenarios, N+4)
      - N compartments + 4 intervention features (vacc_rate, vacc_delay, isol_rate, isol_delay)
      - Compartments are IDENTICAL across scenarios (shared history)
      - Intervention features can differ for dynamic interventions
    - Targets Y: (batch, horizon, n_scenarios, N)
      - Future compartment values per scenario
      - Scenarios diverge based on different intervention policies
    
    Metrics (Treatment Effect):
    - PEHE (Precision in Estimation of Heterogeneous Effects):
      Measures accuracy in predicting the difference between intervention and baseline
    - ATE Error (Average Treatment Effect Error):
      Measures accuracy in predicting the average effect of intervention
    
    Usage:
        from epilearn.tasks import ScenarioTask
        from epilearn.models.Temporal import GRUModel
        
        # Create task with any model
        task = ScenarioTask(prototype=GRUModel, lookback=60, horizon=30, n_scenarios=4)
        
        # Generate dataset from compartmental model
        dataset = task.generate_dataset(n_samples=100)
        
        # Run training with rolling evaluation
        results = task.rolling_train(dataset=dataset, ...)
    """
    
    def __init__(
        self,
        prototype=None,
        model=None,
        lookback: int = 60,
        horizon: int = 30,
        n_scenarios: int = 4,
        compartmental_model: Optional[SEIRVIModel] = None,
        baseline_scenario_idx: int = 0,
        target_compartment: str = 'I',
        device: str = 'cpu',
    ):
        """
        Initialize ScenarioTask.
        
        Args:
            prototype: Model class (e.g., GRUModel, LSTMModel)
            model: Pre-initialized model (optional)
            lookback: Historical window size (L)
            horizon: Future projection size (H)
            n_scenarios: Number of intervention scenarios per sample
            compartmental_model: SEIRVIModel for data generation (optional)
            baseline_scenario_idx: Index of baseline/control scenario (default: 0)
            target_compartment: Compartment name to measure treatment effects on (default: 'I' for Infectious)
            device: 'cpu' or 'cuda'
        """
        super().__init__(prototype, model, None, lookback, horizon, 0, device)
        self.n_scenarios = n_scenarios
        self.baseline_scenario_idx = baseline_scenario_idx
        
        # Default compartmental model for data generation
        if compartmental_model is None:
            self.comp_model = SEIRVIModel(
                beta=0.3, gamma=0.1, sigma=0.2,
                vaccine_efficacy=0.8, isolation_efficacy=0.9,
            )
        else:
            self.comp_model = compartmental_model
        
        self.N = len(self.comp_model.compartments)  # Number of compartments
        
        # Resolve target compartment name to index
        self.target_compartment = target_compartment
        self.target_compartment_idx = self._resolve_compartment_idx(target_compartment)
    
    def _resolve_compartment_idx(self, compartment: str) -> int:
        """
        Resolve compartment name to index.
        
        Args:
            compartment: Compartment name (e.g., 'S', 'E', 'I', 'R', 'V', 'Isolated')
                         or integer index
        
        Returns:
            Integer index of the compartment
        """
        if isinstance(compartment, int):
            return compartment
        
        compartments = self.comp_model.compartments
        if compartment in compartments:
            return compartments.index(compartment)
        
        # Common aliases
        aliases = {
            'infectious': 'I',
            'infected': 'I', 
            'susceptible': 'S',
            'exposed': 'E',
            'recovered': 'R',
            'vaccinated': 'V',
            'isolated': 'Isolated',
        }
        resolved = aliases.get(compartment.lower(), compartment)
        if resolved in compartments:
            return compartments.index(resolved)
        
        raise ValueError(
            f"Unknown compartment '{compartment}'. "
            f"Available: {compartments}"
        )
    
    def generate_dataset(
        self,
        n_samples: int = 100,
        population: float = 1e6,
        process_noise: float = None,
        seed: int = 42,
        **kwargs,
    ) -> Dataset:
        """
        Generate scenario modeling dataset using compartmental model.
        
        Args:
            n_samples: Number of samples to generate
            population: Population size
            process_noise: Optional noise in simulation
            seed: Random seed
            **kwargs: Additional parameters for generate_multi_scenario_dataset
            
        Returns:
            Dataset with:
            - x: (n_samples, lookback, n_scenarios, N+4)
            - y: (n_samples, horizon, n_scenarios, N)
        """
        X, Y, metadata = self.comp_model.generate_multi_scenario_dataset(
            n_samples=n_samples,
            lookback=self.lookback,
            horizon=self.horizon,
            n_scenarios=self.n_scenarios,
            population=population,
            process_noise=process_noise,
            seed=seed,
            **kwargs,
        )
        
        # Store metadata for later use
        self._metadata = metadata
        
        # Create Dataset
        dataset = Dataset(
            x=X,  # (n_samples, lookback, n_scenarios, N+4)
            y=Y,  # (n_samples, horizon, n_scenarios, N)
            timestamps=list(range(n_samples)),
        )
        
        return dataset
    
    def _generate_split(self, ds, lookback=None, interval=None):
        """
        Generate split dictionary from Dataset.
        
        Overrides BaseTask._generate_split for scenario-specific shapes.
        
        For temporal models, we flatten scenarios into the batch dimension:
        - Input: (batch, lookback, n_scenarios, N+4) -> (batch * n_scenarios, lookback, N+4)
        - Output: (batch, H, n_scenarios, N) -> (batch * n_scenarios, H*N) [flattened for model]
        """
        features = ds.x  # (T, L, n_scenarios, N+4)
        targets = ds.y   # (T, H, n_scenarios, N)
        
        T, L, n_scenarios, n_feat = features.shape
        _, H, _, N = targets.shape
        
        # Reshape features: (T, L, n_scenarios, F) -> (T * n_scenarios, L, F)
        # This treats each scenario as a separate sample for training
        features_flat = features.permute(0, 2, 1, 3).reshape(T * n_scenarios, L, n_feat)
        
        # Reshape targets: (T, H, n_scenarios, N) -> (T * n_scenarios, H*N)
        # Flatten H*N for model output compatibility
        targets_flat = targets.permute(0, 2, 1, 3).reshape(T * n_scenarios, H * N)
        
        return {
            'features': features_flat.float().to(self.device),
            'targets': targets_flat.float().to(self.device),
            'graph': ds.graph,
            'dynamic_graph': ds.dynamic_graph,
            'states': ds.states,
            # Store original shapes for metric computation
            '_original_shape': (T, n_scenarios),
            '_n_scenarios': n_scenarios,
            '_H': H,
            '_N': N,
        }
    
    def _init_model_from_split(self, train_split, model_args=None):
        """
        Initialize model from train split.
        
        Features shape: (batch, lookback, N+4)
        - num_timesteps_input = lookback
        - num_features = N+4 (compartments + intervention info)
        - num_timesteps_output = horizon * N (flattened)
        """
        if model_args is None:
            model_args = {}
        
        features = train_split['features']
        targets = train_split['targets']
        
        # Features: (batch, lookback, N+4)
        num_timesteps_input = features.shape[1]  # lookback
        num_features = features.shape[2]         # N+4
        
        # Output: targets are already flattened to (batch, H*N)
        num_timesteps_output = targets.shape[1]  # H * N
        
        base_inputs = {
            'num_features': num_features,
            'num_timesteps_input': num_timesteps_input,
            'num_timesteps_output': num_timesteps_output,
            'device': self.device,
        }
        base_inputs.update(model_args)
        
        self.model = self.prototype(**base_inputs)
        self.model = self.model.to(self.device)
    
    def _build_model_inputs(self, train_split, model_args):
        """
        Override base _build_model_inputs for scenario modeling.
        
        For scenario modeling, num_timesteps_output = H * N (flattened targets).
        The base class uses self.horizon which is just H.
        """
        features = train_split['features']
        targets = train_split['targets']
        
        # targets shape: (batch, H*N) - already flattened by _generate_split
        num_timesteps_output = targets.shape[1]  # H * N
        
        return {
            "num_features": features.shape[2],
            "num_timesteps_input": features.shape[1],  # lookback
            "num_timesteps_output": num_timesteps_output,
            "device": self.device,
            **model_args
        }
    
    def evaluate_model(
        self,
        model=None,
        dataset=None,
        baseline_scenario_idx: int = None,
        target_compartment: str = None,
        **kwargs,
    ) -> Dict:
        """
        Evaluate scenario model with treatment effect metrics.
        
        Unlike standard forecasting metrics (MSE, MAE), scenario modeling uses
        metrics that measure how well the model captures the EFFECT of interventions.
        
        Metrics:
        1. PEHE (Precision in Estimation of Heterogeneous Effects):
           sqrt(E[(tau_pred - tau_true)^2])
           where tau = Y_intervention - Y_baseline
           
        2. ATE Error (Average Treatment Effect Error):
           ``|E[tau_pred] - E[tau_true]|``
        
        Args:
            model: Model to evaluate
            dataset: Dataset dict with 'features', 'targets'
            baseline_scenario_idx: Index of baseline scenario (default: uses self.baseline_scenario_idx)
            target_compartment: Compartment name or index (default: uses self.target_compartment)
            
        Returns:
            Dictionary with PEHE, ATE Error, and predictions
        """
        # Use class defaults if not specified
        if baseline_scenario_idx is None:
            baseline_scenario_idx = self.baseline_scenario_idx
        if target_compartment is None:
            target_compartment_idx = self.target_compartment_idx
        else:
            target_compartment_idx = self._resolve_compartment_idx(target_compartment)
        if model is None:
            if not hasattr(self, "model"):
                raise RuntimeError("Model not loaded")
            model = self.model
        
        features = dataset['features'].to(self.device)
        targets = dataset['targets'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            out = model.predict(
                feature=features,
                graph=dataset.get('graph'),
                states=dataset.get('states'),
                dynamic_graph=dataset.get('dynamic_graph'),
            )
        
        if isinstance(out, tuple):
            out = out[0]
        
        preds = out.detach().cpu()
        targets = targets.detach().cpu()
        
        # Reshape back to (T, n_scenarios, H, N)
        n_scenarios = dataset.get('_n_scenarios', self.n_scenarios)
        original_shape = dataset.get('_original_shape')
        
        if original_shape:
            T, _ = original_shape
            H = self.horizon
            N = self.N
            
            # preds shape: (T * n_scenarios, H * N) -> (T, n_scenarios, H, N)
            preds = preds.reshape(T, n_scenarios, H, N)
            targets = targets.reshape(T, n_scenarios, H, N)
        else:
            # Assume preds/targets are already properly shaped
            T = preds.shape[0] // n_scenarios
            H = self.horizon
            N = self.N
            preds = preds.reshape(T, n_scenarios, H, N)
            targets = targets.reshape(T, n_scenarios, H, N)
        
        # Compute treatment effects
        # tau = Y_intervention - Y_baseline for target compartment
        comp_idx = target_compartment_idx
        
        # True treatment effects: (T, n_scenarios-1, H)
        baseline_true = targets[:, baseline_scenario_idx, :, comp_idx]  # (T, H)
        tau_true_all = []
        for s in range(n_scenarios):
            if s != baseline_scenario_idx:
                tau_true = targets[:, s, :, comp_idx] - baseline_true  # (T, H)
                tau_true_all.append(tau_true)
        tau_true = torch.stack(tau_true_all, dim=1)  # (T, n_scenarios-1, H)
        
        # Predicted treatment effects
        baseline_pred = preds[:, baseline_scenario_idx, :, comp_idx]  # (T, H)
        tau_pred_all = []
        for s in range(n_scenarios):
            if s != baseline_scenario_idx:
                tau_pred = preds[:, s, :, comp_idx] - baseline_pred
                tau_pred_all.append(tau_pred)
        tau_pred = torch.stack(tau_pred_all, dim=1)  # (T, n_scenarios-1, H)
        
        # PEHE: sqrt(mean((tau_pred - tau_true)^2))
        pehe = torch.sqrt(torch.mean((tau_pred - tau_true) ** 2)).item()
        
        # ATE Error: |mean(tau_pred) - mean(tau_true)|
        ate_error = torch.abs(tau_pred.mean() - tau_true.mean()).item()
        
        # Standard MSE for reference
        mse = torch.mean((preds - targets) ** 2).item()
        
        print(f"\n{'='*60}")
        print(f"SCENARIO MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"\n--- Treatment Effect Metrics ---")
        print(f"PEHE (Precision in Heterogeneous Effect):  {pehe:.4f}")
        print(f"ATE Error (Average Treatment Effect):      {ate_error:.4f}")
        print(f"\n--- Standard Metrics ---")
        print(f"MSE:                                       {mse:.6f}")
        print(f"{'='*60}")
        
        return {
            'pehe': pehe,
            'ate_error': ate_error,
            'mse': mse,
            'predictions': preds,
            'targets': targets,
            'tau_pred': tau_pred,
            'tau_true': tau_true,
        }
    
    def _compute_fold_metrics(self, preds, targets, metric_names=None, residual_fn=None):
        """
        Compute scenario-specific metrics for a fold.
        
        Overrides base class to use treatment effect metrics.
        """
        # Reshape if needed
        n_scenarios = self.n_scenarios
        
        if preds.dim() == 2:
            # (batch, H*N) -> need to infer T and reshape
            batch_size = preds.shape[0]
            T = batch_size // n_scenarios
            H = self.horizon
            N = self.N
            
            preds = preds.reshape(T, n_scenarios, H, N)
            targets = targets.reshape(T, n_scenarios, H, N)
        
        # Compute treatment effects using configured baseline and target compartment
        comp_idx = self.target_compartment_idx
        baseline_idx = self.baseline_scenario_idx
        
        baseline_true = targets[:, baseline_idx, :, comp_idx]
        baseline_pred = preds[:, baseline_idx, :, comp_idx]
        
        tau_true_all = []
        tau_pred_all = []
        for s in range(n_scenarios):
            if s == baseline_idx:
                continue  # Skip baseline scenario
            tau_true_all.append(targets[:, s, :, comp_idx] - baseline_true)
            tau_pred_all.append(preds[:, s, :, comp_idx] - baseline_pred)
        
        tau_true = torch.stack(tau_true_all, dim=1)
        tau_pred = torch.stack(tau_pred_all, dim=1)
        
        pehe = torch.sqrt(torch.mean((tau_pred - tau_true) ** 2)).item()
        ate_error = torch.abs(tau_pred.mean() - tau_true.mean()).item()
        
        return {
            'pehe': pehe,
            'ate_error': ate_error,
        }
    
    def plot_scenario_comparison(
        self,
        eval_results: Dict,
        sample_idx: int = 0,
        compartment_idx: int = 2,
        scenario_names: List[str] = None,
        figsize: Tuple[int, int] = (14, 5),
    ):
        """
        Plot scenario comparison for a single sample.
        
        Args:
            eval_results: Results from evaluate_model
            sample_idx: Which sample to plot
            compartment_idx: Which compartment to plot (default: 2 = Infectious)
            scenario_names: Names for each scenario
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        preds = eval_results['predictions'][sample_idx].numpy()  # (n_scenarios, H, N)
        targets = eval_results['targets'][sample_idx].numpy()
        
        n_scenarios = preds.shape[0]
        H = preds.shape[1]
        
        if scenario_names is None:
            scenario_names = [f'Scenario {i}' for i in range(n_scenarios)]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        colors = plt.cm.Set2(np.linspace(0, 1, n_scenarios))
        
        # Plot 1: True vs Predicted trajectories
        ax = axes[0]
        t = np.arange(H)
        for s in range(n_scenarios):
            ax.plot(t, targets[s, :, compartment_idx], '-', color=colors[s], 
                    linewidth=2, label=f'{scenario_names[s]} (True)')
            ax.plot(t, preds[s, :, compartment_idx], '--', color=colors[s],
                    linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Horizon Step')
        ax.set_ylabel('Compartment Value')
        ax.set_title('Scenario Trajectories (Solid=True, Dashed=Predicted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Treatment effects
        ax = axes[1]
        baseline_true = targets[0, :, compartment_idx]
        baseline_pred = preds[0, :, compartment_idx]
        
        for s in range(1, n_scenarios):
            tau_true = targets[s, :, compartment_idx] - baseline_true
            tau_pred = preds[s, :, compartment_idx] - baseline_pred
            
            ax.plot(t, tau_true, '-', color=colors[s], linewidth=2, 
                    label=f'{scenario_names[s]} Effect (True)')
            ax.plot(t, tau_pred, '--', color=colors[s], linewidth=2, alpha=0.7)
        
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Horizon Step')
        ax.set_ylabel('Treatment Effect (vs Baseline)')
        ax.set_title('Treatment Effects')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes


# Alias for compatibility
Scenario = ScenarioTask
