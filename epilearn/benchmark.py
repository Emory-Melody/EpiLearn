"""
EpiLearn Benchmark Module

A concise benchmarking framework for epidemic forecasting, nowcasting, and scenario modeling with:
- Parallel execution across multiple GPUs
- Rolling evaluation with Optuna hyperparameter tuning
- Detailed result saving (aggregate + per-fold + per-model)

Supports three task types:
- Forecasting: Predict future values from historical features
- Nowcasting: Predict final values from incomplete reporting triangles
- Scenario: Counterfactual analysis comparing intervention scenarios (treatment effect estimation)

Usage:
    python -m epilearn.benchmark --config configs/benchmark_config.yaml
    python -m epilearn.benchmark --config configs/nowcast_benchmark_config.yaml
    python -m epilearn.benchmark --config configs/scenario_benchmark_config.yaml
"""

import csv
import json
import multiprocessing as mp
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import random

import numpy as np
import torch
import yaml


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelResult:
    """Result from training a single model."""
    model_name: str
    requires_graph: bool
    lookback: int
    horizon: int
    task_type: str = 'forecast'  # 'forecast', 'nowcast', or 'scenario'
    n_scenarios: int = 0  # For scenario modeling
    evaluation_paradigm: str = 'unknown'  # 'node-independent', 'graph-joint', 'zero-shot'
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    all_predictions: List[Any] = field(default_factory=list)
    all_targets: List[Any] = field(default_factory=list)
    all_inputs: List[Any] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    optuna_trials: List[Dict[str, Any]] = field(default_factory=list)  # Node-level trial results
    error: Optional[str] = None
    runtime_seconds: float = 0.0


# =============================================================================
# Model Registry
# =============================================================================

# Model category registries
TEMPORAL_MODELS = {
    'GRUModel', 'LSTMModel', 'CNNModel', 'MLPModel',
    'DlinearModel', 'PatchTSTModel',
    'LinearRegressionModel', 'RidgeModel', 'LassoModel',
    'ElasticNetModel', 'RandomForestModel', 'GradientBoostingModel',
    'SVRModel', 'KNNModel', 'DecisionTreeModel',
    'ARIMAModel', 'VARMAXModel', 'SeasonalNaiveModel', 'RKINowcastModel', 'NobBSModel',
    'iTransformerModel',
    'TSMixerModel', 'FreTSModel',
    'SIRModel', 'SEIRModel',
    'EINNModel', 'EpiDeepModel', 'CALINetModel',
}

FOUNDATION_MODELS = {
    'ChronosModel', 'ChronosBoltModel',
    'MoiraiModel', 'MoiraiBaseModel', 'MoiraiLargeModel',
    'MomentModel', 'MomentSmallModel', 'MomentBaseModel',
    'TimesFMModel',
}

# Map foundation models → required Python package
FOUNDATION_DEPS = {
    'ChronosModel':     'chronos',
    'ChronosBoltModel': 'chronos',
    'MoiraiModel':      'uni2ts',
    'MoiraiBaseModel':  'uni2ts',
    'MoiraiLargeModel': 'uni2ts',
    'MomentModel':      'momentfm',
    'MomentSmallModel': 'momentfm',
    'MomentBaseModel':  'momentfm',
    'TimesFMModel':     'timesfm',
}

SPATIOTEMPORAL_MODELS = {
    'STGCN', 'DSTGCN', 'DCRNN', 'GraphWaveNet', 
    'EpiGNN', 'ColaGNN', 'MepoGNN', 'ATMGNN',
}


def is_foundation_model(model_name: str) -> bool:
    """Check if a model name corresponds to a foundation model."""
    return model_name in FOUNDATION_MODELS


def check_foundation_deps(model_name: str) -> Optional[str]:
    """
    Check if the required dependency for a foundation model is installed.
    Returns None if OK, or an error message string if the dependency is missing.
    """
    pkg = FOUNDATION_DEPS.get(model_name)
    if pkg is None:
        return None  # Not a foundation model, no special deps
    try:
        __import__(pkg)
        return None
    except ImportError:
        install_hint = {
            'chronos':   'pip install chronos-forecasting',
            'uni2ts':    'pip install uni2ts',
            'momentfm':  'pip install momentfm',
            'timesfm':   'pip install timesfm',
        }.get(pkg, f'pip install {pkg}')
        return (
            f"{model_name} requires '{pkg}' which is not installed. "
            f"Install with: {install_hint}"
        )


def get_model_class(model_name: str):
    """Dynamically import and return a model class by name."""
    if model_name in TEMPORAL_MODELS:
        from epilearn.models import Temporal
        return getattr(Temporal, model_name)
    elif model_name in FOUNDATION_MODELS:
        from epilearn.models import Temporal
        return getattr(Temporal, model_name)
    elif model_name in SPATIOTEMPORAL_MODELS:
        from epilearn.models import SpatialTemporal
        return getattr(SpatialTemporal, model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Training Function (handles both forecasting and nowcasting)
# =============================================================================

def train_model(
    model_name: str,
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: str = 'cuda',
    task_type: str = 'forecast',
) -> ModelResult:
    """
    Train a model with rolling evaluation and optional Optuna tuning.
    
    Supports three task types:
    - forecast: Uses Forecast task with CSV data
    - nowcast: Uses NowcastTask with .npz triangle data
    - scenario: Uses ScenarioTask for counterfactual intervention analysis
    """
    import time
    import traceback
    from epilearn.data import Dataset
    from epilearn.tasks import Forecast, NowcastTask
    from epilearn.tasks.scenario_modeling import ScenarioTask
    from epilearn.utils import transforms
    from epilearn.utils.compartmental_models import SEIRVIModel
    
    start_time = time.time()
    
    requires_graph = model_config.get('requires_graph', False)
    optuna_model_args = model_config.get('optuna_model_args', {})
    optimizer_params = model_config.get('optimizer_params', {})
    model_args = model_config.get('model_args', {})
    if task_type == 'nowcast':
        model_args.setdefault('nowcast', True)
    class_name = model_config.get('class_name', model_name)
    _is_foundation = is_foundation_model(class_name)

    # Evaluation settings
    lookback = eval_config.get('lookback', 14)
    horizon = eval_config.get('horizon', 7)
    train_size = eval_config.get('train_size', 100)
    val_size = eval_config.get('val_size', 30)
    test_size = eval_config.get('test_size', 35)
    step_size = eval_config.get('step_size', 30)
    max_folds = eval_config.get('max_folds', 3)
    n_trials = model_config.get('n_trials', eval_config.get('n_trials', 5))
    use_optuna = eval_config.get('use_optuna', True)

    # Determine evaluation paradigm for fair-comparison documentation
    if _is_foundation:
        eval_paradigm = 'zero-shot'
    elif requires_graph:
        eval_paradigm = 'graph-joint'
    else:
        eval_paradigm = 'node-independent'
    
    result = ModelResult(
        model_name=model_name,
        requires_graph=requires_graph,
        lookback=lookback,
        horizon=horizon,
        task_type=task_type,
        evaluation_paradigm=eval_paradigm,
    )
    
    # Check foundation model dependencies before doing any work
    if _is_foundation:
        dep_error = check_foundation_deps(class_name)
        if dep_error:
            result.error = f"SKIPPED (missing dependency): {dep_error}"
            result.runtime_seconds = 0.0
            return result
    
    try:
        # Get max lookback from optuna args
        if 'lookback' in optuna_model_args:
            init_lookback = max(optuna_model_args['lookback'])
        else:
            init_lookback = lookback
        
        # Get model class (use class_name if specified, e.g. for RKI variants)
        model_class = get_model_class(class_name)
        
        # Task-specific setup
        if task_type == 'scenario':
            # =========================================================
            # Scenario Modeling Task
            # =========================================================
            scenario_cfg = dataset_config.get('scenario', {})
            n_scenarios = scenario_cfg.get('n_scenarios', 4)
            n_samples = scenario_cfg.get('n_samples', 200)
            population = scenario_cfg.get('population', 1e6)
            process_noise = scenario_cfg.get('process_noise', 0.01)
            seed = scenario_cfg.get('seed', 42)
            baseline_scenario_idx = scenario_cfg.get('baseline_scenario_idx', 0)
            target_compartment = scenario_cfg.get('target_compartment', 'I')
            
            # Compartmental model parameters
            comp_params = scenario_cfg.get('compartmental_model', {})
            comp_model = SEIRVIModel(
                beta=comp_params.get('beta', 0.3),
                gamma=comp_params.get('gamma', 0.1),
                sigma=comp_params.get('sigma', 0.2),
                vaccine_efficacy=comp_params.get('vaccine_efficacy', 0.8),
                isolation_efficacy=comp_params.get('isolation_efficacy', 0.9),
            )
            
            task = ScenarioTask(
                prototype=model_class,
                lookback=init_lookback,
                horizon=horizon,
                n_scenarios=n_scenarios,
                compartmental_model=comp_model,
                baseline_scenario_idx=baseline_scenario_idx,
                target_compartment=target_compartment,
                device=device,
            )
            
            # Generate synthetic scenario dataset
            dataset = task.generate_dataset(
                n_samples=n_samples,
                population=population,
                process_noise=process_noise,
                seed=seed,
            )
            
            # Set up transforms
            transform_config = dataset_config.get('transforms', {})
            transform_dict = {}
            if transform_config.get('normalize_features', True):
                transform_dict['features'] = [transforms.normalize_feat()]
            if transform_config.get('normalize_target', True):
                transform_dict['target'] = [transforms.normalize_target()]
            if transform_dict:
                dataset.set_transforms(transforms.Compose(transform_dict))
            
            result.n_scenarios = n_scenarios
            baseline = None
            report_metrics = ['pehe', 'ate_error']  # Scenario-specific metrics
            
        elif task_type == 'nowcast':
            # Load triangle data for nowcasting
            triangle_path = dataset_config.get('triangle_path')
            if not triangle_path:
                result.error = "No triangle_path specified in dataset config"
                return result
            
            data = NowcastTask.load_triangle(triangle_path)
            min_delay = dataset_config.get('min_delay', 3)
            max_delay = dataset_config.get('max_delay', None)
            
            task = NowcastTask(
                prototype=model_class,
                lookback=init_lookback,
                horizon=horizon,
                min_delay=min_delay,
                max_delay=max_delay,
                device=device,
            )
            if model_config.get('extract_univariate', False):
                task._extract_univariate = True
            dataset = task.create_dataset(data['triangle'], data['final_counts'], data['delays'])
            
            # Compute naive baseline for nowcasting
            baseline = task.compute_naive_baseline(dataset)
            report_metrics = ['mse', 'mae', 'rmse']  # Standard metrics for nowcasting
            
        else:
            # Load CSV data for forecasting
            dataset = Dataset.from_csv(
                file_path=dataset_config['feature_path'],
                timestamp_col=dataset_config.get('timestamp_col', 'time'),
                region_col=dataset_config.get('region_col', 'node'),
                feature_cols=dataset_config.get('feature_cols'),
                target_cols=dataset_config.get('target_cols'),
                graph_file=dataset_config.get('graph_path') if requires_graph else None,
                graph_weight_col=dataset_config.get('graph_weight_col'),  # FIX: Pass edge weight column parameter
            )
            
            # Set up transforms
            # All models receive the same normalized input/target so that
            # evaluation metrics are computed on the same scale.
            transform_config = dataset_config.get('transforms', {})
            transform_dict = {}
            if transform_config.get('normalize_features', True):
                transform_dict['features'] = [transforms.normalize_feat()]
            if transform_config.get('normalize_target', True):
                transform_dict['target'] = [transforms.normalize_target()]
            if requires_graph and transform_config.get('normalize_graph', True):
                transform_dict['graph'] = [transforms.normalize_adj()]
            
            if transform_dict:
                dataset.set_transforms(transforms.Compose(transform_dict))
            
            # For temporal models, remove graph
            if not requires_graph:
                dataset = Dataset(
                    x=dataset.x.clone(),
                    y=dataset.y.clone(),
                    graph=None,
                    dynamic_graph=None,
                    timestamps=dataset.timestamps,
                    regions=dataset.regions,
                )
                if transform_dict:
                    dataset.set_transforms(transforms.Compose(transform_dict))
            
            # Check graph requirement
            if requires_graph and dataset.graph is None:
                result.error = f"{model_name} requires graph but none available"
                return result
            
            task = Forecast(
                prototype=model_class,
                lookback=init_lookback,
                horizon=horizon,
                device=device
            )
            baseline = None
            report_metrics = ['mse', 'mae', 'rmse']  # Standard forecasting metrics

        # Run rolling evaluation (same for all task types)
        results = task.rolling_train(
            dataset=dataset,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            step_size=step_size,
            expanding=True,
            max_folds=max_folds,
            patience=eval_config.get('patience', 15),
            verbose=eval_config.get('verbose', False),
            use_optuna=use_optuna,
            n_trials=n_trials if use_optuna else 1,
            optimizer_params=optimizer_params,
            optuna_model_args=optuna_model_args if use_optuna else None,
            model_args=model_args,
            conformal_alpha=eval_config.get('conformal_alpha', 0.1),
            report_metrics=report_metrics,
        )
        
        # Extract results
        if results:
            result.fold_results = results.get('fold_results', [])
            result.aggregate_metrics = results.get('aggregate_metrics') or {}
            result.all_predictions = results.get('all_predictions', [])
            result.all_targets = results.get('all_targets', [])
            result.all_inputs = [
                fr.get('test_split', {}).get('features')
                for fr in results.get('fold_results', [])
            ]
            result.best_params = results.get('best_params', {})
            
            # Add naive baseline for nowcasting
            if baseline and baseline.get('naive_mae') is not None and result.aggregate_metrics:
                result.aggregate_metrics['naive_mae'] = baseline['naive_mae']
            
            # Extract per-fold Optuna trial details
            optuna_trials = []
            for fold_res in result.fold_results:
                fold_trials = fold_res.get('optuna_trials', [])
                fold_idx = fold_res.get('fold', 0)
                for trial in fold_trials:
                    trial['fold'] = fold_idx
                    trial['model'] = model_name
                    optuna_trials.append(trial)
            result.optuna_trials = optuna_trials
        
    except Exception as e:
        import traceback
        import gc
        result.error = f"{str(e)}\n{traceback.format_exc()}"
        # Cleanup on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    result.runtime_seconds = time.time() - start_time
    return result


def _train_model_worker(args: tuple) -> ModelResult:
    """Worker function for parallel execution."""
    import gc
    model_name, model_config, dataset_config, eval_config, device, task_type = args

    # Seed this worker process for reproducibility.  Each worker gets a
    # deterministic but distinct seed derived from the global seed + its
    # model name, so parallel runs are both reproducible and independent.
    # NOTE: use zlib.crc32 (not the builtin hash()) for the per-model offset.
    # Python salts str hashing per interpreter unless PYTHONHASHSEED is pinned,
    # so hash(model_name) would make every spawned-worker run irreproducible.
    base_seed = eval_config.get('seed', 42)
    worker_seed = base_seed + zlib.crc32(model_name.encode('utf-8')) % (2**31)
    set_seed(worker_seed)

    # Set device
    if torch.cuda.is_available() and 'cuda' in device:
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        torch.cuda.set_device(gpu_id)
    
    print(f"[{device}] Starting {model_name} ({task_type})")
    sys.stdout.flush()
    
    try:
        result = train_model(model_name, model_config, dataset_config, eval_config, device, task_type)
    finally:
        # Clean up GPU memory after each model to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    status = "✅" if result.error is None else "❌"
    print(f"[{device}] {status} Completed {model_name} ({result.runtime_seconds:.1f}s)")
    sys.stdout.flush()
    
    return result


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Main benchmark runner with parallel execution support.
    
    Supports both forecasting and nowcasting tasks.
    
    Usage:
        runner = BenchmarkRunner(config)
        results = runner.run()
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[ModelResult] = []
        
        # Task type: 'forecast' (default) or 'nowcast'
        self.task_type = config.get('task', 'forecast')
        
        # Output directory
        save_path = config.get('output', {}).get('save_path', './benchmark_results/')
        self.save_dir = Path(save_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Parallel config
        parallel_cfg = config.get('parallel', {})
        self.enable_parallel = parallel_cfg.get('enable', True)
        self.gpu_ids = parallel_cfg.get('gpu_ids')
        self.workers_per_gpu = parallel_cfg.get('workers_per_gpu', 1)
        
        # Auto-detect GPUs
        if self.gpu_ids is None and torch.cuda.is_available():
            self.gpu_ids = list(range(torch.cuda.device_count()))
        elif not torch.cuda.is_available():
            self.gpu_ids = []
        
        # Device setup
        if self.gpu_ids:
            print(f"🚀 Using GPUs: {self.gpu_ids}")
        else:
            print("⚙️  Using CPU")
        
        # Initialize result files (write headers)
        self._init_result_files()
    
    def run(self) -> List[ModelResult]:
        """Run the benchmark for all configured models."""
        seed = self.config.get('evaluation', {}).get('seed', 42)
        set_seed(seed)

        models_cfg = self.config.get('models', [])
        dataset_cfg = self.config.get('dataset', {})
        eval_cfg = self.config.get('evaluation', {})
        
        if not models_cfg:
            raise ValueError("No models configured")

        # Fail fast on missing input data, once, instead of letting every model
        # rediscover it and print an identical traceback.
        if self.task_type == 'nowcast':
            triangle_path = dataset_cfg.get('triangle_path')
            if triangle_path and not Path(triangle_path).expanduser().exists():
                raise FileNotFoundError(
                    f"Reporting triangle not found: {triangle_path}\n"
                    "EpiLearn does not redistribute the COVID-19 triangle this config "
                    "names (it is CMU Delphi Epidata). Regenerate it with:\n"
                    "    python datasets/build_nowcast_triangle.py "
                    f"-o {triangle_path}\n"
                    "or point `dataset.triangle_path` at your own .npz. For a nowcasting "
                    "example that needs no external data, see tests/nowcast.py."
                )
        elif self.task_type == 'forecast':
            feature_path = dataset_cfg.get('feature_path')
            if feature_path and not Path(feature_path).expanduser().exists():
                raise FileNotFoundError(
                    f"Feature CSV not found: {feature_path}\n"
                    "Paths in a config are resolved from the current working directory, "
                    "so run the benchmark from the repository root, or use an absolute path."
                )

        # Pre-check foundation model dependencies once (avoid redundant checks per worker)
        dep_issues = {}
        for model_cfg in models_cfg:
            model_name = model_cfg.get('name', '')
            _cls = model_cfg.get('class_name', model_name)
            if is_foundation_model(_cls):
                err = check_foundation_deps(_cls)
                if err:
                    dep_issues[model_name] = err
        
        if dep_issues:
            print(f"\n⚠️  Foundation model dependency check:")
            for name, err in dep_issues.items():
                print(f"   ⏭️  {name}: {err}")
            print()
        
        # Build job list (include task_type)
        jobs = []
        skipped = []
        for model_cfg in models_cfg:
            model_name = model_cfg.get('name')
            if not model_name:
                continue
            # Skip spatiotemporal models for nowcasting (single-region only)
            if self.task_type == 'nowcast' and model_cfg.get('requires_graph', False):
                print(f"⚠️  Skipping {model_name}: spatiotemporal models not supported for nowcasting")
                continue
            # Skip foundation models with missing dependencies (record as error immediately)
            if model_name in dep_issues:
                requires_graph = model_cfg.get('requires_graph', False)
                skip_result = ModelResult(
                    model_name=model_name,
                    requires_graph=requires_graph,
                    lookback=eval_cfg.get('lookback', 14),
                    horizon=eval_cfg.get('horizon', 7),
                    task_type=self.task_type,
                    evaluation_paradigm='zero-shot',
                    error=f"SKIPPED (missing dependency): {dep_issues[model_name]}",
                )
                self.results.append(skip_result)
                self._save_single_result(skip_result)
                skipped.append(model_name)
                continue
            jobs.append((model_name, model_cfg, dataset_cfg, eval_cfg, self.task_type))
        
        print(f"\n{'='*60}")
        print(f"Running {self.task_type.upper()} benchmark for {len(jobs)} models")
        if skipped:
            print(f"⏭️  Skipped {len(skipped)} models (missing dependencies): {', '.join(skipped)}")
        print(f"📁 Results saving to: {self.save_dir}")
        print(f"{'='*60}\n")

        if self.enable_parallel and len(self.gpu_ids) > 0:
            self._run_parallel(jobs)
        else:
            self._run_sequential(jobs)
        
        # Final summary
        self._print_final_summary()
        
        return self.results
    
    def _run_sequential(self, jobs: List[tuple]):
        """Run models sequentially."""
        device = f'cuda:0' if self.gpu_ids else 'cpu'
        
        for model_name, model_cfg, dataset_cfg, eval_cfg, task_type in jobs:
            result = train_model(model_name, model_cfg, dataset_cfg, eval_cfg, device, task_type)
            self.results.append(result)
            self._print_result(result)
            self._save_single_result(result)  # Save immediately
    
    def _run_parallel(self, jobs: List[tuple]):
        """Run models in parallel across GPUs."""
        n_gpus = len(self.gpu_ids)
        max_workers = min(n_gpus * self.workers_per_gpu, len(jobs))
        
        print(f"🚀 Parallel execution: {max_workers} workers across {n_gpus} GPUs\n")
        
        # Assign devices to jobs (jobs already include task_type)
        job_args = []
        for i, (model_name, model_cfg, dataset_cfg, eval_cfg, task_type) in enumerate(jobs):
            gpu_id = self.gpu_ids[i % n_gpus]
            device = f'cuda:{gpu_id}'
            job_args.append((model_name, model_cfg, dataset_cfg, eval_cfg, device, task_type))
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
            futures = {executor.submit(_train_model_worker, args): args[0] for args in job_args}
            
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                    self.results.append(result)
                    self._print_result(result)
                    self._save_single_result(result)  # Save immediately after each model
                except Exception as e:
                    print(f"❌ {model_name} failed: {e}")
                    error_result = ModelResult(
                        model_name=model_name,
                        requires_graph=False,
                        lookback=0,
                        horizon=0,
                        error=str(e)
                    )
                    self.results.append(error_result)
                    self._save_single_result(error_result)
    
    def _print_result(self, result: ModelResult):
        """Print result summary."""
        if result.error:
            if result.error.startswith('SKIPPED'):
                print(f"⏭️  {result.model_name}: {result.error}")
            else:
                print(f"❌ {result.model_name}: ERROR - {result.error[:200]}")
        else:
            agg = result.aggregate_metrics
            if agg:
                # Scenario task uses PEHE/ATE, others use MSE/MAE
                if result.task_type == 'scenario':
                    pehe = agg.get('pehe_mean', float('nan'))
                    ate = agg.get('ate_error_mean', float('nan'))
                    print(f"✅ {result.model_name}: PEHE={pehe:.4f}, ATE={ate:.4f} [{result.evaluation_paradigm}] ({result.runtime_seconds:.1f}s)")
                else:
                    mse = agg.get('mse_mean', float('nan'))
                    mae = agg.get('mae_mean', float('nan'))
                    print(f"✅ {result.model_name}: MSE={mse:.4f}, MAE={mae:.4f} [{result.evaluation_paradigm}] ({result.runtime_seconds:.1f}s)")
            else:
                print(f"⚠️  {result.model_name}: No metrics (all folds failed) [{result.evaluation_paradigm}] ({result.runtime_seconds:.1f}s)")
        sys.stdout.flush()
    
    def _init_result_files(self):
        """Initialize CSV files with headers."""
        # Summary file - includes both forecasting and scenario metrics
        self.summary_path = self.save_dir / f"benchmark_summary_{self.timestamp}.csv"
        self.summary_fields = [
            'model', 'task_type', 'status', 'lookback', 'horizon', 'n_scenarios', 'requires_graph',
            'evaluation_paradigm',
            # Forecasting/Nowcasting metrics
            'mse_mean', 'mse_std', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std',
            # Scenario metrics
            'pehe_mean', 'pehe_std', 'ate_error_mean', 'ate_error_std',
            # Uncertainty quantification
            'coverage_mean', 'coverage_std', 'interval_width_mean', 'interval_width_std',
            'best_params', 'runtime_seconds', 'outlier_flag', 'error'
        ]
        with self.summary_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_fields)
            writer.writeheader()
        
        # Detailed file - includes both types of metrics
        self.detailed_path = self.save_dir / f"benchmark_detailed_{self.timestamp}.csv"
        self.detailed_fields = [
            'model', 'task_type', 'fold', 
            'mse', 'mae', 'rmse', 'coverage', 'interval_width',  # Forecasting + UQ
            'pehe', 'ate_error',  # Scenario
            'best_params'
        ]
        with self.detailed_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.detailed_fields)
            writer.writeheader()
        
        # Optuna trials file (will be initialized on first write with dynamic fields)
        self.optuna_path = self.save_dir / f"optuna_trials_{self.timestamp}.csv"
        self.optuna_fields_written = False
        
        # Per-model results directory
        self.models_dir = self.save_dir / f"models_{self.timestamp}"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📄 Summary: {self.summary_path}")
        print(f"📄 Detailed: {self.detailed_path}")
        print(f"📁 Per-model results: {self.models_dir}")
    
    def _save_single_result(self, result: ModelResult):
        """Save a single model result immediately (append to CSV files)."""
        # 1. Append to summary
        self._append_summary(result)
        
        # 2. Append detailed fold results
        self._append_detailed(result)
        
        # 3. Append optuna trials
        self._append_optuna_trials(result)
        
        # 4. Save individual model file
        self._save_model_file(result)
    
    def _save_model_file(self, result: ModelResult):
        """Save individual model results to separate CSV files."""
        model_name = result.model_name
        
        # 1. Summary CSV for this model
        summary_file = self.models_dir / f"{model_name}_summary.csv"
        if result.error:
            status = 'skipped_dependency' if result.error.startswith('SKIPPED') else 'error'
            summary_row = {
                'model': model_name,
                'task_type': result.task_type,
                'status': status,
                'evaluation_paradigm': result.evaluation_paradigm,
                'error': result.error[:500],
            }
        else:
            agg = result.aggregate_metrics or {}  # Handle None case
            summary_row = {
                'model': model_name,
                'task_type': result.task_type,
                'status': 'success' if agg else 'no_results',
                'lookback': result.lookback,
                'horizon': result.horizon,
                'n_scenarios': result.n_scenarios if result.task_type == 'scenario' else None,
                'requires_graph': result.requires_graph,
                'evaluation_paradigm': result.evaluation_paradigm,
                # Forecasting/Nowcasting metrics
                'mse_mean': agg.get('mse_mean'),
                'mse_std': agg.get('mse_std'),
                'mae_mean': agg.get('mae_mean'),
                'mae_std': agg.get('mae_std'),
                'rmse_mean': agg.get('rmse_mean'),
                'rmse_std': agg.get('rmse_std'),
                # Scenario metrics
                'pehe_mean': agg.get('pehe_mean'),
                'pehe_std': agg.get('pehe_std'),
                'ate_error_mean': agg.get('ate_error_mean'),
                'ate_error_std': agg.get('ate_error_std'),
                # Uncertainty quantification
                'coverage_mean': agg.get('coverage_mean'),
                'coverage_std': agg.get('coverage_std'),
                'interval_width_mean': agg.get('interval_width_mean'),
                'interval_width_std': agg.get('interval_width_std'),
                'best_params': json.dumps(result.best_params),
                'runtime_seconds': result.runtime_seconds,
            }
        
        with summary_file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_fields)
            writer.writeheader()
            writer.writerow(summary_row)
        
        # 2. Detailed fold results CSV for this model
        if not result.error and result.fold_results:
            detailed_file = self.models_dir / f"{model_name}_detailed.csv"
            rows = []
            for fold in result.fold_results:
                rows.append({
                    'model': model_name,
                    'task_type': result.task_type,
                    'fold': fold.get('fold'),
                    # Forecasting/Nowcasting metrics
                    'mse': fold.get('mse'),
                    'mae': fold.get('mae'),
                    'rmse': fold.get('rmse'),
                    'coverage': fold.get('coverage'),
                    'interval_width': fold.get('interval_width'),
                    # Scenario metrics
                    'pehe': fold.get('pehe'),
                    'ate_error': fold.get('ate_error'),
                    'best_params': json.dumps(fold.get('best_params', {})),
                })
            
            with detailed_file.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.detailed_fields)
                writer.writeheader()
                writer.writerows(rows)
        
        # 3. Per-sample predictions/targets (for re-computing metrics later)
        if not result.error and result.all_predictions and result.all_targets:
            pred_file = self.models_dir / f"{model_name}_predictions.npz"
            save_data = {}
            n_folds = len(result.all_predictions)
            for i in range(n_folds):
                p = result.all_predictions[i]
                t = result.all_targets[i]
                save_data[f'fold_{i}_predictions'] = p.cpu().numpy() if hasattr(p, 'cpu') else np.asarray(p)
                save_data[f'fold_{i}_targets'] = t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)
                # Save test inputs (lookback window) for epidemic metrics and visualization
                if i < len(result.all_inputs) and result.all_inputs[i] is not None:
                    inp = result.all_inputs[i]
                    save_data[f'fold_{i}_inputs'] = inp.cpu().numpy() if hasattr(inp, 'cpu') else np.asarray(inp)
                # Save normalization stats if available (for denormalization)
                fold_stats = result.fold_results[i].get('process_history', {}) if i < len(result.fold_results) else {}
                for key in ('target_mean', 'target_std', 'feat_mean', 'feat_std'):
                    if key in fold_stats:
                        val = fold_stats[key]
                        save_data[f'fold_{i}_{key}'] = val.cpu().numpy() if hasattr(val, 'cpu') else np.asarray(val)
                # Save conformal calibration data for post-hoc ACI
                if i < len(result.fold_results):
                    fr = result.fold_results[i]
                    if 'val_residuals' in fr and fr['val_residuals'] is not None:
                        vr = fr['val_residuals']
                        save_data[f'fold_{i}_val_residuals'] = vr.cpu().numpy() if hasattr(vr, 'cpu') else np.asarray(vr)
                    if 'conformal_quantile' in fr:
                        save_data[f'fold_{i}_conformal_quantile'] = np.array(fr['conformal_quantile'])
            save_data['n_folds'] = np.array(n_folds)
            save_data['conformal_alpha'] = np.array(
                self.config.get('evaluation', {}).get('conformal_alpha', 0.1)
            )
            np.savez_compressed(pred_file, **save_data)

        # 4. Optuna trials CSV for this model (if available)
        if not result.error and result.optuna_trials:
            optuna_file = self.models_dir / f"{model_name}_optuna.csv"
            rows = []
            for trial in result.optuna_trials:
                fold_idx = trial.get('fold', 0)
                trial_num = trial.get('trial_number', 0)
                hyperparams = trial.get('hyperparams', {})
                val_loss_agg = trial.get('val_loss_aggregate', float('nan'))
                
                for node_result in trial.get('node_results', []):
                    row = {
                        'model': model_name,
                        'fold': fold_idx,
                        'trial': trial_num,
                        'node_idx': node_result.get('node_idx'),
                        'node_mse': node_result.get('mse'),
                        'node_mae': node_result.get('mae'),
                        'node_rmse': node_result.get('rmse'),
                        'node_mape': node_result.get('mape'),
                        'val_loss_aggregate': val_loss_agg,
                        'hyperparams': json.dumps(hyperparams),
                    }
                    for hp_key, hp_val in hyperparams.items():
                        row[f'hp_{hp_key}'] = hp_val
                    rows.append(row)
            
            if rows:
                all_keys = sorted(set().union(*[r.keys() for r in rows]))
                with optuna_file.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=all_keys)
                    writer.writeheader()
                    writer.writerows(rows)
        
        print(f"    📄 Saved: {model_name}_*.csv")
        sys.stdout.flush()
    
    def _append_summary(self, result: ModelResult):
        """Append a single result to summary CSV."""
        if result.error:
            status = 'skipped_dependency' if result.error.startswith('SKIPPED') else 'error'
            row = {
                'model': result.model_name,
                'task_type': result.task_type,
                'status': status,
                'evaluation_paradigm': result.evaluation_paradigm,
                'error': result.error[:500],
            }
        else:
            agg = result.aggregate_metrics or {}  # Handle None case
            
            # Compute outlier flag: will be updated in _print_final_summary
            # For now, mark based on available data
            outlier_flag = ''
            
            row = {
                'model': result.model_name,
                'task_type': result.task_type,
                'status': 'success' if agg else 'no_results',
                'lookback': result.lookback,
                'horizon': result.horizon,
                'n_scenarios': result.n_scenarios if result.task_type == 'scenario' else None,
                'requires_graph': result.requires_graph,
                'evaluation_paradigm': result.evaluation_paradigm,
                # Forecasting/Nowcasting metrics
                'mse_mean': agg.get('mse_mean'),
                'mse_std': agg.get('mse_std'),
                'mae_mean': agg.get('mae_mean'),
                'mae_std': agg.get('mae_std'),
                'rmse_mean': agg.get('rmse_mean'),
                'rmse_std': agg.get('rmse_std'),
                # Scenario metrics
                'pehe_mean': agg.get('pehe_mean'),
                'pehe_std': agg.get('pehe_std'),
                'ate_error_mean': agg.get('ate_error_mean'),
                'ate_error_std': agg.get('ate_error_std'),
                # Uncertainty quantification
                'coverage_mean': agg.get('coverage_mean'),
                'coverage_std': agg.get('coverage_std'),
                'interval_width_mean': agg.get('interval_width_mean'),
                'interval_width_std': agg.get('interval_width_std'),
                'best_params': json.dumps(result.best_params),
                'runtime_seconds': result.runtime_seconds,
                'outlier_flag': outlier_flag,
            }
        
        with self.summary_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_fields)
            writer.writerow(row)
    
    def _append_detailed(self, result: ModelResult):
        """Append fold results to detailed CSV."""
        if result.error:
            return
        
        rows = []
        for fold in result.fold_results:
            rows.append({
                'model': result.model_name,
                'task_type': result.task_type,
                'fold': fold.get('fold'),
                # Forecasting/Nowcasting metrics
                'mse': fold.get('mse'),
                'mae': fold.get('mae'),
                'rmse': fold.get('rmse'),
                'coverage': fold.get('coverage'),
                'interval_width': fold.get('interval_width'),
                # Scenario metrics
                'pehe': fold.get('pehe'),
                'ate_error': fold.get('ate_error'),
                'best_params': json.dumps(fold.get('best_params', {})),
            })
        
        if rows:
            with self.detailed_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.detailed_fields)
                writer.writerows(rows)
    
    def _append_optuna_trials(self, result: ModelResult):
        """Append optuna trial results to CSV."""
        if result.error or not result.optuna_trials:
            return
        
        rows = []
        for trial in result.optuna_trials:
            fold_idx = trial.get('fold', 0)
            trial_num = trial.get('trial_number', 0)
            hyperparams = trial.get('hyperparams', {})
            val_loss_agg = trial.get('val_loss_aggregate', float('nan'))
            
            for node_result in trial.get('node_results', []):
                row = {
                    'model': result.model_name,
                    'fold': fold_idx,
                    'trial': trial_num,
                    'node_idx': node_result.get('node_idx'),
                    'node_mse': node_result.get('mse'),
                    'node_mae': node_result.get('mae'),
                    'node_rmse': node_result.get('rmse'),
                    'node_mape': node_result.get('mape'),
                    'val_loss_aggregate': val_loss_agg,
                    'train_start_idx': node_result.get('train_start_idx'),
                    'train_end_idx': node_result.get('train_end_idx'),
                    'val_start_idx': node_result.get('val_start_idx'),
                    'val_end_idx': node_result.get('val_end_idx'),
                    'hyperparams': json.dumps(hyperparams),
                }
                for hp_key, hp_val in hyperparams.items():
                    row[f'hp_{hp_key}'] = hp_val
                rows.append(row)
        
        if rows:
            # Get all keys for this batch
            all_keys = set()
            for row in rows:
                all_keys.update(row.keys())
            all_keys = sorted(all_keys)
            
            # Write header only on first write
            mode = 'a' if self.optuna_fields_written else 'w'
            with self.optuna_path.open(mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                if not self.optuna_fields_written:
                    writer.writeheader()
                    self.optuna_fields_written = True
                writer.writerows(rows)
    
    def _print_final_summary(self):
        """Print final summary after all models complete."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE ({self.task_type.upper()})")
        print(f"{'='*60}")
        
        # A model whose every fold failed carries error=None but no metrics
        # (status='no_results'), so it must not be counted as a success.
        success = sum(1 for r in self.results if r.error is None and r.aggregate_metrics)
        print(f"✅ Success: {success}/{len(self.results)}")
        
        if success > 0:
            valid_results = [r for r in self.results if r.error is None and r.aggregate_metrics]
            if valid_results:
                # Use appropriate metric based on task type
                if self.task_type == 'scenario':
                    best = min(valid_results, key=lambda r: r.aggregate_metrics.get('pehe_mean', float('inf')))
                    pehe = best.aggregate_metrics.get('pehe_mean', 0)
                    print(f"🏆 Best Model: {best.model_name} (PEHE: {pehe:.4f})")
                else:
                    best = min(valid_results, key=lambda r: r.aggregate_metrics.get('mse_mean', float('inf')))
                    mse = best.aggregate_metrics.get('mse_mean', 0)
                    print(f"🏆 Best Model: {best.model_name} (MSE: {mse:.4f})")
                
                # Outlier detection: flag models with MSE > 10× median (Fix #5)
                self._detect_outlier_models(valid_results)
        
        # Warn about temporal model single-node flattening (Fix #8)
        temporal_on_graph = [r for r in self.results if r.error is None and not r.requires_graph
                            and r.aggregate_metrics]
        graph_models = [r for r in self.results if r.error is None and r.requires_graph
                       and r.aggregate_metrics]
        
        print(f"\n📁 Results saved to: {self.save_dir}")
        print(f"   - {self.summary_path.name}")
        print(f"   - {self.detailed_path.name}")
        if self.optuna_fields_written:
            print(f"   - {self.optuna_path.name}")
        n_model_files = len(list(self.models_dir.glob('*_summary.csv')))
        print(f"   - {self.models_dir.name}/ ({n_model_files} models, each with _summary/_detailed/_optuna.csv)")
    
    def _detect_outlier_models(self, valid_results: List[ModelResult]):
        """Detect and flag models with extreme outlier performance (MSE > 10× median)."""
        # Determine primary metric based on task type
        if self.task_type == 'scenario':
            metric_key = 'pehe_mean'
        else:
            metric_key = 'mse_mean'
        
        metric_values = []
        for r in valid_results:
            val = r.aggregate_metrics.get(metric_key)
            if val is not None and np.isfinite(val):
                metric_values.append((r.model_name, val))
        
        if len(metric_values) < 3:
            return
        
        values = [v for _, v in metric_values]
        median_val = float(np.median(values))
        threshold = median_val * 10
        
        outliers = [(name, val) for name, val in metric_values if val > threshold]
        if outliers:
            print(f"\n🚩 OUTLIER WARNING: {len(outliers)} model(s) have {metric_key} > 10× median ({median_val:.2f}):")
            for name, val in sorted(outliers, key=lambda x: -x[1]):
                print(f"   ⚠️  {name}: {metric_key}={val:.2f} ({val/median_val:.0f}× median) — may indicate convergence failure")
            print(f"   Consider excluding these from aggregate analyses.")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='EpiLearn Benchmark Runner')
    parser.add_argument('--config', '-c', required=True, help='Path to config YAML file')
    parser.add_argument('--output', '-o', help='Output directory (overrides config)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override output if specified
    if args.output:
        config.setdefault('output', {})['save_path'] = args.output
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()

    # Summary is already printed by _print_final_summary(). Exit non-zero when no
    # model produced metrics, so a failing run cannot look like a success to CI or
    # to a wrapper script. Models skipped for a missing optional dependency do not
    # count as failures as long as something else succeeded.
    succeeded = [r for r in (results or []) if r.error is None and r.aggregate_metrics]
    if results and not succeeded:
        print("\n❌ No model produced any metrics -- see the errors above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


# =============================================================================
# Post-hoc Evaluation Utilities
# =============================================================================

def load_predictions(results_dir: str, model_name: str, timestamp: str = None):
    """
    Load saved per-sample predictions and targets for a model.

    Parameters
    ----------
    results_dir : str
        Path to the benchmark results directory (e.g., './benchmark_results/nowcast/').
    model_name : str
        Name of the model (e.g., 'GRUModel').
    timestamp : str, optional
        Benchmark run timestamp (e.g., '20260209_191645'). If None, uses the
        most recent run.

    Returns
    -------
    dict with keys:
        'folds': list of dicts, each with 'predictions' and 'targets' (numpy arrays)
        'n_folds': int
        'process_history': list of dicts (normalization stats per fold, if available)
    """
    results_dir = Path(results_dir)

    # Find the models directory
    if timestamp:
        models_dir = results_dir / f"models_{timestamp}"
    else:
        # Find most recent
        dirs = sorted(results_dir.glob("models_*"), key=lambda p: p.name)
        if not dirs:
            raise FileNotFoundError(f"No models_* directories found in {results_dir}")
        models_dir = dirs[-1]

    pred_file = models_dir / f"{model_name}_predictions.npz"
    if not pred_file.exists():
        raise FileNotFoundError(f"No predictions file: {pred_file}")

    data = np.load(pred_file, allow_pickle=True)
    n_folds = int(data['n_folds'])

    folds = []
    process_history = []
    for i in range(n_folds):
        fold = {
            'predictions': data[f'fold_{i}_predictions'],
            'targets': data[f'fold_{i}_targets'],
        }
        k_inputs = f'fold_{i}_inputs'
        if k_inputs in data:
            fold['inputs'] = data[k_inputs]
        # Conformal calibration data (for post-hoc ACI)
        k_vr = f'fold_{i}_val_residuals'
        if k_vr in data:
            fold['val_residuals'] = data[k_vr]
        k_cq = f'fold_{i}_conformal_quantile'
        if k_cq in data:
            fold['conformal_quantile'] = float(data[k_cq])
        folds.append(fold)
        stats = {}
        for key in ('target_mean', 'target_std', 'feat_mean', 'feat_std'):
            k = f'fold_{i}_{key}'
            if k in data:
                stats[key] = data[k]
        process_history.append(stats)

    return {
        'folds': folds,
        'n_folds': n_folds,
        'process_history': process_history,
        'conformal_alpha': float(data['conformal_alpha']) if 'conformal_alpha' in data else 0.1,
    }


# Built-in metric registry (numpy-based for evaluate_from_saved)
# Metrics are stored as (fn, needs_inputs) tuples.
#   needs_inputs=False → fn(pred, target)
#   needs_inputs=True  → fn(pred, target, inputs)  (inputs may be None)
_BUILTIN_METRICS = {}

def _register_builtins():
    _BUILTIN_METRICS['mse'] = (lambda p, t: float(np.mean((p - t) ** 2)), False)
    _BUILTIN_METRICS['mae'] = (lambda p, t: float(np.mean(np.abs(p - t))), False)
    _BUILTIN_METRICS['rmse'] = (lambda p, t: float(np.sqrt(np.mean((p - t) ** 2))), False)
    _BUILTIN_METRICS['mape'] = (lambda p, t: float(np.mean(np.abs((t - p) / np.clip(np.abs(t), 1e-8, None)))) * 100, False)
    _BUILTIN_METRICS['r2'] = (lambda p, t: float(1 - np.sum((t - p) ** 2) / np.clip(np.sum((t - np.mean(t)) ** 2), 1e-8, None)), False)
    _BUILTIN_METRICS['nrmse'] = (lambda p, t: float(np.sqrt(np.mean((p - t) ** 2)) / max(np.std(t), 1e-8)), False)

    try:
        from epilearn.utils.epidemic_metrics import (
            compute_outbreak_recall,
            compute_alert_sensitivity,
            compute_peak_underestimate_rate,
            compute_rising_phase_mae,
            compute_trend_accuracy,
        )
        _BUILTIN_METRICS['outbreak_recall'] = (lambda p, t, inp: compute_outbreak_recall(p, t, inputs=inp), True)
        _BUILTIN_METRICS['alert_sensitivity'] = (lambda p, t, inp: compute_alert_sensitivity(p, t, inputs=inp), True)
        _BUILTIN_METRICS['peak_underestimate'] = (lambda p, t, inp: compute_peak_underestimate_rate(p, t, inputs=inp), True)
        _BUILTIN_METRICS['rising_phase_mae'] = (lambda p, t, inp: compute_rising_phase_mae(p, t, inputs=inp), True)
        _BUILTIN_METRICS['trend_accuracy'] = (lambda p, t, inp: compute_trend_accuracy(p, t, inputs=inp), True)
    except ImportError:
        pass


def evaluate_from_saved(
    results_dir: str,
    model_name: str = None,
    metrics: list = None,
    timestamp: str = None,
    denormalize: bool = False,
):
    """
    Re-compute metrics from saved benchmark predictions.

    Parameters
    ----------
    results_dir : str
        Path to the benchmark results directory.
    model_name : str, optional
        Specific model to evaluate. If None, evaluates all models found.
    metrics : list, optional
        Metric names (str) or callables.
        - Standard callables: ``f(pred, target) -> float``
        - Context-aware callables: ``f(pred, target, inputs) -> float``
        Built-in names: 'mse', 'mae', 'rmse', 'mape', 'r2',
        'outbreak_recall', 'alert_sensitivity', 'peak_underestimate',
        'rising_phase_mae', 'trend_accuracy'.
        Default: ['mse', 'mae', 'rmse'].
    timestamp : str, optional
        Benchmark run timestamp. If None, uses most recent.
    denormalize : bool
        If True and normalization stats are available, inverse-transform
        predictions and targets to original scale before computing metrics.

    Returns
    -------
    dict
        ``{model_name: {'metric_mean': float, 'metric_std': float, ...}}``
        or a single model's dict if ``model_name`` is specified.
    """
    if not _BUILTIN_METRICS:
        _register_builtins()

    if metrics is None:
        metrics = ['mse', 'mae', 'rmse']

    results_dir = Path(results_dir)

    # Discover models
    if model_name:
        model_names = [model_name]
    else:
        if timestamp:
            models_dir = results_dir / f"models_{timestamp}"
        else:
            dirs = sorted(results_dir.glob("models_*"), key=lambda p: p.name)
            models_dir = dirs[-1] if dirs else None
        if models_dir is None or not models_dir.exists():
            raise FileNotFoundError(f"No models directory found in {results_dir}")
        model_names = sorted(set(
            p.stem.replace('_predictions', '')
            for p in models_dir.glob('*_predictions.npz')
        ))

    all_results = {}
    for mname in model_names:
        try:
            saved = load_predictions(results_dir, mname, timestamp)
        except FileNotFoundError:
            continue

        per_fold_metrics = {m if isinstance(m, str) else m.__name__: [] for m in metrics}

        for i, fold in enumerate(saved['folds']):
            pred = fold['predictions'].astype(np.float64)
            target = fold['targets'].astype(np.float64)
            inputs = fold.get('inputs')  # May be None for older results
            if inputs is not None:
                inputs = inputs.astype(np.float64)

            # Optional denormalization
            if denormalize and i < len(saved['process_history']):
                stats = saved['process_history'][i]
                if 'target_mean' in stats and 'target_std' in stats:
                    pred = pred * float(stats['target_std']) + float(stats['target_mean'])
                    target = target * float(stats['target_std']) + float(stats['target_mean'])

            # Filter invalid values (keep inputs unfiltered for context)
            valid = np.isfinite(pred) & np.isfinite(target)
            if valid.ndim > 1:
                valid = valid.all(axis=tuple(range(1, valid.ndim)))
            pred_v, target_v = pred[valid], target[valid]
            inputs_v = inputs[valid] if inputs is not None else None
            if len(pred_v) == 0:
                continue

            for m in metrics:
                if isinstance(m, str):
                    entry = _BUILTIN_METRICS.get(m)
                    key = m
                    if entry is None:
                        continue
                    fn, needs_inputs = entry
                else:
                    fn = m
                    key = m.__name__
                    # Detect if custom callable accepts 3 args
                    import inspect
                    sig = inspect.signature(fn)
                    needs_inputs = len(sig.parameters) >= 3
                try:
                    if needs_inputs:
                        val = fn(pred_v, target_v, inputs_v)
                    else:
                        val = fn(pred_v.flatten(), target_v.flatten())
                    if val is not None and np.isfinite(val):
                        per_fold_metrics[key].append(float(val))
                except Exception:
                    pass

        model_result = {}
        for key, values in per_fold_metrics.items():
            if values:
                model_result[f'{key}_mean'] = float(np.mean(values))
                model_result[f'{key}_std'] = float(np.std(values))
                model_result[f'{key}_per_fold'] = values
            else:
                model_result[f'{key}_mean'] = None
                model_result[f'{key}_std'] = None
        all_results[mname] = model_result

    if model_name:
        return all_results.get(model_name, {})
    return all_results


# =============================================================================
# Uncertainty / Conformal Prediction  (re-exported from epilearn.utils.uncertainty)
# =============================================================================
from epilearn.utils.uncertainty import (          # noqa: E402, F401
    compute_aci,
    compute_uncertainty_metrics,
    evaluate_aci_from_saved,
    winkler_score as _winkler_score,
    static_conformal,
    locally_weighted_conformal,
    locally_weighted_aci,
    difficulty_from_predictions,
    difficulty_reference_stats,
)


__all__ = [
    'BenchmarkRunner', 'ModelResult', 'load_config', 'train_model',
    'load_predictions', 'evaluate_from_saved',
    'compute_aci', 'compute_uncertainty_metrics', 'evaluate_aci_from_saved',
    'static_conformal', 'locally_weighted_conformal', 'locally_weighted_aci',
    'difficulty_from_predictions', 'difficulty_reference_stats',
]
