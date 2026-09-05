"""Trainable ensemble model for epidemic forecasting.

Provides TrainableEnsembleModel: trains base epilearn models from raw Datasets,
then fits an ensemble meta-learner on their held-out predictions.

Typical usage
-------------
    from epilearn.ensemble import TrainableEnsembleModel, make_trainable_ensemble

    ensemble = make_trainable_ensemble(
        result, model_hps=model_hps, lookback=16, horizon=4)
    ensemble.fit(train_dataset)
    out = ensemble.evaluate(test_dataset)
    print(f"NRMSE: {out['nrmse']:.4f}")
"""

import warnings

import numpy as np


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_nrmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute NRMSE = RMSE / std(targets).

    Parameters
    ----------
    predictions : array (n_samples, horizon)
    targets     : array (n_samples, horizon)

    Returns
    -------
    float  Scale-invariant normalised RMSE.
    """
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    return float(rmse / max(np.std(targets), 1e-8))


# ── Model class registry ────────────────────────────────────────────────────


def _resolve_model_class(model_name: str):
    """Return the epilearn model class for *model_name* (e.g. 'GRUModel')."""
    import epilearn.models.Temporal as T
    cls = getattr(T, model_name, None)
    if cls is None:
        raise ValueError(
            f"Model class '{model_name}' not found in epilearn.models.Temporal. "
            f"Available: {[n for n in dir(T) if n.endswith('Model')]}"
        )
    return cls


# ── HP key classification ───────────────────────────────────────────────────

_OPTIMIZER_KEYS = {'epochs', 'lr', 'batch_size', 'weight_decay', 'patience'}

# Models that need the spatial (4D) data format with graph adjacency.
_GRAPH_MODELS = {
    'iTransformerModel', 'FreTSModel',
    'STGCN', 'DCRNN', 'GraphWaveNet', 'ATMGNN',
    'EpiGNN', 'ColaGNN', 'DSTGCN', 'MepoGNN',
}


def _split_hps(hps: dict) -> tuple:
    """Split HPs into (model_args, optimizer_kwargs)."""
    model_args      = {k: v for k, v in hps.items()
                       if k not in _OPTIMIZER_KEYS and k != 'lookback'}
    optimizer_kwargs = {k: v for k, v in hps.items() if k in _OPTIMIZER_KEYS}
    return model_args, optimizer_kwargs


# ── Core class ───────────────────────────────────────────────────────────────


class TrainableEnsembleModel:
    """Full ensemble pipeline: trains base epilearn models then fits meta-learner.

    Parameters
    ----------
    strategy           : ensemble strategy object with fit/predict methods
    base_model_configs : list of dicts, each with keys:
                           'name'        : str  (e.g. 'GRUModel')
                           'model_class' : type (epilearn model class)
                           'hps'         : dict (suggested hyperparameters)
    lookback           : int — input context window (timesteps)
    horizon            : int — forecast horizon
    device             : str — 'cpu' or 'cuda'
    """

    def __init__(self, strategy, base_model_configs: list,
                 lookback: int, horizon: int, device: str = 'cpu'):
        self.strategy           = strategy
        self.base_model_configs = base_model_configs
        self.lookback           = lookback
        self.horizon            = horizon
        self.device             = device
        self.name               = getattr(strategy, 'name', type(strategy).__name__)
        self._trained_tasks     = []   # list of Forecast tasks after fit()

    # ── Public API ───────────────────────────────────────────────────────

    def fit(self, train_dataset, graph=None) -> 'TrainableEnsembleModel':
        """Train each base model on *train_dataset*, then fit meta-learner.

        Uses an 80/20 internal split: base models train on 80%, meta-learner
        fits on 20%'s out-of-sample predictions (avoids train-set leakage).
        """
        import torch
        from epilearn.tasks.forecast import Forecast

        self._trained_tasks = []
        all_preds = []
        targets   = None

        for cfg in self.base_model_configs:
            lookback  = cfg['hps'].get('lookback', self.lookback)
            task = Forecast(
                prototype = cfg['model_class'],
                lookback  = lookback,
                horizon   = self.horizon,
                device    = self.device,
            )

            requires_graph = cfg.get('requires_graph', False)
            ds = _strip_graph(train_dataset) if not requires_graph else train_dataset

            full_split = task._generate_split(ds)
            if (not requires_graph
                    and full_split['features'].ndim == 4
                    and full_split.get('graph') is None):
                _, lb, n_nodes, n_feats = full_split['features'].shape
                task._reshape_4d_to_temporal(full_split, lb, n_feats)

            # 80 / 20 train / val split for early stopping
            n       = full_split['features'].shape[0]
            val_n   = max(2, int(n * 0.2))
            train_n = n - val_n

            tr = _slice_split(full_split, 0, train_n)
            va = _slice_split(full_split, train_n, n)

            model_args, opt_kw = _split_hps(cfg['hps'])

            task._init_model_from_split(tr, model_args)
            task.model.fit(
                train_input          = tr['features'],
                train_target         = tr['targets'],
                train_graph          = tr.get('graph'),
                train_dynamic_graph  = tr.get('dynamic_graph'),
                val_input            = va['features'],
                val_target           = va['targets'],
                val_graph            = va.get('graph'),
                val_dynamic_graph    = va.get('dynamic_graph'),
                epochs               = opt_kw.get('epochs', 150),
                lr                   = opt_kw.get('lr', 1e-3),
                batch_size           = opt_kw.get('batch_size', 32),
                weight_decay         = opt_kw.get('weight_decay', 0.0),
                patience             = opt_kw.get('patience', 15),
                verbose              = False,
            )

            # Predict on val set (out-of-sample) to fit meta-learner
            with torch.no_grad():
                raw = task.model.predict(
                    feature       = va['features'],
                    graph         = va.get('graph'),
                    dynamic_graph = va.get('dynamic_graph'),
                )
            preds = _extract_preds(raw)
            all_preds.append(preds)

            if targets is None:
                t = va['targets'].detach().cpu().numpy()
                targets = t.reshape(-1, t.shape[-1]) if t.ndim == 3 else t

            self._trained_tasks.append(task)
            print(f"  [TrainableEnsemble] {cfg['name']} trained  "
                  f"({preds.shape[0]} samples, lookback={lookback})")

        # Align to minimum sample count (different lookbacks → different counts)
        min_n = min(p.shape[0] for p in all_preds)
        all_preds = [p[-min_n:] for p in all_preds]

        if targets.ndim == 3:
            targets = targets.reshape(-1, targets.shape[-1])
        targets = targets[-min_n:]

        predictions = np.stack(all_preds)    # (n_models, min_n, horizon)
        self.strategy.fit(predictions, targets)

        # Store val predictions/targets for strategy comparison
        self._val_predictions = predictions
        self._val_targets = targets
        return self

    def predict(self, test_dataset, graph=None) -> np.ndarray:
        """Return ensemble predictions (n_samples, horizon) for *test_dataset*.

        Requires ``fit()`` to have been called first.
        """
        import torch

        if not self._trained_tasks:
            raise RuntimeError("Call fit() before predict().")

        all_preds = []
        for task, cfg in zip(self._trained_tasks, self.base_model_configs):
            requires_graph = cfg.get('requires_graph', False)
            ds = _strip_graph(test_dataset) if not requires_graph else test_dataset
            test_split = task._generate_split(ds)
            if (not requires_graph
                    and test_split['features'].ndim == 4
                    and test_split.get('graph') is None):
                _, lb, _, n_feats = test_split['features'].shape
                task._reshape_4d_to_temporal(test_split, lb, n_feats)
            with torch.no_grad():
                raw = task.model.predict(
                    feature       = test_split['features'],
                    graph         = test_split.get('graph'),
                    dynamic_graph = test_split.get('dynamic_graph'),
                )
            all_preds.append(_extract_preds(raw))

        min_n = min(p.shape[0] for p in all_preds)
        all_preds = [p[-min_n:] for p in all_preds]

        predictions = np.stack(all_preds)    # (n_models, min_n, horizon)
        return self.strategy.predict(predictions)

    def evaluate(self, test_dataset, graph=None) -> dict:
        """Compute NRMSE on *test_dataset*.

        Returns dict with keys: 'nrmse', 'predictions', 'targets'
        """
        pred = self.predict(test_dataset, graph)
        # Get targets from the model with the largest lookback (fewest samples)
        max_lb_idx = max(
            range(len(self.base_model_configs)),
            key=lambda i: self.base_model_configs[i]['hps'].get(
                'lookback', self.lookback),
        )
        cfg0            = self.base_model_configs[max_lb_idx]
        requires_graph0 = cfg0.get('requires_graph', False)
        ds0             = _strip_graph(test_dataset) if not requires_graph0 else test_dataset
        test_split      = self._trained_tasks[max_lb_idx]._generate_split(ds0)
        if (not requires_graph0
                and test_split['features'].ndim == 4
                and test_split.get('graph') is None):
            _, lb, _, n_feats = test_split['features'].shape
            self._trained_tasks[max_lb_idx]._reshape_4d_to_temporal(
                test_split, lb, n_feats)
        targets = test_split['targets'].detach().cpu().numpy()
        if targets.ndim == 3:
            targets = targets.reshape(-1, targets.shape[-1])
        nrmse = compute_nrmse(pred, targets)
        return {'nrmse': nrmse, 'predictions': pred, 'targets': targets}

    def get_weights(self):
        return self.strategy.get_weights()

    def __repr__(self):
        n = len(self.base_model_configs)
        return (f"TrainableEnsembleModel(strategy={self.name!r}, "
                f"n_base={n}, lookback={self.lookback}, horizon={self.horizon})")


# ── Factory ──────────────────────────────────────────────────────────────────


def make_trainable_ensemble(
    strategy_name_or_result,
    model_hps: dict,
    lookback: int,
    horizon: int,
    device: str = 'cpu',
) -> TrainableEnsembleModel:
    """Build a TrainableEnsembleModel from a suggest() result or strategy name.

    Parameters
    ----------
    strategy_name_or_result : str or suggest() dict
        If dict, uses the 'best_strategy' key.
    model_hps : dict
        Per-model HP dict, e.g. ``{'GRUModel': {'lookback': 16, 'lr': 0.001}, ...}``.
    lookback  : int — default context window (overridden per-model if 'lookback' in hps)
    horizon   : int — forecast horizon
    device    : str
    """
    # Lazy imports to avoid circular dependency at module load time
    from .recommender import _build_configs
    from .strategies import MedianEnsemble

    if isinstance(strategy_name_or_result, dict):
        name = strategy_name_or_result.get('best_strategy', 'Median (all)')
    else:
        name = str(strategy_name_or_result)

    factory_map = {desc: factory for factory, desc in _build_configs()}
    factory = factory_map.get(name)
    if factory is None:
        warnings.warn(f"Unknown strategy '{name}'. Falling back to Median.")
        strategy = MedianEnsemble()
    else:
        strategy = factory()

    # Build base model configs from model_hps
    base_model_configs = []
    for model_name, hps in model_hps.items():
        try:
            cls = _resolve_model_class(model_name)
        except ValueError as e:
            warnings.warn(str(e))
            continue
        base_model_configs.append({
            'name':           model_name,
            'model_class':    cls,
            'hps':            hps,
            'requires_graph': model_name in _GRAPH_MODELS,
        })

    if not base_model_configs:
        raise ValueError("No valid base models found in model_hps.")

    return TrainableEnsembleModel(strategy, base_model_configs,
                                  lookback, horizon, device)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _strip_graph(dataset):
    """Return a copy of *dataset* with graph set to None.

    When a dataset has a graph, _generate_split creates 4D features
    (n_windows, lookback, n_nodes, n_features) and BaseTask does NOT
    reshape them for non-spatial models.  Removing the graph triggers the
    4D→3D reshape: (n_windows*n_nodes, lookback, n_features), which is the
    correct format for channel-independent temporal and sklearn models.
    """
    from epilearn.data import Dataset
    return Dataset(x=dataset.x, y=dataset.y, graph=None,
                   dynamic_graph=None, states=dataset.states
                   if hasattr(dataset, 'states') else None)


def _slice_split(split: dict, start: int, end: int) -> dict:
    """Slice all tensor values in a split dict along the sample dimension."""
    import torch
    return {
        k: v[start:end] if isinstance(v, torch.Tensor) else v
        for k, v in split.items()
    }


def _extract_preds(raw) -> np.ndarray:
    """Normalise model output to np.ndarray of shape (n_samples, horizon)."""
    import torch
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    if isinstance(raw, dict):
        raw = raw.get('predictions', raw)
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    # Flatten spatial dimension if present: (n_windows, n_nodes, horizon) → (n_samples, horizon)
    if raw.ndim == 3:
        raw = raw.reshape(-1, raw.shape[-1])
    return raw
