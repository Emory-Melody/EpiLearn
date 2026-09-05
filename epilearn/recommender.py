"""L2E recommender pipeline for epidemic forecasting ensembles.

Two recommenders, both training-free at deployment:
1. StrategyRecommender: (input_series_features, strategy) → ranking score
2. HPRecommender: (timeseries_features, hp_values) → val_loss via per-model Ridge

Training (offline, once):
    rec = L2ERecommender()
    rec.train(data, optuna_dir, fold_regimes, train_folds=[0,1,2,3])
    rec.save('recommender.pkl')

Deployment (training-free):
    rec = L2ERecommender.load('recommender.pkl')
    result = rec.suggest(observed_series)
    print_rankings(result)
"""

import glob
import os
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .ensemble import compute_nrmse
from .regime import compute_regime_features, classify_regime
from .strategies import (
    EnsembleModel,
    MedianEnsemble,
    MeanEnsemble,
    TrimmedMeanEnsemble,
    InverseNRMSEEnsemble,
    StackingEnsemble,
    LassoStackingEnsemble,
    ElasticNetStackingEnsemble,
    ConstrainedStackingEnsemble,
    GBStackingEnsemble,
    RegimeStackingEnsemble,
    TopKMedianEnsemble,
    TopKMeanEnsemble,
    ExpWeightedEnsemble,
)


# ── Constants ──────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'growth_rate', 'volatility', 'level_norm', 'acceleration',
    'ac1', 'skewness', 'kurtosis', 'near_peak_frac',
]

META_STRATEGIES = {'L2E-auto', 'Recommender'}


# ── Feature extraction ─────────────────────────────────────────────────

def _extract_features(series: np.ndarray) -> np.ndarray:
    """Compute 8-dimensional feature vector from a 1D time series."""
    feats = compute_regime_features(series)
    vec = np.array([feats.get(f, 0.0) for f in FEATURE_NAMES])
    return np.nan_to_num(vec, nan=0.0)


def _detect_regime(features_dict: dict, level_q15: float, level_q85: float,
                   vol_q75: float) -> str:
    """Detect regime from pre-computed feature dict using stored thresholds."""
    return classify_regime(
        features_dict.get('growth_rate', 0),
        features_dict.get('volatility', 0),
        features_dict.get('level', 0),
        level_q15, level_q85, vol_q75,
    )


def _subset_data(data: dict, fold_indices: list) -> dict:
    """Create a data dict with only the specified folds (re-indexed from 0)."""
    return {
        'models': data['models'],
        'n_folds': len(fold_indices),
        'n_nodes': data['n_nodes'],
        'horizon': data['horizon'],
        'folds': [data['folds'][fi] for fi in fold_indices],
    }


# ── Strategy configuration ────────────────────────────────────────────

def _build_configs():
    """Return list of (strategy_factory, description) tuples.

    Each factory is a callable returning a fresh strategy instance.
    """
    return [
        (lambda: MedianEnsemble(), 'Median (all)'),
        (lambda: MeanEnsemble(), 'Mean (all)'),
        (lambda: TrimmedMeanEnsemble(trim_frac=0.1), 'TrimmedMean10 (all)'),
        (lambda: InverseNRMSEEnsemble(), 'InvNRMSE (all)'),
        (lambda: StackingEnsemble(alpha=0.1), 'Stacking a=0.1'),
        (lambda: StackingEnsemble(alpha=1.0), 'Stacking a=1'),
        (lambda: StackingEnsemble(alpha=10.0), 'Stacking a=10'),
        (lambda: LassoStackingEnsemble(alpha=0.01), 'LassoStack a=0.01'),
        (lambda: LassoStackingEnsemble(alpha=0.1), 'LassoStack a=0.1'),
        (lambda: ElasticNetStackingEnsemble(alpha=0.1, l1_ratio=0.5), 'ElasticStack a=0.1'),
        (lambda: ConstrainedStackingEnsemble(), 'ConstrainedStack'),
        (lambda: GBStackingEnsemble(n_estimators=50), 'GBStack n=50'),
        (lambda: RegimeStackingEnsemble(alpha=1.0), 'RegimeStack'),
        # Model-selection strategies
        (lambda: TopKMedianEnsemble(k=5), 'TopKMedian k=5'),
        (lambda: TopKMedianEnsemble(k=10), 'TopKMedian k=10'),
        (lambda: TopKMedianEnsemble(k=15), 'TopKMedian k=15'),
        (lambda: TopKMeanEnsemble(k=5), 'TopKMean k=5'),
        (lambda: TopKMeanEnsemble(k=10), 'TopKMean k=10'),
        # Exponential weighting
        (lambda: ExpWeightedEnsemble(beta=3.0), 'ExpWeighted b=3'),
        (lambda: ExpWeightedEnsemble(beta=5.0), 'ExpWeighted b=5'),
        (lambda: ExpWeightedEnsemble(beta=10.0), 'ExpWeighted b=10'),
    ]


# ── L2E LOFO evaluation ───────────────────────────────────────────────

class L2E:
    """Learning-to-Ensemble: LOFO cross-validation over ensemble strategies.

    For each test fold k:
      1. Fit strategy on all folds != k
      2. Predict on fold k → test NRMSE
    """

    def __init__(self, fold_regimes: Optional[dict] = None, n_jobs: int = -1):
        self.fold_regimes = fold_regimes or {}
        self.configs = _build_configs()
        self.n_jobs = n_jobs

    def run(self, data: dict, verbose: bool = True) -> dict:
        """Run full LOFO evaluation. Returns strategies, individual_models, oracle_best."""
        n_folds = data['n_folds']
        models = data['models']

        if verbose:
            print(f"\n{'='*60}")
            print(f"L2E Pipeline — {n_folds} folds, {len(models)} models")
            print(f"{'='*60}")

        individual = self._compute_individual_nrmse(data)
        oracle = self._compute_oracle_best(data, individual)

        from joblib import Parallel, delayed

        def _eval_one(factory, desc):
            return desc, self._lofo_evaluate_strategy(
                data, factory, desc, regime_aware='Regime' in desc)

        results_list = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(_eval_one)(factory, desc) for factory, desc in self.configs)
        strategies = {}
        for desc, result in results_list:
            strategies[desc] = result
            if verbose:
                print(f"  {desc:25s}  NRMSE={result['mean_nrmse']:.4f} "
                      f"± {result['std_nrmse']:.4f}")

        return {'strategies': strategies, 'individual_models': individual,
                'oracle_best': oracle}

    def _compute_individual_nrmse(self, data: dict) -> dict:
        models, n_folds = data['models'], data['n_folds']
        individual = {}
        for mi, model in enumerate(models):
            fold_nrmses = [
                compute_nrmse(data['folds'][fi]['predictions'][mi],
                              data['folds'][fi]['targets'])
                for fi in range(n_folds)
            ]
            individual[model] = {
                'per_fold_nrmse': fold_nrmses,
                'mean_nrmse': float(np.mean(fold_nrmses)),
                'std_nrmse': float(np.std(fold_nrmses)),
            }
        return individual

    def _compute_oracle_best(self, data: dict, individual: dict) -> dict:
        models, n_folds = data['models'], data['n_folds']
        oracle_nrmses, oracle_models = [], []
        for fi in range(n_folds):
            best_model = min(models,
                             key=lambda m: individual[m]['per_fold_nrmse'][fi])
            oracle_nrmses.append(individual[best_model]['per_fold_nrmse'][fi])
            oracle_models.append(best_model)
        return {
            'per_fold_nrmse': oracle_nrmses,
            'mean_nrmse': float(np.mean(oracle_nrmses)),
            'std_nrmse': float(np.std(oracle_nrmses)),
            'selected_models': oracle_models,
        }

    def _lofo_evaluate_strategy(self, data: dict, factory, desc: str,
                                regime_aware: bool = False) -> dict:
        n_folds = data['n_folds']
        fold_nrmses, fold_preds_all = [], []
        for test_fi in range(n_folds):
            train_fis = [f for f in range(n_folds) if f != test_fi]
            strategy = factory()
            train_preds, train_targs = self._concat_folds(data, train_fis)
            fit_kw = self._regime_fit_kwargs(data, train_fis) if regime_aware else {}
            strategy.fit(train_preds, train_targs, **fit_kw)
            predict_kw = {'regime': self.fold_regimes.get(test_fi)} if regime_aware else {}
            ensemble_pred = strategy.predict(
                data['folds'][test_fi]['predictions'], **predict_kw)
            nrmse = compute_nrmse(ensemble_pred, data['folds'][test_fi]['targets'])
            fold_nrmses.append(nrmse)
            fold_preds_all.append(ensemble_pred)
        return {
            'per_fold_nrmse': fold_nrmses,
            'mean_nrmse': float(np.mean(fold_nrmses)),
            'std_nrmse': float(np.std(fold_nrmses)),
            'fold_predictions': fold_preds_all,
        }

    def _concat_folds(self, data: dict, fold_indices: list):
        preds = np.concatenate(
            [data['folds'][fi]['predictions'] for fi in fold_indices], axis=1)
        targs = np.concatenate(
            [data['folds'][fi]['targets'] for fi in fold_indices], axis=0)
        return preds, targs

    def _regime_fit_kwargs(self, data: dict, fold_indices: list) -> dict:
        return {
            'fold_predictions': [data['folds'][fi]['predictions']
                                 for fi in fold_indices],
            'fold_targets': [data['folds'][fi]['targets']
                             for fi in fold_indices],
            'fold_regimes': {i: self.fold_regimes.get(fi, 'unknown')
                             for i, fi in enumerate(fold_indices)},
        }


# ── Fold / ensemble helpers ───────────────────────────────────────────

def fold_input_series(fold: dict, n_nodes: int) -> np.ndarray:
    """Extract representative 1D input series from a fold's targets.

    For single-node (nowcast): returns first-step targets directly.
    For multi-node (forecast): averages the first-step values across nodes
    per window to get a representative epidemic curve.

    Parameters
    ----------
    fold    : dict with 'targets' (n_windows*n_nodes, horizon) and 'n_windows'
    n_nodes : number of nodes (42 for forecast, 1 for nowcast)

    Returns
    -------
    (n_windows,) array — representative epidemic curve
    """
    targets = fold['targets']
    n_windows = fold['n_windows']
    if n_nodes <= 1:
        return targets[:, 0]
    first_step = targets[:, 0].reshape(n_windows, n_nodes)
    return first_step.mean(axis=1)


def make_ensemble(strategy_name_or_result):
    """Instantiate an EnsembleModel from a strategy name or suggest() result."""
    if isinstance(strategy_name_or_result, dict):
        name = strategy_name_or_result.get('best_strategy', 'Median (all)')
    else:
        name = str(strategy_name_or_result)

    factory_map = {desc: factory for factory, desc in _build_configs()}
    factory = factory_map.get(name)
    if factory is None:
        warnings.warn(f"Unknown strategy '{name}'. Falling back to MedianEnsemble.")
        return EnsembleModel(MedianEnsemble())
    return EnsembleModel(factory())


# ── Per-sample training data construction ──────────────────────────────

def _build_sample_training_data(train_data: dict, l2e_results: dict,
                                min_context: int = 8,
                                fold_regimes: dict = None) -> tuple:
    """Build per-sample training data from L2E fold predictions.

    For each (fold, window, node) triplet with w >= min_context:
      - Features: extracted from node n's observable past up to window w
      - Targets: per-strategy per-sample RMSE

    Returns
    -------
    features       : (n_valid, d) input series features
    strategy_rmses : {strategy_name: (n_valid,) per-sample RMSE}
    sample_regimes : list of regime strings per sample
    """
    n_folds = train_data['n_folds']
    n_nodes = train_data['n_nodes']

    strategy_names = sorted([
        name for name in l2e_results['strategies']
        if name not in META_STRATEGIES
        and 'fold_predictions' in l2e_results['strategies'][name]
    ])

    all_features = []
    all_rmses = {name: [] for name in strategy_names}
    all_regimes = []

    for fi in range(n_folds):
        fold = train_data['folds'][fi]
        targets = fold['targets']
        n_samples = targets.shape[0]
        n_windows = n_samples // max(n_nodes, 1)
        regime = (fold_regimes or {}).get(fi, 'unknown')

        strategy_preds = {}
        for name in strategy_names:
            fp = l2e_results['strategies'][name].get('fold_predictions')
            if fp is not None and fi < len(fp):
                strategy_preds[name] = fp[fi]

        if len(strategy_preds) < len(strategy_names):
            continue

        for n in range(n_nodes):
            node_series = targets[n::n_nodes, 0]

            for w in range(min_context, n_windows):
                input_past = node_series[:w]
                if len(input_past) < 4:
                    continue

                features = _extract_features(input_past)
                all_features.append(features)
                all_regimes.append(regime)

                sample_idx = w * n_nodes + n
                target_sample = targets[sample_idx]
                for name in strategy_names:
                    pred_sample = strategy_preds[name][sample_idx]
                    rmse = float(np.sqrt(np.mean(
                        (pred_sample - target_sample) ** 2
                    )))
                    all_rmses[name].append(rmse)

    if not all_features:
        return (np.empty((0, len(FEATURE_NAMES))),
                {n: np.array([]) for n in strategy_names},
                [])

    return (np.array(all_features),
            {n: np.array(v) for n, v in all_rmses.items()},
            all_regimes)


# ── Strategy Recommender ───────────────────────────────────────────────

class StrategyRecommender:
    """Maps (input_series_features, regime, strategy) → predicted ranking score.

    Uses Ridge regression with interaction terms::

        X = [features(d) | regime_onehot(R) | strategy_onehot(S)
             | features ⊗ strategy(d×S) | regime ⊗ strategy(R×S)]
        y = z-scored per-sample RMSE

    The regime×strategy interaction lets the model learn which strategies
    excel in which epidemic regimes (e.g., RegimeStack in known regimes).
    """

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.strategy_names_ = []
        self.regime_names_ = []
        self.ridge_ = None
        self.feat_mean_ = None
        self.feat_std_ = None
        self.n_features_ = 0
        self.n_train_samples_ = 0

    def fit(self, sample_features: np.ndarray, strategy_rmses: dict,
            sample_regimes: list = None):
        """Fit Ridge on per-sample interaction features with regime context."""
        from sklearn.linear_model import Ridge

        self.strategy_names_ = sorted([
            name for name in strategy_rmses if name not in META_STRATEGIES
        ])

        n_s = len(self.strategy_names_)
        N = len(sample_features)
        d = sample_features.shape[1]
        self.n_features_ = d
        self.n_train_samples_ = N

        self.feat_mean_ = sample_features.mean(axis=0)
        self.feat_std_ = np.maximum(sample_features.std(axis=0), 1e-8)
        X_feat = (sample_features - self.feat_mean_) / self.feat_std_

        # Build regime one-hot
        if sample_regimes and len(sample_regimes) == N:
            self.regime_names_ = sorted(set(sample_regimes))
        else:
            self.regime_names_ = []
        n_r = len(self.regime_names_)
        regime_to_idx = {r: i for i, r in enumerate(self.regime_names_)}

        R = np.zeros((N, n_r))
        if sample_regimes and n_r > 0:
            for i, reg in enumerate(sample_regimes):
                if reg in regime_to_idx:
                    R[i, regime_to_idx[reg]] = 1.0

        rmse_matrix = np.column_stack([
            strategy_rmses[name] for name in self.strategy_names_
        ])

        # Z-score per sample: normalizes difficulty, targets relative ranking
        sample_means = rmse_matrix.mean(axis=1, keepdims=True)
        sample_stds = np.maximum(rmse_matrix.std(axis=1, keepdims=True), 1e-8)
        z_matrix = (rmse_matrix - sample_means) / sample_stds

        # X layout: [features(d) | regime(R) | strategy(S)
        #            | feat×strat(d×S) | regime×strat(R×S)]
        n_total = N * n_s
        total_dim = d + n_r + n_s + d * n_s + n_r * n_s
        X = np.zeros((n_total, total_dim))
        y = np.zeros(n_total)

        off_r = d
        off_s = d + n_r
        off_fs = d + n_r + n_s
        off_rs = off_fs + d * n_s

        for si in range(n_s):
            start = si * N
            end = start + N
            X[start:end, :d] = X_feat                                     # features
            X[start:end, off_r:off_r + n_r] = R                           # regime
            X[start:end, off_s + si] = 1.0                                # strategy
            X[start:end, off_fs + si*d : off_fs + (si+1)*d] = X_feat      # feat×strat
            X[start:end, off_rs + si*n_r : off_rs + (si+1)*n_r] = R       # regime×strat
            y[start:end] = z_matrix[:, si]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=True)
            self.ridge_.fit(X, y)

        return self

    def predict_score(self, features: np.ndarray, strategy_name: str,
                      regime: str = None) -> float:
        """Predict z-scored RMSE for a (features, regime, strategy) pair."""
        si = self.strategy_names_.index(strategy_name)
        x = self._build_x(features, si, regime)
        return float(self.ridge_.predict(x)[0])

    def rank(self, features: np.ndarray, regime: str = None) -> list:
        """Rank all strategies for given input features (best first)."""
        d = self.n_features_
        n_s = len(self.strategy_names_)
        n_r = len(self.regime_names_)
        x_feat = (features - self.feat_mean_) / self.feat_std_

        # Regime one-hot
        r_vec = np.zeros(n_r)
        if regime and n_r > 0:
            regime_to_idx = {r: i for i, r in enumerate(self.regime_names_)}
            if regime in regime_to_idx:
                r_vec[regime_to_idx[regime]] = 1.0

        off_r = d
        off_s = d + n_r
        off_fs = d + n_r + n_s
        off_rs = off_fs + d * n_s
        total_dim = off_rs + n_r * n_s

        X = np.zeros((n_s, total_dim))
        X[:, :d] = x_feat
        X[:, off_r:off_r + n_r] = r_vec
        X[np.arange(n_s), off_s + np.arange(n_s)] = 1.0
        for si in range(n_s):
            X[si, off_fs + si*d : off_fs + (si+1)*d] = x_feat
            X[si, off_rs + si*n_r : off_rs + (si+1)*n_r] = r_vec

        scores = self.ridge_.predict(X)
        return sorted(
            zip(self.strategy_names_, scores.tolist()),
            key=lambda kv: kv[1],
        )

    def _build_x(self, features: np.ndarray, strategy_idx: int,
                 regime: str = None) -> np.ndarray:
        d = self.n_features_
        n_s = len(self.strategy_names_)
        n_r = len(self.regime_names_)
        x_feat = (features - self.feat_mean_) / self.feat_std_

        r_vec = np.zeros(n_r)
        if regime and n_r > 0:
            regime_to_idx = {r: i for i, r in enumerate(self.regime_names_)}
            if regime in regime_to_idx:
                r_vec[regime_to_idx[regime]] = 1.0

        off_r = d
        off_s = d + n_r
        off_fs = d + n_r + n_s
        off_rs = off_fs + d * n_s
        total_dim = off_rs + n_r * n_s

        x = np.zeros(total_dim)
        x[:d] = x_feat
        x[off_r:off_r + n_r] = r_vec
        x[off_s + strategy_idx] = 1.0
        x[off_fs + strategy_idx*d : off_fs + (strategy_idx+1)*d] = x_feat
        x[off_rs + strategy_idx*n_r : off_rs + (strategy_idx+1)*n_r] = r_vec
        return x.reshape(1, -1)


# ── HP Recommender ─────────────────────────────────────────────────────

class HPRecommender:
    """Recommends hyperparameters per model based on time series features.

    Per-model Ridge surrogate: (input_features, hp_values) → per_sample_rmse.
    """

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.surrogates_ = {}
        self.candidates_ = {}
        self.feat_mean_ = None
        self.feat_std_ = None

    def fit(self, train_data: dict, optuna_data: dict,
            min_context: int = 8):
        """Train per-model surrogates using per-(window, node) samples."""
        from sklearn.linear_model import Ridge

        n_folds = train_data['n_folds']
        n_nodes = train_data['n_nodes']
        models = train_data['models']

        fold_samples = {}
        all_feat_vectors = []

        for fi in range(n_folds):
            fold = train_data['folds'][fi]
            targets = fold['targets']
            n_windows = fold['n_windows']

            samples = []
            for n in range(n_nodes):
                node_series = targets[n::n_nodes, 0]
                for w in range(min_context, n_windows):
                    input_past = node_series[:w]
                    if len(input_past) < 4:
                        continue
                    sample_idx = w * n_nodes + n
                    feats = _extract_features(input_past)
                    samples.append((sample_idx, feats))
                    all_feat_vectors.append(feats)
            fold_samples[fi] = samples

        if not all_feat_vectors:
            return

        all_feat_arr = np.array(all_feat_vectors)
        self.feat_mean_ = all_feat_arr.mean(axis=0)
        self.feat_std_ = np.maximum(all_feat_arr.std(axis=0), 1e-8)
        n_ff = len(FEATURE_NAMES)

        for model_name, df in optuna_data.items():
            if model_name not in models:
                continue
            mi = models.index(model_name)

            trial_df = df.groupby(['fold', 'trial']).first().reset_index()
            hp_cols = sorted([c for c in trial_df.columns if c.startswith('hp_')])

            if not hp_cols or len(trial_df) < 3:
                continue

            numeric_hp = []
            for c in hp_cols:
                vals = pd.to_numeric(trial_df[c], errors='coerce')
                if vals.notna().sum() > len(trial_df) * 0.5:
                    numeric_hp.append(c)
            if not numeric_hp:
                continue

            best_loss_per_fold = {}
            for fi_1b in trial_df['fold'].unique():
                fold_trials = trial_df[trial_df['fold'] == fi_1b]
                best_loss_per_fold[int(fi_1b) - 1] = float(
                    fold_trials['val_loss_aggregate'].min()
                )

            fold_best_rmse = {}
            for fi in range(n_folds):
                fold = train_data['folds'][fi]
                model_preds = fold['predictions'][mi]
                targets = fold['targets']
                rmse_map = {}
                for sample_idx, _ in fold_samples.get(fi, []):
                    pred = model_preds[sample_idx]
                    targ = targets[sample_idx]
                    rmse_map[sample_idx] = float(np.sqrt(
                        np.mean((pred - targ) ** 2)
                    ))
                fold_best_rmse[fi] = rmse_map

            X_rows, y_rows, candidates = [], [], []
            for _, row in trial_df.iterrows():
                fi = int(row['fold']) - 1
                if fi < 0 or fi >= n_folds:
                    continue
                if fi not in fold_samples or not fold_samples[fi]:
                    continue

                hp_vals = np.array([
                    float(row[c]) if pd.notna(row[c]) else 0.0
                    for c in numeric_hp
                ])
                val_loss = float(row['val_loss_aggregate'])
                if not np.isfinite(val_loss):
                    continue

                best_loss = best_loss_per_fold.get(fi, val_loss)
                ratio = val_loss / max(best_loss, 1e-8)

                for sample_idx, feats in fold_samples[fi]:
                    ff = (feats - self.feat_mean_) / self.feat_std_
                    best_rmse = fold_best_rmse.get(fi, {}).get(sample_idx, 0.0)
                    sample_rmse = best_rmse * ratio
                    X_rows.append(np.concatenate([ff, hp_vals]))
                    y_rows.append(sample_rmse)

                candidates.append({
                    c.replace('hp_', ''): row[c]
                    for c in numeric_hp if pd.notna(row[c])
                })

            if len(X_rows) < 3:
                continue

            X = np.array(X_rows)
            y = np.array(y_rows)

            hp_block = X[:, n_ff:]
            hp_mean = hp_block.mean(axis=0)
            hp_std = np.maximum(hp_block.std(axis=0), 1e-8)
            X[:, n_ff:] = (hp_block - hp_mean) / hp_std

            valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if valid.sum() < 3:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ridge = Ridge(alpha=self.alpha, fit_intercept=True)
                ridge.fit(X[valid], y[valid])

            seen = set()
            unique = []
            for c in candidates:
                key = tuple(sorted((k, round(float(v), 8)) for k, v in c.items()
                                   if isinstance(v, (int, float))))
                if key not in seen:
                    seen.add(key)
                    unique.append(c)

            self.surrogates_[model_name] = {
                'ridge': ridge,
                'hp_cols': numeric_hp,
                'hp_mean': hp_mean,
                'hp_std': hp_std,
                'n_features': n_ff,
            }
            self.candidates_[model_name] = unique

    def suggest(self, features: np.ndarray, model_name: str) -> dict:
        """Suggest best HP config for a model.

        Returns dict of HP values (hp_ prefix stripped), or {} if unavailable.
        """
        if model_name not in self.surrogates_:
            return {}

        info = self.surrogates_[model_name]
        ff = (features - self.feat_mean_) / self.feat_std_

        best_loss = float('inf')
        best_hp = {}

        for hp_dict in self.candidates_[model_name]:
            hp_vals = np.array([
                float(hp_dict.get(c.replace('hp_', ''), 0.0))
                for c in info['hp_cols']
            ])
            hp_std = (hp_vals - info['hp_mean']) / info['hp_std']
            x = np.concatenate([ff, hp_std]).reshape(1, -1)
            pred = float(info['ridge'].predict(x)[0])
            if pred < best_loss:
                best_loss = pred
                best_hp = hp_dict

        return best_hp

    def rank(self, features: np.ndarray, model_name: str) -> list:
        """Rank HP configs for a model (best first)."""
        if model_name not in self.surrogates_:
            return []

        info = self.surrogates_[model_name]
        ff = (features - self.feat_mean_) / self.feat_std_

        results = []
        for hp_dict in self.candidates_[model_name]:
            hp_vals = np.array([
                float(hp_dict.get(c.replace('hp_', ''), 0.0))
                for c in info['hp_cols']
            ])
            hp_std = (hp_vals - info['hp_mean']) / info['hp_std']
            x = np.concatenate([ff, hp_std]).reshape(1, -1)
            pred = float(info['ridge'].predict(x)[0])
            results.append((hp_dict, pred))

        return sorted(results, key=lambda kv: kv[1])


# ── Combined L2E Recommender ──────────────────────────────────────────

class L2ERecommender:
    """Combined L2E recommender with proper train/test split.

    Training (offline):
        rec = L2ERecommender()
        rec.train(data, optuna_dir, fold_regimes, train_folds=[0,1,2,3])

    Deployment (training-free):
        result = rec.suggest(observed_series)
        print_rankings(result)
    """

    def __init__(self, strategy_alpha: float = 10.0, hp_alpha: float = 10.0):
        self.strategy_rec = StrategyRecommender(alpha=strategy_alpha)
        self.hp_rec = HPRecommender(alpha=hp_alpha)
        self.model_names_ = []
        self.train_folds_ = []
        self.regime_thresholds_ = {}
        self.l2e_results_ = None
        self.regime_strategy_map_ = {}
        self.global_best_strategy_ = None
        self.strategies_ = {}  # pre-fitted strategies for predict()

    def train(self, data: dict, optuna_dir: str,
              fold_regimes: dict = None, train_folds: list = None):
        """Train both recommenders on specified training folds.

        Runs L2E internally on train folds (LOFO within train set only),
        then trains recommenders on per-sample results. No test fold data
        is used.
        """
        n_folds = data['n_folds']
        n_nodes = data['n_nodes']
        if train_folds is None:
            train_folds = list(range(n_folds))
        self.train_folds_ = list(train_folds)
        self.model_names_ = list(data['models'])

        # 1. Run L2E on train folds only (LOFO within train set)
        train_data = _subset_data(data, train_folds)
        train_regimes = {}
        if fold_regimes:
            train_regimes = {
                i: fold_regimes.get(fi, 'unknown')
                for i, fi in enumerate(train_folds)
            }

        l2e = L2E(fold_regimes=train_regimes)
        self.l2e_results_ = l2e.run(train_data, verbose=False)

        # 1b. Build fold-level regime → best strategy map from LOFO results
        ROBUST_STRATEGIES = {
            'Median (all)', 'Mean (all)', 'TrimmedMean10 (all)',
            'InvNRMSE (all)', 'ConstrainedStack',
            'Stacking a=0.1', 'Stacking a=1', 'Stacking a=10',
            'RegimeStack',
        }
        if fold_regimes:
            strategy_names = [
                name for name in self.l2e_results_['strategies']
                if name not in META_STRATEGIES and name in ROBUST_STRATEGIES
            ]
            regime_nrmses = {}
            for local_i, global_fi in enumerate(train_folds):
                regime = fold_regimes.get(global_fi, 'unknown')
                if regime not in regime_nrmses:
                    regime_nrmses[regime] = {s: [] for s in strategy_names}
                for name in strategy_names:
                    pfn = self.l2e_results_['strategies'][name].get(
                        'per_fold_nrmse', [])
                    if local_i < len(pfn):
                        regime_nrmses[regime][name].append(pfn[local_i])
            # Regime-specific map: only when >=3% advantage over Median
            self.regime_strategy_map_ = {}
            MIN_ADVANTAGE = 0.03
            for regime, strat_nrmses in regime_nrmses.items():
                valid = {s: v for s, v in strat_nrmses.items() if v}
                if not valid:
                    continue
                best = min(valid, key=lambda s: float(np.mean(valid[s])))
                median_nrmse = float(np.mean(
                    valid.get('Median (all)', [float('inf')])))
                best_nrmse = float(np.mean(valid[best]))
                if ((median_nrmse - best_nrmse) / max(median_nrmse, 1e-8)
                        >= MIN_ADVANTAGE):
                    self.regime_strategy_map_[regime] = best

        # Global best strategy (safe, uniformly-behaved strategies only)
        SAFE_GLOBAL = {'Median (all)', 'Mean (all)', 'TrimmedMean10 (all)',
                       'InvNRMSE (all)', 'ConstrainedStack',
                       'Stacking a=0.1', 'Stacking a=1', 'Stacking a=10',
                       'RegimeStack'}
        all_strat = {
            name: res for name, res in self.l2e_results_['strategies'].items()
            if name not in META_STRATEGIES
            and name in SAFE_GLOBAL
            and res.get('per_fold_nrmse')
        }
        if all_strat:
            median_global = float(np.mean(
                all_strat.get('Median (all)', {}).get(
                    'per_fold_nrmse', [float('inf')])
            ))
            global_best = min(
                all_strat,
                key=lambda s: float(np.mean(all_strat[s]['per_fold_nrmse'])),
            )
            global_best_nrmse = float(np.mean(
                all_strat[global_best]['per_fold_nrmse']))
            global_advantage = ((median_global - global_best_nrmse)
                                / max(median_global, 1e-8))
            MIN_ADVANTAGE = 0.03
            self.global_best_strategy_ = (
                global_best if global_advantage >= MIN_ADVANTAGE else None
            )

        # 2. Compute regime thresholds from training data
        fold_feature_dicts = [
            compute_regime_features(
                fold_input_series(train_data['folds'][i], n_nodes)
            )
            for i in range(len(train_folds))
        ]
        levels = [d.get('level', 0) for d in fold_feature_dicts]
        vols = [d.get('volatility', 0) for d in fold_feature_dicts]
        self.regime_thresholds_ = {
            'level_q15': float(np.percentile(levels, 15)),
            'level_q85': float(np.percentile(levels, 85)),
            'vol_q75': float(np.percentile(vols, 75)),
        }

        # 3. Build per-sample training data from L2E fold predictions
        sample_features, strategy_rmses, sample_regimes = \
            _build_sample_training_data(
                train_data, self.l2e_results_, min_context=8,
                fold_regimes=train_regimes,
            )
        n_valid = len(sample_features)
        print(f"[Recommender] Per-sample training: {n_valid} samples, "
              f"{len(strategy_rmses)} strategies, "
              f"{len(set(sample_regimes))} regimes")

        # 4. Train strategy recommender on per-sample data (regime-aware)
        self.strategy_rec.fit(sample_features, strategy_rmses,
                              sample_regimes=sample_regimes)

        # 5. Train HP recommender
        csvs = glob.glob(os.path.join(optuna_dir, '*_optuna.csv'))
        optuna_data = {}
        fold_remap = {fi + 1: i + 1 for i, fi in enumerate(train_folds)}
        for path in csvs:
            name = os.path.basename(path).replace('_optuna.csv', '')
            try:
                df = pd.read_csv(path)
                train_fold_1based = [fi + 1 for fi in train_folds]
                df = df[df['fold'].isin(train_fold_1based)].copy()
                df['fold'] = df['fold'].map(fold_remap)
                if len(df) > 0:
                    optuna_data[name] = df
            except Exception:
                continue
        self.hp_rec.fit(train_data, optuna_data, min_context=8)

        # 6. Fit all strategies on full training data (for predict() API)
        all_train_preds, all_train_targs = np.concatenate(
            [train_data['folds'][fi]['predictions'] for fi in range(len(train_folds))],
            axis=1,
        ), np.concatenate(
            [train_data['folds'][fi]['targets'] for fi in range(len(train_folds))],
            axis=0,
        )
        self.strategies_ = {}
        for factory, name in _build_configs():
            if name in META_STRATEGIES:
                continue
            strat = factory()
            kw = {}
            if 'Regime' in name:
                kw['fold_predictions'] = [
                    train_data['folds'][fi]['predictions']
                    for fi in range(len(train_folds))]
                kw['fold_targets'] = [
                    train_data['folds'][fi]['targets']
                    for fi in range(len(train_folds))]
                kw['fold_regimes'] = train_regimes
            strat.fit(all_train_preds, all_train_targs, **kw)
            self.strategies_[name] = strat
        print(f"[Recommender] Fitted {len(self.strategies_)} strategies "
              f"for predict() API")

        return self.l2e_results_

    def predict(self, observable_past: np.ndarray,
                model_predictions: np.ndarray) -> np.ndarray:
        """Per-sample ensemble prediction.

        Parameters
        ----------
        observable_past   : (T,) observed epidemic curve up to now
        model_predictions : (n_models, horizon) base model predictions
                            for a single sample

        Returns
        -------
        (horizon,) ensemble prediction using the top-ranked strategy
        """
        features = _extract_features(observable_past)
        feats_dict = compute_regime_features(observable_past)
        regime = _detect_regime(
            feats_dict,
            self.regime_thresholds_.get('level_q15', 0),
            self.regime_thresholds_.get('level_q85', 1),
            self.regime_thresholds_.get('vol_q75', 0.5),
        )

        ranking = self.strategy_rec.rank(features, regime=regime)

        # Pick best strategy that has been fitted
        for name, _score in ranking:
            if name in self.strategies_:
                preds_3d = model_predictions[:, np.newaxis, :]  # (n_models, 1, H)
                kw = {'regime': regime} if 'Regime' in name else {}
                return self.strategies_[name].predict(preds_3d, **kw)[0]

        # Fallback: median
        return np.median(model_predictions, axis=0)

    def suggest(self, series: np.ndarray) -> dict:
        """Get ensemble recommendations for a new observed time series.

        Parameters
        ----------
        series : (n_timesteps,) observed past values of the epidemic curve

        Returns
        -------
        dict:
            strategy_ranking : [(name, score), ...] best first
            best_strategy    : str
            regime           : str (auto-detected)
            model_hps        : {model_name: {hp_name: value}}
        """
        features = _extract_features(series)

        feats_dict = compute_regime_features(series)
        regime = _detect_regime(
            feats_dict,
            self.regime_thresholds_.get('level_q15', 0),
            self.regime_thresholds_.get('level_q85', 1),
            self.regime_thresholds_.get('vol_q75', 0.5),
        )

        ranking = self.strategy_rec.rank(features, regime=regime)

        model_hps = {}
        for m in self.model_names_:
            hp = self.hp_rec.suggest(features, m)
            if hp:
                model_hps[m] = hp

        best_strategy = ranking[0][0] if ranking else 'Median (all)'
        return {
            'strategy_ranking': ranking,
            'best_strategy':    best_strategy,
            'strategy':         make_ensemble(best_strategy),
            'regime':           regime,
            'model_hps':        model_hps,
        }

    def save(self, path: str):
        """Save trained recommender to disk."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'L2ERecommender':
        """Load pre-trained recommender from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── Print rankings ─────────────────────────────────────────────────────

def print_rankings(result: dict, top_n: int = None):
    """Pretty-print strategy rankings and HP recommendations."""
    print(f"\n{'='*60}")
    print(f"L2E Recommendations")
    print(f"{'='*60}")

    print(f"\nDetected regime: {result['regime']}")
    print(f"Best strategy:   {result['best_strategy']}")

    rankings = result['strategy_ranking']
    n_show = top_n or len(rankings)

    print(f"\n{'Rank':>4s}  {'Strategy':25s}  {'Score':>10s}")
    print(f"{'':->4s}  {'':->25s}  {'':->10s}")
    for i, (name, score) in enumerate(rankings[:n_show], 1):
        marker = '  <-- best' if i == 1 else ''
        print(f"{i:>4d}  {name:25s}  {score:>10.4f}{marker}")

    if top_n and top_n < len(rankings):
        print(f"  ... ({len(rankings) - top_n} more strategies)")

    model_hps = result.get('model_hps', {})
    if model_hps:
        print(f"\nHP Recommendations ({len(model_hps)} models):")
        print(f"{'':->60s}")
        for model in sorted(model_hps):
            hps = model_hps[model]
            hp_str = ', '.join(f'{k}={v}' for k, v in sorted(hps.items()))
            print(f"  {model:22s}  {hp_str}")
    print()
