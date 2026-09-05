"""Ensemble strategy implementations (numpy-based, no PyTorch dependency)."""

from abc import ABC, abstractmethod
from typing import Optional
import warnings

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
from scipy.stats import trim_mean


class EnsembleStrategyBase(ABC):
    """Base class for ensemble aggregation strategies."""

    name: str = 'base'

    @abstractmethod
    def fit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        model_names: Optional[list] = None,
        **kwargs,
    ) -> 'EnsembleStrategyBase':
        """Learn from training data.

        Parameters
        ----------
        predictions : (n_models, n_samples, horizon)
        targets     : (n_samples, horizon)
        """

    @abstractmethod
    def predict(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """Combine model predictions.

        Parameters
        ----------
        predictions : (n_models, n_samples, horizon)

        Returns
        -------
        (n_samples, horizon)
        """

    def get_weights(self) -> Optional[np.ndarray]:
        """Return model weights for interpretability (if applicable)."""
        return None


# ── Simple aggregation ────────────────────────────────────────────────


class MedianEnsemble(EnsembleStrategyBase):
    name = 'Median'

    def fit(self, predictions, targets, **kwargs):
        return self

    def predict(self, predictions, **kwargs):
        return np.median(predictions, axis=0)


class MeanEnsemble(EnsembleStrategyBase):
    name = 'Mean'

    def fit(self, predictions, targets, **kwargs):
        return self

    def predict(self, predictions, **kwargs):
        return np.mean(predictions, axis=0)


class TrimmedMeanEnsemble(EnsembleStrategyBase):
    name = 'TrimmedMean'

    def __init__(self, trim_frac: float = 0.1):
        self.trim_frac = trim_frac

    def fit(self, predictions, targets, **kwargs):
        return self

    def predict(self, predictions, **kwargs):
        return trim_mean(predictions, proportiontocut=self.trim_frac, axis=0)


# ── Score-weighted ────────────────────────────────────────────────────


class InverseNRMSEEnsemble(EnsembleStrategyBase):
    """Weight each model by 1/NRMSE computed on the training set."""

    name = 'InvNRMSE'

    def __init__(self):
        self.weights_ = None

    def fit(self, predictions, targets, **kwargs):
        from .ensemble import compute_nrmse
        n_models = predictions.shape[0]
        scores = np.array([
            compute_nrmse(predictions[m], targets) for m in range(n_models)
        ])
        w = 1.0 / np.maximum(scores, 1e-8)
        self.weights_ = w / w.sum()
        return self

    def predict(self, predictions, **kwargs):
        return np.tensordot(self.weights_, predictions, axes=([0], [0]))

    def get_weights(self):
        return self.weights_


class TopKMedianEnsemble(EnsembleStrategyBase):
    """Median of the top-K models ranked by training NRMSE."""

    def __init__(self, k: int = 10):
        self.k = k
        self.top_indices_ = None

    @property
    def name(self):
        return f'TopKMedian_k{self.k}'

    def fit(self, predictions, targets, **kwargs):
        from .ensemble import compute_nrmse
        n_models = predictions.shape[0]
        scores = [compute_nrmse(predictions[m], targets) for m in range(n_models)]
        k = min(self.k, n_models)
        self.top_indices_ = np.argsort(scores)[:k]
        return self

    def predict(self, predictions, **kwargs):
        return np.median(predictions[self.top_indices_], axis=0)


class TopKMeanEnsemble(EnsembleStrategyBase):
    """Mean of the top-K models ranked by training NRMSE."""

    def __init__(self, k: int = 10):
        self.k = k
        self.top_indices_ = None

    @property
    def name(self):
        return f'TopKMean_k{self.k}'

    def fit(self, predictions, targets, **kwargs):
        from .ensemble import compute_nrmse
        n_models = predictions.shape[0]
        scores = [compute_nrmse(predictions[m], targets) for m in range(n_models)]
        k = min(self.k, n_models)
        self.top_indices_ = np.argsort(scores)[:k]
        return self

    def predict(self, predictions, **kwargs):
        return np.mean(predictions[self.top_indices_], axis=0)


class ExpWeightedEnsemble(EnsembleStrategyBase):
    """Exponential weighting: w_i = exp(-beta * NRMSE_i), normalized.

    More aggressive than InvNRMSE — sharply concentrates weight on the
    best models.
    """

    def __init__(self, beta: float = 5.0):
        self.beta = beta
        self.weights_ = None

    @property
    def name(self):
        return f'ExpWeighted_b{self.beta}'

    def fit(self, predictions, targets, **kwargs):
        from .ensemble import compute_nrmse
        n_models = predictions.shape[0]
        scores = np.array([
            compute_nrmse(predictions[m], targets) for m in range(n_models)
        ])
        log_w = -self.beta * scores
        log_w -= log_w.max()  # numerical stability
        w = np.exp(log_w)
        self.weights_ = w / w.sum()
        return self

    def predict(self, predictions, **kwargs):
        return np.tensordot(self.weights_, predictions, axes=([0], [0]))

    def get_weights(self):
        return self.weights_


# ── Ridge stacking ────────────────────────────────────────────────────


class StackingEnsemble(EnsembleStrategyBase):
    """Per-horizon Ridge regression stacking."""

    name = 'Stacking'

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights_ = None  # (horizon, n_models)

    def fit(self, predictions, targets, **kwargs):
        n_models, n_samples, horizon = predictions.shape
        self.weights_ = np.zeros((horizon, n_models))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            for h in range(horizon):
                X = predictions[:, :, h].T  # (n_samples, n_models)
                y = targets[:, h]
                ridge = Ridge(alpha=self.alpha, fit_intercept=False)
                ridge.fit(X, y)
                self.weights_[h] = ridge.coef_
        return self

    def predict(self, predictions, **kwargs):
        return np.einsum('msh,hm->sh', predictions, self.weights_)

    def get_weights(self):
        return self.weights_


# ── L1 / ElasticNet stacking ──────────────────────────────────────────


class LassoStackingEnsemble(EnsembleStrategyBase):
    """Per-horizon Lasso (L1) stacking; L1 penalty achieves soft model selection."""

    name = 'LassoStack'

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.weights_ = None

    def fit(self, predictions, targets, **kwargs):
        n_models, n_samples, horizon = predictions.shape
        self.weights_ = np.zeros((horizon, n_models))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for h in range(horizon):
                X = predictions[:, :, h].T   # (n_samples, n_models)
                y = targets[:, h]
                lasso = Lasso(alpha=self.alpha, fit_intercept=False,
                              max_iter=10000)
                lasso.fit(X, y)
                self.weights_[h] = lasso.coef_
        return self

    def predict(self, predictions, **kwargs):
        return np.einsum('msh,hm->sh', predictions, self.weights_)

    def get_weights(self):
        return self.weights_


class ElasticNetStackingEnsemble(EnsembleStrategyBase):
    """Per-horizon ElasticNet (L1+L2) stacking; combines Lasso sparsity with Ridge grouping."""

    name = 'ElasticStack'

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weights_ = None

    def fit(self, predictions, targets, **kwargs):
        n_models, n_samples, horizon = predictions.shape
        self.weights_ = np.zeros((horizon, n_models))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for h in range(horizon):
                X = predictions[:, :, h].T
                y = targets[:, h]
                enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                  fit_intercept=False, max_iter=10000)
                enet.fit(X, y)
                self.weights_[h] = enet.coef_
        return self

    def predict(self, predictions, **kwargs):
        return np.einsum('msh,hm->sh', predictions, self.weights_)

    def get_weights(self):
        return self.weights_


# ── Constrained stacking ──────────────────────────────────────────────


class ConstrainedStackingEnsemble(EnsembleStrategyBase):
    """Non-negative sum-to-one stacking via SLSQP — convex combination, no extrapolation."""

    name = 'ConstrainedStack'

    def __init__(self):
        self.weights_ = None

    def fit(self, predictions, targets, **kwargs):
        n_models, n_samples, horizon = predictions.shape
        self.weights_ = np.zeros((horizon, n_models))

        for h in range(horizon):
            X = predictions[:, :, h].T   # (n_samples, n_models)
            y = targets[:, h]

            def obj(w):
                r = X @ w - y
                return float(r @ r)

            def jac(w):
                return 2.0 * (X.T @ (X @ w - y))

            w0 = np.ones(n_models) / n_models
            bounds = [(0.0, None)] * n_models
            constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}

            res = minimize(obj, w0, jac=jac, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-12})
            self.weights_[h] = res.x

        return self

    def predict(self, predictions, **kwargs):
        return np.einsum('msh,hm->sh', predictions, self.weights_)

    def get_weights(self):
        return self.weights_


# ── Gradient boosting meta-learner ────────────────────────────────────


class GBStackingEnsemble(EnsembleStrategyBase):
    """Gradient Boosted Trees meta-learner — the only nonlinear combiner in the suite."""

    name = 'GBStack'

    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.models_: dict = {}

    def fit(self, predictions, targets, **kwargs):
        n_models, n_samples, horizon = predictions.shape
        self.models_ = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for h in range(horizon):
                X = predictions[:, :, h].T
                y = targets[:, h]
                gb = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                )
                gb.fit(X, y)
                self.models_[h] = gb
        return self

    def predict(self, predictions, **kwargs):
        n_models, n_samples, horizon = predictions.shape
        result = np.zeros((n_samples, horizon))
        for h in range(horizon):
            result[:, h] = self.models_[h].predict(predictions[:, :, h].T)
        return result

    def get_weights(self):
        """Feature importances as a proxy for per-model contribution."""
        if not self.models_:
            return None
        return np.array([
            self.models_[h].feature_importances_
            for h in sorted(self.models_)
        ])  # (horizon, n_models)


# ── Regime-conditional stacking ───────────────────────────────────────


class RegimeStackingEnsemble(EnsembleStrategyBase):
    """Learn separate Ridge stacking weights per regime.

    During fit(), fold regimes must be provided via the ``fold_regimes``
    keyword so that training data can be grouped by regime.

    During predict(), the ``regime`` keyword selects which weight set to use.
    Falls back to global weights if the test regime was not seen during training.
    """

    name = 'RegimeStack'

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.regime_weights_ = {}   # {regime: (horizon, n_models)}
        self.global_weights_ = None

    def fit(self, predictions, targets, **kwargs):
        """Fit per-regime stacking weights.

        Expects keyword arguments:
            fold_predictions : list of (n_models, n_samples_i, horizon) per fold
            fold_targets     : list of (n_samples_i, horizon) per fold
            fold_regimes     : dict {fold_idx: regime_label}
        """
        fold_predictions = kwargs.get('fold_predictions', [])
        fold_targets = kwargs.get('fold_targets', [])
        fold_regimes = kwargs.get('fold_regimes', {})

        if not fold_predictions:
            # Fallback: fit global Ridge weights (no fold/regime structure)
            stacker = StackingEnsemble(alpha=self.alpha)
            stacker.fit(predictions, targets)
            self.global_weights_ = stacker.weights_
            return self

        # Group folds by regime
        regime_data = {}
        for fi in range(len(fold_predictions)):
            regime = fold_regimes.get(fi, 'unknown')
            if regime not in regime_data:
                regime_data[regime] = {'preds': [], 'targs': []}
            regime_data[regime]['preds'].append(fold_predictions[fi])
            regime_data[regime]['targs'].append(fold_targets[fi])

        # Fit per-regime stacking
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            for regime, rdata in regime_data.items():
                preds_cat = np.concatenate(rdata['preds'], axis=1)
                targs_cat = np.concatenate(rdata['targs'], axis=0)
                n_models, n_samples, horizon = preds_cat.shape
                weights = np.zeros((horizon, n_models))
                for h in range(horizon):
                    X = preds_cat[:, :, h].T
                    y = targs_cat[:, h]
                    ridge = Ridge(alpha=self.alpha, fit_intercept=False)
                    ridge.fit(X, y)
                    weights[h] = ridge.coef_
                self.regime_weights_[regime] = weights

            # Global fallback: use constrained (NNLS) stacking for robustness
            # Unconstrained Ridge overfits when n_models >> n_regimes
            all_preds = np.concatenate(fold_predictions, axis=1)
            all_targs = np.concatenate(fold_targets, axis=0)
            global_stacker = ConstrainedStackingEnsemble()
            global_stacker.fit(all_preds, all_targs)
            self.global_weights_ = global_stacker.weights_

        return self

    def predict(self, predictions, **kwargs):
        regime = kwargs.get('regime', None)
        weights = self.regime_weights_.get(regime, self.global_weights_)
        if weights is None:
            weights = self.global_weights_
        return np.einsum('msh,hm->sh', predictions, weights)

    def get_weights(self):
        return {'regime_weights': self.regime_weights_, 'global': self.global_weights_}


# ── Deployment wrapper ─────────────────────────────────────────────────


class EnsembleModel:
    """Deployment wrapper for any ensemble strategy.

    Provides a fold-aware API that mirrors the standard epilearn
    ``rolling_train`` / ``evaluate`` pattern.

    Typical usage::

        ensemble = make_ensemble(result)          # from suggest()
        ensemble.rolling_train(train_fold)        # fit on train split
        out = ensemble.evaluate(test_fold)        # NRMSE on test split
        print(f"NRMSE: {out['nrmse']:.4f}")

    The ensemble is also available directly as ``result['strategy']``
    after calling ``L2ERecommender.suggest()``.
    """

    def __init__(self, strategy: 'EnsembleStrategyBase'):
        self.strategy = strategy
        self.name = getattr(strategy, 'name', type(strategy).__name__)

    def rolling_train(self, fold: dict, **fit_kwargs) -> 'EnsembleModel':
        """Fit the ensemble on a fold's model predictions and targets.

        Parameters
        ----------
        fold       : fold dict with 'predictions' (n_models, n_samples, horizon)
                     and 'targets' (n_samples, horizon)
        **fit_kwargs : forwarded to the underlying strategy's fit()
        """
        self.strategy.fit(fold['predictions'], fold['targets'], **fit_kwargs)
        return self

    def predict(self, fold: dict, **predict_kwargs) -> np.ndarray:
        """Return ensemble predictions (n_samples, horizon) for a fold."""
        return self.strategy.predict(fold['predictions'], **predict_kwargs)

    def evaluate(self, fold: dict, **predict_kwargs) -> dict:
        """Compute NRMSE on a fold.

        Returns
        -------
        dict with 'nrmse', 'predictions', 'targets'
        """
        from .ensemble import compute_nrmse
        pred = self.predict(fold, **predict_kwargs)
        nrmse = compute_nrmse(pred, fold['targets'])
        return {
            'nrmse': nrmse,
            'predictions': pred,
            'targets': fold['targets'],
        }

    def get_weights(self):
        """Return model weights (if applicable)."""
        return self.strategy.get_weights()

    def __repr__(self):
        return f"EnsembleModel(strategy={self.name!r})"

