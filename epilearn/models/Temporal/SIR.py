"""Backwards-compatible import path.

The compartmental models used to live in ``epilearn.models.Temporal.SIR``.
They moved to ``epilearn.models.Temporal.Compartmental`` in 0.1.0, which also
holds the network variants. This module keeps the old path importable.

Deprecated -- import from ``epilearn.models.Temporal.Compartmental`` instead.
"""

from .Compartmental import SIR, SIS, SEIR  # noqa: F401

__all__ = ['SIR', 'SIS', 'SEIR']
