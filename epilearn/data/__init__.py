"""
EpiLearn Data Module.

This module provides data loading, storage, and manipulation for time series
epidemic data with proper indexing support.
"""

# Core data structures
from .core import TimeSeriesData, LoadedData

# Dataset class (main interface)
from .dataset import Dataset, Data, custom_collate

# Loaders
from .loaders import (
    BaseLoader,
    DataLoaderRegistry,
    CSVLoader,
    TensorLoader,
    NumpyLoader,
    load_from_file,
    load_csv,
    load_tensor,
    load_numpy,
)

# Backwards-compatible alias: `Dataset` was called `UniversalDataset` up to
# 0.0.19. Its constructor signature is a strict superset of the old one, so
# existing scripts keep working. Deprecated -- prefer `Dataset`.
UniversalDataset = Dataset

__all__ = [
    # Core
    'TimeSeriesData',
    'LoadedData',

    # Dataset
    'Dataset',
    'UniversalDataset',
    'Data',
    'custom_collate',
    
    # Loaders
    'BaseLoader',
    'DataLoaderRegistry',
    'CSVLoader',
    'TensorLoader',
    'NumpyLoader',
    'load_from_file',
    'load_csv',
    'load_tensor',
    'load_numpy',
]
