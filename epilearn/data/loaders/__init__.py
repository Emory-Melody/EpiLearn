"""
Data loaders for EpiLearn.

This module provides various loaders for loading time series data from
different file formats (CSV, PT, NPY, etc.) into the unified LoadedData format.
"""

from .base import BaseLoader, DataLoaderRegistry
from .csv_loader import CSVLoader
from .tensor_loader import TensorLoader, NumpyLoader

__all__ = [
    'BaseLoader',
    'DataLoaderRegistry',
    'CSVLoader',
    'TensorLoader',
    'NumpyLoader',
]


def load_from_file(file_path, **kwargs):
    """
    Convenience function to load data from a file using the appropriate loader.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments passed to the loader
        
    Returns:
        LoadedData object
        
    Example:
        # Load CSV with column specifications
        data = load_from_file(
            "data.csv",
            timestamp_col="date",
            feature_cols=["cases", "deaths"],
            region_col="state"
        )
        
        # Load tensor file
        data = load_from_file("data.pt")
    """
    loader = DataLoaderRegistry.get_loader(file_path, **kwargs)
    if loader is None:
        supported = DataLoaderRegistry.supported_extensions()
        raise ValueError(f"No loader found for file: {file_path}. Supported: {supported}")
    
    return loader.load()


def load_csv(
    file_path,
    timestamp_col,
    feature_cols,
    target_cols=None,
    region_col=None,
    **kwargs
):
    """
    Load time series data from a CSV file.
    
    Args:
        file_path: Path to CSV file
        timestamp_col: Column name for timestamps
        feature_cols: List of feature column names
        target_cols: List of target column names (optional)
        region_col: Column name for regions (for spatiotemporal data)
        **kwargs: Additional arguments for CSVLoader
        
    Returns:
        TimeSeriesData object
    """
    loader = CSVLoader(
        file_path=file_path,
        timestamp_col=timestamp_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
        region_col=region_col,
        **kwargs
    )
    return loader.load_as_time_series()


def load_tensor(file_path, **kwargs):
    """
    Load time series data from a PyTorch tensor file (.pt).
    
    Args:
        file_path: Path to .pt file
        **kwargs: Additional arguments for TensorLoader
        
    Returns:
        TimeSeriesData object
    """
    loader = TensorLoader(file_path=file_path, **kwargs)
    return loader.load_as_time_series()


def load_numpy(file_path, **kwargs):
    """
    Load time series data from a NumPy file (.npy or .npz).
    
    Args:
        file_path: Path to .npy or .npz file
        **kwargs: Additional arguments for NumpyLoader
        
    Returns:
        TimeSeriesData object
    """
    loader = NumpyLoader(file_path=file_path, **kwargs)
    return loader.load_as_time_series()
