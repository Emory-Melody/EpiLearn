"""
Tensor data loaders for EpiLearn.

Provides loaders for PyTorch (.pt) and NumPy (.npy/.npz) files with
proper timestamp and region handling.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings

from .base import BaseLoader, DataLoaderRegistry
from ..core import LoadedData


class TensorLoader(BaseLoader):
    """
    Load time series data from PyTorch tensor files (.pt).
    
    Supports various data formats stored in .pt files, including:
    - Single tensor (interpreted as features)
    - Dictionary with 'features', 'targets', etc. keys
    - Dictionary with custom key mappings
    
    Example usage:
        loader = TensorLoader(
            file_path="data.pt",
            feature_key="features",
            target_key="targets",
            graph_key="adjacency"
        )
        data = loader.load()
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        feature_key: Optional[str] = None,
        target_key: Optional[str] = None,
        graph_key: Optional[str] = None,
        dynamic_graph_key: Optional[str] = None,
        states_key: Optional[str] = None,
        timestamps_key: Optional[str] = None,
        regions_key: Optional[str] = None,
        feature_names_key: Optional[str] = None,
        target_names_key: Optional[str] = None,
        auto_detect_keys: bool = True,
        **kwargs
    ):
        """
        Initialize tensor loader.
        
        Args:
            file_path: Path to the .pt file
            feature_key: Key for features in dict (if dict format)
            target_key: Key for targets in dict
            graph_key: Key for static graph adjacency
            dynamic_graph_key: Key for dynamic graph
            states_key: Key for state variables
            timestamps_key: Key for timestamp list/array
            regions_key: Key for region list
            feature_names_key: Key for feature names list
            target_names_key: Key for target names list
            auto_detect_keys: Auto-detect common key names if not specified
        """
        self.file_path = Path(file_path)
        self.feature_key = feature_key
        self.target_key = target_key
        self.graph_key = graph_key
        self.dynamic_graph_key = dynamic_graph_key
        self.states_key = states_key
        self.timestamps_key = timestamps_key
        self.regions_key = regions_key
        self.feature_names_key = feature_names_key
        self.target_names_key = target_names_key
        self.auto_detect_keys = auto_detect_keys
    
    # Common key aliases for auto-detection
    FEATURE_ALIASES = ['features', 'x', 'X', 'inputs', 'input', 'data']
    TARGET_ALIASES = ['targets', 'target', 'y', 'Y', 'labels', 'label', 'output', 'outputs']
    GRAPH_ALIASES = ['graph', 'adj', 'adjacency', 'A', 'edge_index']
    DYNAMIC_GRAPH_ALIASES = ['dynamic_graph', 'dynamic_adj', 'temporal_graph', 'od']
    STATES_ALIASES = ['states', 'state', 'SIR', 'sir', 'compartments']
    TIMESTAMP_ALIASES = ['timestamps', 'timestamp', 'time', 'times', 'dates', 'date', 'index', 'time_stamp']
    REGION_ALIASES = ['regions', 'region', 'nodes', 'node', 'locations', 'location']
    FEATURE_NAMES_ALIASES = ['feature_names', 'features_names', 'column_names', 'columns']
    TARGET_NAMES_ALIASES = ['target_names', 'label_names']
    
    def load(self) -> LoadedData:
        """Load and process the tensor file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Tensor file not found: {self.file_path}")
        
        data = torch.load(self.file_path, weights_only=False)
        
        if isinstance(data, dict):
            return self._load_from_dict(data)
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            return self._load_from_tensor(data)
        else:
            raise ValueError(f"Unsupported data type in .pt file: {type(data)}")
    
    def _find_key(self, data: dict, aliases: List[str], explicit_key: Optional[str] = None) -> Optional[str]:
        """Find a key in the dict using explicit key or aliases."""
        if explicit_key and explicit_key in data:
            return explicit_key
        
        if self.auto_detect_keys:
            for alias in aliases:
                if alias in data:
                    return alias
        
        return None
    
    def _get_value(self, data: dict, key: Optional[str]) -> Any:
        """Get value from dict, converting to numpy if needed."""
        if key is None or key not in data:
            return None
        
        value = data[key]
        if isinstance(value, torch.Tensor):
            return value.numpy()
        return value
    
    def _load_from_dict(self, data: dict) -> LoadedData:
        """Load data from dictionary format."""
        # Find keys
        feature_key = self._find_key(data, self.FEATURE_ALIASES, self.feature_key)
        target_key = self._find_key(data, self.TARGET_ALIASES, self.target_key)
        graph_key = self._find_key(data, self.GRAPH_ALIASES, self.graph_key)
        dynamic_graph_key = self._find_key(data, self.DYNAMIC_GRAPH_ALIASES, self.dynamic_graph_key)
        states_key = self._find_key(data, self.STATES_ALIASES, self.states_key)
        timestamps_key = self._find_key(data, self.TIMESTAMP_ALIASES, self.timestamps_key)
        regions_key = self._find_key(data, self.REGION_ALIASES, self.regions_key)
        feature_names_key = self._find_key(data, self.FEATURE_NAMES_ALIASES, self.feature_names_key)
        target_names_key = self._find_key(data, self.TARGET_NAMES_ALIASES, self.target_names_key)
        
        # Extract data
        features = self._get_value(data, feature_key)
        if features is None:
            raise ValueError(f"Could not find features in tensor file. Available keys: {list(data.keys())}")
        
        if isinstance(features, np.ndarray):
            features = features.astype(np.float32)
        
        targets = self._get_value(data, target_key)
        if targets is not None and isinstance(targets, np.ndarray):
            targets = targets.astype(np.float32)
        
        graph = self._get_value(data, graph_key)
        if graph is not None and isinstance(graph, np.ndarray):
            graph = graph.astype(np.float32)
        
        dynamic_graph = self._get_value(data, dynamic_graph_key)
        if dynamic_graph is not None and isinstance(dynamic_graph, np.ndarray):
            dynamic_graph = dynamic_graph.astype(np.float32)
        
        states = self._get_value(data, states_key)
        if states is not None and isinstance(states, np.ndarray):
            states = states.astype(np.float32)
        
        timestamps = self._get_value(data, timestamps_key)
        if timestamps is not None:
            timestamps = list(timestamps)
        
        regions = self._get_value(data, regions_key)
        if regions is not None:
            regions = list(regions)
        
        feature_names = self._get_value(data, feature_names_key)
        if feature_names is not None:
            feature_names = list(feature_names)
        else:
            feature_names = []
        
        target_names = self._get_value(data, target_names_key)
        if target_names is not None:
            target_names = list(target_names)
        else:
            target_names = []
        
        return LoadedData(
            features=features,
            targets=targets,
            timestamps=timestamps,
            regions=regions,
            graph=graph,
            dynamic_graph=dynamic_graph,
            states=states,
            feature_names=feature_names,
            target_names=target_names,
            metadata={
                'source_file': str(self.file_path),
                'detected_keys': {
                    'features': feature_key,
                    'targets': target_key,
                    'graph': graph_key,
                    'dynamic_graph': dynamic_graph_key,
                    'states': states_key,
                }
            }
        )
    
    def _load_from_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> LoadedData:
        """Load data from single tensor."""
        if isinstance(data, torch.Tensor):
            features = data.numpy()
        else:
            features = data
        
        features = features.astype(np.float32)
        
        return LoadedData(
            features=features,
            metadata={'source_file': str(self.file_path)}
        )


class NumpyLoader(BaseLoader):
    """
    Load time series data from NumPy files (.npy or .npz).
    
    For .npy files, the array is treated as features.
    For .npz files, keys are used to identify features, targets, etc.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        feature_key: str = 'features',
        target_key: str = 'targets',
        graph_key: str = 'graph',
        dynamic_graph_key: str = 'dynamic_graph',
        states_key: str = 'states',
        allow_pickle: bool = True,
        **kwargs
    ):
        """
        Initialize NumPy loader.
        
        Args:
            file_path: Path to .npy or .npz file
            feature_key: Key for features in .npz file
            target_key: Key for targets in .npz file
            graph_key: Key for graph in .npz file
            dynamic_graph_key: Key for dynamic graph in .npz file
            states_key: Key for states in .npz file
            allow_pickle: Whether to allow loading pickled objects
        """
        self.file_path = Path(file_path)
        self.feature_key = feature_key
        self.target_key = target_key
        self.graph_key = graph_key
        self.dynamic_graph_key = dynamic_graph_key
        self.states_key = states_key
        self.allow_pickle = allow_pickle
    
    def load(self) -> LoadedData:
        """Load and process the NumPy file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"NumPy file not found: {self.file_path}")
        
        if self.file_path.suffix.lower() == '.npz':
            return self._load_npz()
        else:
            return self._load_npy()
    
    def _load_npy(self) -> LoadedData:
        """Load from .npy file."""
        data = np.load(self.file_path, allow_pickle=self.allow_pickle)
        
        # Handle dict-like objects that may be stored
        if isinstance(data, np.ndarray) and data.dtype == object:
            # This might be a pickled dict
            item = data.item() if data.ndim == 0 else data
            if isinstance(item, dict):
                return self._load_from_dict(item)
        
        features = data.astype(np.float32)
        return LoadedData(
            features=features,
            metadata={'source_file': str(self.file_path)}
        )
    
    def _load_npz(self) -> LoadedData:
        """Load from .npz file."""
        data = np.load(self.file_path, allow_pickle=self.allow_pickle)
        return self._load_from_dict(dict(data))
    
    def _load_from_dict(self, data: dict) -> LoadedData:
        """Load from dictionary."""
        features = data.get(self.feature_key)
        if features is None:
            # Try first key as features
            first_key = list(data.keys())[0] if data else None
            if first_key:
                features = data[first_key]
                warnings.warn(f"Using '{first_key}' as features ('{self.feature_key}' not found)")
        
        if features is None:
            raise ValueError(f"Could not find features. Available keys: {list(data.keys())}")
        
        features = np.asarray(features, dtype=np.float32)
        
        targets = data.get(self.target_key)
        if targets is not None:
            targets = np.asarray(targets, dtype=np.float32)
        
        graph = data.get(self.graph_key)
        if graph is not None:
            graph = np.asarray(graph, dtype=np.float32)
        
        dynamic_graph = data.get(self.dynamic_graph_key)
        if dynamic_graph is not None:
            dynamic_graph = np.asarray(dynamic_graph, dtype=np.float32)
        
        states = data.get(self.states_key)
        if states is not None:
            states = np.asarray(states, dtype=np.float32)
        
        return LoadedData(
            features=features,
            targets=targets,
            graph=graph,
            dynamic_graph=dynamic_graph,
            states=states,
            metadata={'source_file': str(self.file_path)}
        )


# Register loaders
DataLoaderRegistry.register(['pt', 'pth'], TensorLoader)
DataLoaderRegistry.register(['npy', 'npz'], NumpyLoader)
