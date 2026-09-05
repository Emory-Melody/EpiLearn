"""
Core data structures for EpiLearn.

This module provides the fundamental data containers for time series data
with proper timestamp and region indexing support.
"""

import torch
import numpy as np
from typing import Optional, List, Union, Dict, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class TimeSeriesData:
    """
    Core container for time series data with proper indexing.
    
    This class stores numerical data along with timestamp and region indices,
    supporting both temporal-only and spatiotemporal data formats.
    
    Attributes:
        features: Tensor of shape (T, N, F) for spatiotemporal or (T, F) for temporal
                  where T=timesteps, N=regions/nodes, F=features
        targets: Tensor of shape (T, N, T_out) or (T, T_out) for targets
        timestamps: List of timestamp values (can be datetime, int, str)
        regions: Optional list of region identifiers (for spatiotemporal data)
        graph: Optional static adjacency matrix of shape (N, N)
        dynamic_graph: Optional dynamic graph of shape (T, N, N)
        states: Optional state variables (e.g., SIR states) of shape matching features
        feature_names: List of feature column names
        target_names: List of target column names
        metadata: Additional metadata dictionary
    """
    features: torch.Tensor
    targets: Optional[torch.Tensor] = None
    timestamps: Optional[List[Any]] = None
    regions: Optional[List[Any]] = None
    graph: Optional[torch.Tensor] = None
    dynamic_graph: Optional[torch.Tensor] = None
    states: Optional[torch.Tensor] = None
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set up indices after initialization."""
        self._validate_data()
        self._setup_indices()
    
    def _validate_data(self):
        """Validate data shapes and consistency."""
        if self.features is None:
            raise ValueError("Features cannot be None")
        
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.FloatTensor(self.features)
        
        if self.targets is not None and not isinstance(self.targets, torch.Tensor):
            self.targets = torch.FloatTensor(self.targets)
        
        if self.graph is not None and not isinstance(self.graph, torch.Tensor):
            self.graph = torch.FloatTensor(self.graph)
        
        if self.dynamic_graph is not None and not isinstance(self.dynamic_graph, torch.Tensor):
            self.dynamic_graph = torch.FloatTensor(self.dynamic_graph)
        
        if self.states is not None and not isinstance(self.states, torch.Tensor):
            self.states = torch.FloatTensor(self.states)
    
    def _setup_indices(self):
        """Set up default timestamps and regions if not provided."""
        T = self.features.shape[0]
        
        # Setup default timestamps
        if self.timestamps is None:
            self.timestamps = list(range(T))
        elif len(self.timestamps) != T:
            raise ValueError(f"Timestamps length ({len(self.timestamps)}) must match time dimension ({T})")
        
        # Setup default regions (only for spatiotemporal data)
        if self.is_spatiotemporal:
            N = self.features.shape[1]
            if self.regions is None:
                self.regions = list(range(N))
            elif len(self.regions) != N:
                raise ValueError(f"Regions length ({len(self.regions)}) must match node dimension ({N})")
    
    @property
    def is_spatiotemporal(self) -> bool:
        """Check if data is spatiotemporal (has region dimension)."""
        return len(self.features.shape) >= 3
    
    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return self.features.shape[0]
    
    @property
    def n_regions(self) -> int:
        """Number of regions (1 for temporal-only data)."""
        if self.is_spatiotemporal:
            return self.features.shape[1]
        return 1
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        if self.is_spatiotemporal:
            return self.features.shape[2]
        return self.features.shape[1]
    
    def __repr__(self) -> str:
        shape_str = f"features={tuple(self.features.shape)}"
        if self.targets is not None:
            shape_str += f", targets={tuple(self.targets.shape)}"
        if self.graph is not None:
            shape_str += f", graph={tuple(self.graph.shape)}"
        data_type = "Spatiotemporal" if self.is_spatiotemporal else "Temporal"
        return f"TimeSeriesData({data_type}, {shape_str}, T={self.n_timesteps}, N={self.n_regions})"


@dataclass
class LoadedData:
    """
    Container for data loaded from external sources before creating TimeSeriesData.
    
    This is an intermediate representation used by loaders.
    """
    features: np.ndarray  # Shape: (T, N, F) or (T, F)
    targets: Optional[np.ndarray] = None  # Shape: (T, N, T_out) or (T, T_out) or (T, N) or (T,)
    timestamps: Optional[List[Any]] = None
    regions: Optional[List[Any]] = None
    graph: Optional[np.ndarray] = None
    dynamic_graph: Optional[np.ndarray] = None
    states: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_time_series_data(self) -> TimeSeriesData:
        """Convert to TimeSeriesData."""
        return TimeSeriesData(
            features=torch.FloatTensor(self.features) if self.features is not None else None,
            targets=torch.FloatTensor(self.targets) if self.targets is not None else None,
            timestamps=self.timestamps,
            regions=self.regions,
            graph=torch.FloatTensor(self.graph) if self.graph is not None else None,
            dynamic_graph=torch.FloatTensor(self.dynamic_graph) if self.dynamic_graph is not None else None,
            states=torch.FloatTensor(self.states) if self.states is not None else None,
            feature_names=self.feature_names,
            target_names=self.target_names,
            metadata=self.metadata
        )
