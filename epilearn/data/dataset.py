"""
Dataset class for EpiLearn.

This module provides the main Dataset class that wraps TimeSeriesData
and provides methods for data manipulation, windowing, and splits.
"""

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
import numpy as np
import os
import urllib.request
import pandas as pd
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple, Iterator

from .core import TimeSeriesData, LoadedData
from .loaders import CSVLoader, TensorLoader, NumpyLoader, load_from_file


def _download(url: str, dest: str):
    """Download `url` to `dest` atomically.

    Writing straight to `dest` means an interrupted download leaves a truncated
    file behind, and every later call sees it as a valid cache and fails with an
    unpickling error forever. Download to a sibling `.part` file and rename only
    on success, so a failed attempt leaves no cache at all.
    """
    tmp = dest + '.part'
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class Data:
    """Simple data container for batch processing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to(self, device):
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self


def custom_collate(batch):
    """Custom collate function for DataLoader."""
    elem = batch[0]
    collated = {}

    for key, value in elem.items():
        if torch.is_tensor(value):
            collated[key] = torch.stack([d[key] for d in batch])

    if 'edge_index' in elem and elem['edge_index'] is not None:
        edge_indices = []
        offset = 0
        for d in batch:
            edge_indices.append(d['edge_index'] + offset)
            offset += d['x'].size(0)
        collated['edge_index'] = torch.cat(edge_indices, dim=1)

    if 'edge_attr' in elem and elem['edge_attr'] is not None:
        collated['edge_attr'] = torch.cat([d['edge_attr'] for d in batch], dim=0)

    if 'x' in collated:
        batch_size = len(batch)
        num_nodes = [d['x'].size(0) for d in batch]
        collated['batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes)])

    data_obj = Data(**collated)

    if len(data_obj.x.shape) >= 3:
        data_obj.x = data_obj.x.view(-1, data_obj.x.size(-1))

    return data_obj


class Dataset(TorchDataset):
    """
    Dataset class for time series data with proper indexing and slicing.
    
    This class provides:
    - Loading from various sources (CSV, PT, NPY, or direct numpy/tensor)
    - Timestamp and region-based slicing
    - Train/val/test splitting with proper normalization
    - Sliding window dataset generation for training
    - Backward compatibility with existing code
    
    Parameters
    ----------
    name : str, optional
        Name of built-in dataset to load
    root : str, optional
        Root directory for downloads
    x : torch.Tensor or numpy.ndarray, optional
        Features of shape (T, N, F) for spatiotemporal or (T, F) for temporal
    y : torch.Tensor or numpy.ndarray, optional
        Targets
    graph : torch.Tensor or numpy.ndarray, optional
        Static adjacency matrix
    dynamic_graph : torch.Tensor or numpy.ndarray, optional
        Dynamic graph over time
    states : torch.Tensor or numpy.ndarray, optional
        State variables (e.g., SIR compartments)
    timestamps : list, optional
        List of timestamp values
    regions : list, optional
        List of region identifiers
    feature_names : list, optional
        Names of feature columns
    target_names : list, optional
        Names of target columns
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        root: str = './',
        x: Optional[Union[torch.Tensor, np.ndarray]] = None,
        states: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y: Optional[Union[torch.Tensor, np.ndarray]] = None,
        graph: Optional[Union[torch.Tensor, np.ndarray]] = None,
        dynamic_graph: Optional[Union[torch.Tensor, np.ndarray]] = None,
        edge_index: Optional[torch.LongTensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[List[Any]] = None,
        regions: Optional[List[Any]] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Core data attributes
        self.x = x
        self.y = y
        self.graph = graph
        self.dynamic_graph = dynamic_graph
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edge_attr = edge_attr
        self.states = states
        self.output_dim = None
        
        # Index attributes
        self.timestamps = timestamps
        self.regions = regions
        self.feature_names = feature_names or []
        self.target_names = target_names or []
        
        # Internal mappings (built lazily)
        self._timestamp_to_idx = None
        self._region_to_idx = None
        
        # Transformation settings
        self.transforms = None
        self.process_history = {}
        
        # Load built-in dataset if name provided
        if name is not None:
            self._load_builtin_dataset(name, root)
        
        # Post-initialization setup
        self._setup()
    
    def _setup(self):
        """Set up derived attributes after initialization."""
        # Convert to tensors if needed
        if self.x is not None and not isinstance(self.x, torch.Tensor):
            self.x = torch.FloatTensor(self.x)
        if self.y is not None and not isinstance(self.y, torch.Tensor):
            self.y = torch.FloatTensor(self.y)
        if self.graph is not None and not isinstance(self.graph, torch.Tensor):
            self.graph = torch.FloatTensor(self.graph)
        if self.dynamic_graph is not None and not isinstance(self.dynamic_graph, torch.Tensor):
            self.dynamic_graph = torch.FloatTensor(self.dynamic_graph)
        if self.states is not None and not isinstance(self.states, torch.Tensor):
            self.states = torch.FloatTensor(self.states)
        
        # Set output dimension
        if self.y is not None:
            self.output_dim = self.y.shape
        
        # Build edge index from graph if needed
        if self.graph is not None and self.edge_index is None:
            sparse_adj = torch.Tensor(self.graph).float().to_sparse()
            self.edge_index = sparse_adj.indices()
            self.edge_weight = sparse_adj.values()
        
        # Build index mappings
        self._build_index_mappings()
    
    def _build_index_mappings(self):
        """Build timestamp and region index mappings."""
        if self.x is None:
            return
        
        T = self.x.shape[0]
        
        # Set up timestamps
        if self.timestamps is None:
            self.timestamps = list(range(T))
        self._timestamp_to_idx = {t: i for i, t in enumerate(self.timestamps)}
        
        # Set up regions for spatiotemporal data
        if self.is_spatiotemporal:
            N = self.x.shape[1]
            if self.regions is None:
                self.regions = list(range(N))
            self._region_to_idx = {r: i for i, r in enumerate(self.regions)}
        else:
            self._region_to_idx = {}
    
    @property
    def is_spatiotemporal(self) -> bool:
        """Check if data is spatiotemporal (has region dimension)."""
        return self.x is not None and len(self.x.shape) >= 3
    
    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return self.x.shape[0] if self.x is not None else 0
    
    @property
    def n_regions(self) -> int:
        """Number of regions."""
        if self.x is None:
            return 0
        if self.is_spatiotemporal:
            return self.x.shape[1]
        return 1
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        if self.x is None:
            return 0
        if self.is_spatiotemporal:
            return self.x.shape[2]
        return self.x.shape[1]
    
    # ================== Index-Based Slicing Methods ==================
    
    def get_timestamp_idx(self, timestamp: Any) -> int:
        """Get index for a timestamp value."""
        if self._timestamp_to_idx is None:
            self._build_index_mappings()
        if timestamp in self._timestamp_to_idx:
            return self._timestamp_to_idx[timestamp]
        raise KeyError(f"Timestamp {timestamp} not found")
    
    def get_region_idx(self, region: Any) -> int:
        """Get index for a region identifier."""
        if self._region_to_idx is None:
            self._build_index_mappings()
        if region in self._region_to_idx:
            return self._region_to_idx[region]
        raise KeyError(f"Region {region} not found")
    
    def get_timestamp_range_indices(
        self,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        start_rate: Optional[float] = None,
        end_rate: Optional[float] = None,
        end_inclusive: bool = False
    ) -> Tuple[int, int]:
        """
        Get start and end indices for a timestamp range.
        
        Supports two modes:
        - Timestamp values: Use actual values from the data (dates, integers, etc.)
        - Rate-based: Proportions (0.0 to 1.0) of the total timesteps
        
        Timestamp values take precedence over rates if both are provided.
        
        Args:
            start: Start timestamp value (inclusive). Format should match timestamps
                   in the dataset (e.g., integers 0,1,2... or dates '2020-01-01').
            end: End timestamp value. By default exclusive, set end_inclusive=True
                 to include the end timestamp.
            start_rate: Start as proportion of total timesteps (0.0 to 1.0)
            end_rate: End as proportion of total timesteps (0.0 to 1.0)
            end_inclusive: If True, include the end timestamp in the slice.
                          Only applies when 'end' is a timestamp value, not rate.
            
        Returns:
            Tuple of (start_idx, end_idx) where end_idx is exclusive
            
        Example:
            # Using timestamp values (integers)
            start_idx, end_idx = dataset.get_timestamp_range_indices(start=10, end=50)
            
            # Using timestamp values (dates)
            start_idx, end_idx = dataset.get_timestamp_range_indices(
                start='2020-01-01', end='2020-06-01'
            )
            
            # Using rates
            start_idx, end_idx = dataset.get_timestamp_range_indices(
                start_rate=0.0, end_rate=0.7
            )
        """
        T = self.n_timesteps
        
        # Start index: timestamp takes precedence over rate
        if start is not None:
            start_idx = self.get_timestamp_idx(start)
        elif start_rate is not None:
            start_idx = int(T * start_rate)
        else:
            start_idx = 0
        
        # End index: timestamp takes precedence over rate
        if end is not None:
            end_idx = self.get_timestamp_idx(end)
            if end_inclusive:
                end_idx += 1  # Include the end timestamp
        elif end_rate is not None:
            end_idx = int(T * end_rate)
        else:
            end_idx = T
        
        # Clamp to valid range
        start_idx = max(0, min(start_idx, T))
        end_idx = max(start_idx, min(end_idx, T))
        
        return start_idx, end_idx
    
    def get_region_indices(self, regions: Optional[List[Any]] = None) -> Optional[List[int]]:
        """Get indices for a list of regions."""
        if regions is None:
            return None
        return [self.get_region_idx(r) for r in regions]
    
    def get_slice(
        self,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        start_rate: Optional[float] = None,
        end_rate: Optional[float] = None,
        end_inclusive: bool = False,
        regions: Optional[List[Any]] = None
    ) -> 'Dataset':
        """
        Get a slice of the dataset for a specific timestamp range and regions.
        
        This method supports both timestamp values and rate-based slicing,
        making it ideal for rolling evaluation where you progressively expand
        or slide the training/validation/test windows.
        
        Timestamp values should match the format in your data file:
        - Integer indices: 0, 1, 2, 3, ...
        - Date strings: '2020-01-01', '2020-01-02', ...
        - Pandas Timestamps: pd.Timestamp('2020-01-01'), ...
        
        Args:
            start: Start timestamp value (inclusive). Use actual values from
                   your data, not indices. If None, starts from beginning.
            end: End timestamp value. By default exclusive (like Python slicing).
                 Set end_inclusive=True to include this timestamp.
            start_rate: Alternative to 'start' - proportion of total timesteps (0.0-1.0)
            end_rate: Alternative to 'end' - proportion of total timesteps (0.0-1.0)
            end_inclusive: If True, include the 'end' timestamp in the slice.
            regions: List of region identifiers to include (None = all regions)
            
        Returns:
            New Dataset object with sliced data
            
        Example - Rolling Evaluation with timestamp values:
            # Timestamps are integers: 1, 2, 3, ..., 100
            
            # Train on timestamps 1-70 (inclusive)
            train = dataset.get_slice(start=1, end=70, end_inclusive=True)
            
            # Validate on 71-85
            val = dataset.get_slice(start=71, end=85, end_inclusive=True)
            
            # Test on 86-100
            test = dataset.get_slice(start=86, end=100, end_inclusive=True)
            
        Example - Rolling Evaluation with date timestamps:
            # Timestamps are dates: '2020-01-01', '2020-01-02', ...
            
            train = dataset.get_slice(start='2020-01-01', end='2020-07-01')
            val = dataset.get_slice(start='2020-07-01', end='2020-09-01')
            test = dataset.get_slice(start='2020-09-01')  # To the end
            
        Example - Rolling Evaluation with rates:
            train = dataset.get_slice(end_rate=0.7)
            val = dataset.get_slice(start_rate=0.7, end_rate=0.85)
            test = dataset.get_slice(start_rate=0.85)
            
        Example - Region subsetting:
            subset = dataset.get_slice(regions=['RegionA', 'RegionB'])
            
        Example - Combined timestamp and region slicing:
            subset = dataset.get_slice(start=50, end=100, regions=['RegionA'])
        """
        start_idx, end_idx = self.get_timestamp_range_indices(
            start, end, start_rate, end_rate, end_inclusive
        )
        region_indices = self.get_region_indices(regions)
        
        # Slice timestamps
        sliced_timestamps = self.timestamps[start_idx:end_idx] if self.timestamps else None
        
        # Slice features
        if self.is_spatiotemporal:
            if region_indices is not None:
                sliced_x = self.x[start_idx:end_idx, region_indices, :]
                sliced_regions = [self.regions[i] for i in region_indices]
            else:
                sliced_x = self.x[start_idx:end_idx]
                sliced_regions = self.regions.copy() if self.regions else None
        else:
            sliced_x = self.x[start_idx:end_idx] if self.x is not None else None
            sliced_regions = None
        
        # Slice targets
        sliced_y = None
        if self.y is not None:
            if self.is_spatiotemporal and region_indices is not None and len(self.y.shape) >= 2:
                sliced_y = self.y[start_idx:end_idx, region_indices]
            else:
                sliced_y = self.y[start_idx:end_idx]
        
        # Slice states
        sliced_states = None
        if self.states is not None:
            if self.is_spatiotemporal and region_indices is not None:
                sliced_states = self.states[start_idx:end_idx, region_indices]
            else:
                sliced_states = self.states[start_idx:end_idx]
        
        # Slice dynamic graph
        sliced_dynamic_graph = None
        if self.dynamic_graph is not None:
            sliced_dynamic_graph = self.dynamic_graph[start_idx:end_idx]
            if region_indices is not None:
                sliced_dynamic_graph = sliced_dynamic_graph[:, region_indices][:, :, region_indices]
        
        # Handle static graph
        sliced_graph = None
        if self.graph is not None:
            if region_indices is not None:
                sliced_graph = self.graph[region_indices][:, region_indices]
            else:
                sliced_graph = self.graph.clone()
        
        # Create new dataset
        new_dataset = Dataset(
            x=sliced_x,
            y=sliced_y,
            graph=sliced_graph,
            dynamic_graph=sliced_dynamic_graph,
            states=sliced_states,
            timestamps=sliced_timestamps,
            regions=sliced_regions,
            feature_names=self.feature_names.copy() if self.feature_names else None,
            target_names=self.target_names.copy() if self.target_names else None,
        )
        
        # Copy transforms and process history
        new_dataset.transforms = self.transforms
        new_dataset.process_history = self.process_history.copy() if self.process_history else {}
        
        return new_dataset
    
    def rolling_splits(
        self,
        train_size: int,
        test_size: int,
        step_size: Optional[int] = None,
        val_size: int = 0,
        expanding: bool = True,
        regions: Optional[List[Any]] = None
    ) -> Iterator[Tuple['Dataset', Optional['Dataset'], 'Dataset']]:
        """
        Generate rolling train/val/test splits for time series cross-validation.
        
        This is the recommended way to evaluate forecasting models, as it respects
        the temporal ordering of the data and avoids data leakage.
        
        Two modes are supported:
        - Expanding window: Training window grows, test window slides
        - Sliding window: Both train and test windows slide (fixed train size)
        
        Args:
            train_size: Initial training window size (number of timesteps)
            test_size: Test window size (number of timesteps)
            step_size: How many timesteps to advance each iteration.
                      Defaults to test_size (non-overlapping test windows).
            val_size: Validation window size between train and test. Default 0.
            expanding: If True, training window expands. If False, slides.
            regions: Optional list of regions to include in all splits.
            
        Yields:
            Tuple of (train_dataset, val_dataset, test_dataset)
            val_dataset is None if val_size=0
            
        Example - Basic rolling evaluation:
            for train, val, test in dataset.rolling_splits(
                train_size=100, test_size=20, val_size=10
            ):
                # Train on train, validate on val, evaluate on test
                model.fit(train)
                model.evaluate(test)
                
        Example - Sliding window (fixed train size):
            for train, _, test in dataset.rolling_splits(
                train_size=100, test_size=20, expanding=False
            ):
                model.fit(train)
                model.evaluate(test)
                
        Example - Walk-forward validation:
            results = []
            for train, val, test in dataset.rolling_splits(
                train_size=200, test_size=7, val_size=14, step_size=7
            ):
                # train_size expands, we step forward 7 days each iteration
                metrics = train_and_evaluate(train, val, test)
                results.append(metrics)
        """
        if step_size is None:
            step_size = test_size
            
        T = self.n_timesteps
        timestamps = self.timestamps
        
        # Starting position
        if expanding:
            train_start_idx = 0
        else:
            train_start_idx = 0  # Will slide with test
            
        train_end_idx = train_size
        
        while train_end_idx + val_size + test_size <= T:
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + val_size
            test_start_idx = val_end_idx
            test_end_idx = test_start_idx + test_size
            
            # Get slices using timestamp values
            train_start = timestamps[train_start_idx]
            train_end = timestamps[train_end_idx - 1]  # inclusive
            
            train_split = self.get_slice(
                start=train_start, end=train_end, 
                end_inclusive=True, regions=regions
            )
            
            if val_size > 0:
                val_start = timestamps[val_start_idx]
                val_end = timestamps[val_end_idx - 1]
                val_split = self.get_slice(
                    start=val_start, end=val_end,
                    end_inclusive=True, regions=regions
                )
            else:
                val_split = None
                
            test_start = timestamps[test_start_idx]
            test_end = timestamps[test_end_idx - 1]
            test_split = self.get_slice(
                start=test_start, end=test_end,
                end_inclusive=True, regions=regions
            )
            
            yield train_split, val_split, test_split

            # Advance windows
            if expanding:
                train_end_idx += step_size
            else:
                train_start_idx += step_size
                train_end_idx += step_size

        # Final partial fold: use remaining data as a shorter test set if at
        # least half the requested test_size remains after the validation window.
        remaining_test = T - train_end_idx - val_size
        if remaining_test >= test_size // 2 and remaining_test < test_size:
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + val_size
            test_start_idx = val_end_idx
            test_end_idx = T

            train_start = timestamps[train_start_idx]
            train_end = timestamps[train_end_idx - 1]
            train_split = self.get_slice(
                start=train_start, end=train_end,
                end_inclusive=True, regions=regions
            )

            if val_size > 0:
                val_start = timestamps[val_start_idx]
                val_end = timestamps[val_end_idx - 1]
                val_split = self.get_slice(
                    start=val_start, end=val_end,
                    end_inclusive=True, regions=regions
                )
            else:
                val_split = None

            test_start = timestamps[test_start_idx]
            test_end = timestamps[test_end_idx - 1]
            test_split = self.get_slice(
                start=test_start, end=test_end,
                end_inclusive=True, regions=regions
            )

            yield train_split, val_split, test_split

    # ================== Loading Methods ==================
    
    @classmethod
    def from_time_series_data(cls, ts_data: TimeSeriesData) -> 'Dataset':
        """Create Dataset from TimeSeriesData object."""
        return cls(
            x=ts_data.features,
            y=ts_data.targets,
            graph=ts_data.graph,
            dynamic_graph=ts_data.dynamic_graph,
            states=ts_data.states,
            timestamps=ts_data.timestamps,
            regions=ts_data.regions,
            feature_names=ts_data.feature_names,
            target_names=ts_data.target_names,
        )
    
    @classmethod
    def from_csv(
        cls,
        file_path: str,
        timestamp_col: str,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        region_col: Optional[str] = None,
        graph_file: Optional[str] = None,
        **kwargs
    ) -> 'Dataset':
        """
        Load dataset from CSV file with flexible column mapping.
        
        Args:
            file_path: Path to CSV file
            timestamp_col: Column name for timestamps
            feature_cols: List of feature column names
            target_cols: List of target column names. If None, uses the first 
                        feature column as target (common for forecasting tasks)
            region_col: Column name for regions (for spatiotemporal data).
                       If provided, creates spatiotemporal dataset.
            graph_file: Optional path to CSV with graph edges
            **kwargs: Additional arguments for CSVLoader
            
        Returns:
            Dataset instance
            
        Example:
            # Minimal usage - timestamp, region, features (first feature = target)
            dataset = Dataset.from_csv(
                "data.csv",
                timestamp_col="date",
                feature_cols=["cases", "deaths"],  # 'cases' will be target
                region_col="state"
            )
            
            # Explicit target
            dataset = Dataset.from_csv(
                "data.csv",
                timestamp_col="date",
                feature_cols=["cases", "deaths", "tests"],
                target_cols=["cases"],
                region_col="state",
                graph_file="edges.csv"
            )
            
            # Temporal only (no region)
            dataset = Dataset.from_csv(
                "data.csv",
                timestamp_col="date",
                feature_cols=["value"]
            )
        """
        # Default target_cols to first feature column if not specified
        if target_cols is None:
            target_cols = [feature_cols[0]]
        
        loader = CSVLoader(
            file_path=file_path,
            timestamp_col=timestamp_col,
            feature_cols=feature_cols,
            target_cols=target_cols,
            region_col=region_col,
            graph_file=graph_file,
            **kwargs
        )
        ts_data = loader.load_as_time_series()
        return cls.from_time_series_data(ts_data)
    
    # Alias for backward compatibility
    load_from_csv = from_csv
    
    @classmethod
    def from_tensor(cls, file_path: str, **kwargs) -> 'Dataset':
        """Load dataset from PyTorch tensor file."""
        loader = TensorLoader(file_path=file_path, **kwargs)
        ts_data = loader.load_as_time_series()
        return cls.from_time_series_data(ts_data)
    
    @classmethod
    def from_numpy(cls, file_path: str, **kwargs) -> 'Dataset':
        """Load dataset from NumPy file."""
        loader = NumpyLoader(file_path=file_path, **kwargs)
        ts_data = loader.load_as_time_series()
        return cls.from_time_series_data(ts_data)
    
    def _load_builtin_dataset(self, name: str, root: str):
        """Load a built-in dataset by name."""
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        if name == 'JHU_covid':
            self._load_jhu_covid(root)
        elif name == 'Measles':
            self._load_measles(root)
        elif name == 'Tycho_v1':
            self._load_tycho_v1(root)
        elif name.split('_')[0] == 'Covid':
            country = name.split('_')[1]
            self._load_covid_country(root, country)
        else:
            raise ValueError(f"Dataset '{name}' not found!")
    
    def _load_jhu_covid(self, root: str):
        """Load JHU COVID dataset."""
        filepath = f"{root}/JHU_covid.pt"
        if not os.path.exists(filepath):
            print("downloading JHU Covid Dataset")
            url = "https://drive.google.com/uc?export=download&id=13i9OpTweVYOvSOET-91-ZNVMJRURwxb0"
            _download(url, filepath)
        
        data = torch.load(filepath, weights_only=False)
        self.x = data.get('features', data.get('x')).float()
        self.feature_names = data.get('feature_names', [])
        self.timestamps = data.get('index', list(range(self.x.shape[0])))
    
    def _load_measles(self, root: str):
        """Load Measles dataset."""
        filepath = f"{root}/measles.pt"
        if not os.path.exists(filepath):
            print("downloading Measles Dataset")
            url = "https://drive.google.com/uc?export=download&id=1-kLtvyUGN_mYJL5MgafEAd30KyfSmE4U"
            _download(url, filepath)
        
        data = torch.load(filepath, weights_only=False)
        self.x = torch.FloatTensor(np.array(data['weekly_infection'])).T
        self.feature_names = list(data['weekly_infection'].columns)
        # Store additional metadata
        self.metadata = {
            'anual_population': data.get('anual_population'),
            'anual_birth': data.get('anual_birth'),
            'coordinates': data.get('coordinates')
        }
    
    def _load_tycho_v1(self, root: str):
        """Load Tycho v1 dataset."""
        filepath = f"{root}/Tycho_v1.pt"
        if not os.path.exists(filepath):
            print("downloading Tycho_v1 Dataset")
            url = "https://drive.google.com/uc?export=download&id=13gHDO6Rh5gZwqo8MLDyyiLtPCd9gD0nD"
            _download(url, filepath)
        
        data = torch.load(filepath, weights_only=False)
        self.feature_names = list(data.keys())
        self.x = []
        for v in data.values():
            self.x.append(v.float())
    
    # Countries available through `Dataset(name='Covid_<Country>')`. The first
    # three ship a static adjacency graph, the rest a dynamic one.
    COVID_COUNTRIES = (
        'Brazil', 'Austria', 'China',
        'England', 'France', 'Italy', 'NewZealand', 'Spain',
    )

    def _load_covid_country(self, root: str, country: str):
        """Load COVID data for a specific country."""
        if country not in self.COVID_COUNTRIES:
            raise ValueError(
                f"Covid dataset for '{country}' is not available. "
                f"Supported countries: {', '.join(self.COVID_COUNTRIES)}. "
                "(Note the capital Z in 'NewZealand'.)"
            )

        static_path = f"{root}/covid_static.pt"
        dynamic_path = f"{root}/covid_dynamic.pt"

        if not os.path.exists(static_path):
            print("downloading Covid Static Dataset")
            url = "https://drive.google.com/uc?export=download&id=1-l1yVWlKxB0VLwj0IqJRAbruUvyrBUUI"
            _download(url, static_path)
        
        if not os.path.exists(dynamic_path):
            print("downloading Covid Dynamic Dataset")
            url = "https://drive.google.com/uc?export=download&id=1-lUm1uaSDgbto2IV8aalZ3_lXh7LS7z3"
            _download(url, dynamic_path)
        
        try:
            static_data = torch.load(static_path, weights_only=False)
            data = static_data[country]
            self.graph = data['graph']
        except:
            dynamic_data = torch.load(dynamic_path, weights_only=False)
            data = dynamic_data[country]
            self.dynamic_graph = data['Dynamic_graph']
        
        self.x = data['features']
        self.feature_names = data.get('feature_names', [])
        self.timestamps = data.get('time_stamp', list(range(self.x.shape[0])))
    
    def load_toy_dataset(self):
        """Load toy dataset for testing."""
        datasets_dir = "./datasets"
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir, exist_ok=True)
        
        features_path = f"{datasets_dir}/features.npy"
        graphs_path = f"{datasets_dir}/graphs.npy"
        
        if not os.path.exists(features_path):
            print("downloading toy features")
            url = "https://drive.google.com/uc?export=download&id=10VRjabU1m0pluQKOTQ-GKxF3sK9bLFYF"
            _download(url, features_path)
        
        if not os.path.exists(graphs_path):
            print("downloading toy graphs")
            url = "https://drive.google.com/uc?export=download&id=10ZR4k19wdWXQdPN53Tz3QAQxBiZUXEPr"
            _download(url, graphs_path)
        
        graph_data = np.load(graphs_path)
        feature_data = np.load(features_path, allow_pickle=True)
        
        self.x = torch.FloatTensor(feature_data.tolist()['node'])
        self.y = torch.FloatTensor(feature_data.tolist()['node'])[:, :, 0]
        self.dynamic_graph = torch.FloatTensor(feature_data.tolist()['od'])
        self.states = torch.FloatTensor(feature_data.tolist()['SIR'])
        self.graph = torch.FloatTensor(graph_data)
        self.edge_index = self.graph.to_sparse_coo().indices()
        self.edge_weight = self.graph.to_sparse_coo().values()
        
        self._build_index_mappings()
    
    # ================== Transformation Methods ==================
    
    def set_transforms(self, transforms, apply_now: bool = False):
        """
        Set preprocessing transformations for the dataset.
        
        This method properly sets transformation configurations that will be applied
        during training/evaluation. Optionally applies transformations immediately.
        
        Parameters
        ----------
        transforms : Compose
            A Compose object containing transformations for different data components.
            Expected format:
                transforms.Compose({
                    "target": [transforms.normalize_target()],
                    "features": [transforms.normalize_feat()],
                    "graph": [transforms.normalize_adj()],
                    "dynamic_graph": [transforms.normalize_adj()],
                })
        apply_now : bool, default False
            If True, immediately apply transformations to the current data.
            If False, transformations are stored and applied during training/splits.
            
        Returns
        -------
        self : Dataset
            Returns self for method chaining.
            
        Examples
        --------
        >>> from epilearn.utils import transforms
        >>> transformation = transforms.Compose({
        ...     "target": [transforms.normalize_target()],
        ...     "features": [transforms.normalize_feat()],
        ...     "graph": [transforms.normalize_adj()],
        ... })
        >>> dataset.set_transforms(transformation)
        >>> 
        >>> # Or apply immediately
        >>> dataset.set_transforms(transformation, apply_now=True)
        """
        self.transforms = transforms
        
        if apply_now:
            self.apply_transforms()
            
        return self
    
    def apply_transforms(self, inplace: bool = True):
        """
        Apply the stored transformations to the dataset.
        
        Parameters
        ----------
        inplace : bool, default True
            If True, modify the dataset in place.
            If False, return a new dataset with transformations applied.
            
        Returns
        -------
        Dataset or tuple
            If inplace=True, returns self.
            If inplace=False, returns (transformed_data_dict, process_history).
        """
        if self.transforms is None:
            warnings.warn("No transforms set. Call set_transforms() first.")
            return self if inplace else ({}, {})
        
        input_data = {
            "features": self.x,
            "target": self.y,
            "graph": self.graph,
            "dynamic_graph": self.dynamic_graph,
            "states": self.states
        }
        
        transformed_data, process_history = self.transforms(input_data)
        self.process_history = process_history
        
        if inplace:
            # Update dataset attributes with transformed data
            if 'features' in transformed_data and transformed_data['features'] is not None:
                self.x = transformed_data['features']
            if 'target' in transformed_data and transformed_data['target'] is not None:
                self.y = transformed_data['target']
            if 'graph' in transformed_data and transformed_data['graph'] is not None:
                self.graph = transformed_data['graph']
            if 'dynamic_graph' in transformed_data and transformed_data['dynamic_graph'] is not None:
                self.dynamic_graph = transformed_data['dynamic_graph']
            if 'states' in transformed_data and transformed_data['states'] is not None:
                self.states = transformed_data['states']
            return self
        else:
            return transformed_data, process_history
    
    def get_transforms(self):
        """
        Get the current transformation configuration.
        
        Returns
        -------
        Compose or None
            The current transforms object, or None if not set.
        """
        return self.transforms
    
    def get_process_history(self):
        """
        Get the processing history (normalization statistics).
        
        Returns
        -------
        dict
            Dictionary containing normalization statistics like 
            'feat_mean', 'feat_std', 'target_mean', 'target_std'.
        """
        return self.process_history
    
    # def get_transformed(self, input_data=None, transformations=None):
    #     """Apply transformations to data."""
    #     if transformations is None:
    #         transformations = self.transforms
        
    #     if input_data is None:
    #         input_data = {
    #             "features": self.x,
    #             "target": self.y,
    #             "graph": self.graph,
    #             "dynamic_graph": self.dynamic_graph,
    #             "states": self.states
    #         }
        
    #     if transformations is not None:
    #         input_data, self.process_history = transformations(input_data)
        
    #     return input_data, self.process_history
    
    def _apply_normalization_with_stats(self, data, mean, std):
        """Apply normalization using pre-computed mean and std."""
        if data is None:
            return None
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        
        if len(data.shape) == 2:
            normalized = (data - mean) / std
        elif len(data.shape) == 3:
            normalized = (data - mean.unsqueeze(0).unsqueeze(0)) / std.unsqueeze(0).unsqueeze(0)
        elif len(data.shape) == 4:
            normalized = (data - mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)) / std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            normalized = data
        
        normalized[torch.where(torch.isnan(normalized))] = 0
        return normalized
    
    # ================== Split Generation Methods ==================

    
    def generate_dataset(
        self,
        X=None,
        Y=None,
        states=None,
        dynamic_adj=None,
        adj=None,
        lookback_window_size=1,
        horizon_size=1,
        interval=None,
        ahead=0,
        permute=False,
        region_idx=None
    ):
        """
        Generate sliding window dataset for training/evaluation.
        
        Args:
            X: Features tensor of shape (T, N, F) or (T, F)
            Y: Targets tensor
            states: State variables (e.g., SIR compartments)
            dynamic_adj: Dynamic adjacency matrices
            adj: Static adjacency matrix
            lookback_window_size: Number of timesteps in input window
            horizon_size: Number of timesteps to predict
            interval: Offset to align predictions across different lookback sizes.
                     When comparing models with different lookbacks, set this to
                     (max_lookback - current_lookback) so all models predict the
                     same time points. Default None means no offset.
            ahead: Gap between input window and prediction window
            permute: Whether to permute dimensions
            region_idx: Optional region indices to select
            
        Returns:
            Dictionary with 'features', 'targets', 'states', 'dynamic_graph', 'graph'
            
        Example - Aligning different lookbacks:
            # With max_lookback=21, both produce samples predicting same time points
            split_21 = ds.generate_dataset(lookback_window_size=21, interval=0)
            split_7 = ds.generate_dataset(lookback_window_size=7, interval=14)
            
            # split_21 sample 0: input [0:21], predict [21:28]
            # split_7 sample 0:  input [14:21], predict [21:28]  <- same prediction!
        """
        if X is None:
            X = self.x
        if Y is None:
            Y = self.y
        
        # Apply interval offset to align predictions across different lookbacks
        # interval = max_lookback - current_lookback
        offset = interval if interval is not None and interval > 0 else 0
        
        if horizon_size > 0:
            total_window = lookback_window_size + ahead + horizon_size
        else:
            total_window = lookback_window_size
            assert total_window + ahead + horizon_size >= 0
        
        # Generate indices with offset applied
        # Starting from 'offset' ensures predictions align with max_lookback
        n_samples = X.shape[0] - offset - total_window + 1
        if n_samples <= 0:
            return {
                'features': torch.Tensor([[[]]]),
                'targets': torch.Tensor([[[]]]),
                'states': None,
                'dynamic_graph': None,
                'graph': adj
            }
        
        indices = [(offset + i, offset + i + total_window) for i in range(n_samples)]
        
        # Extract targets - prediction window starts at (start + lookback + ahead)
        target = []
        for i, j in indices:
            start = min(i + lookback_window_size, i + lookback_window_size + ahead + horizon_size)
            end = max(i + lookback_window_size, i + lookback_window_size + ahead + horizon_size)
            target.append(Y[start:end])
        
        targets = torch.stack(target) if len(target) > 0 else torch.Tensor([[[]]])
        if not permute:
            targets = targets.transpose(1, 2)
        
        # Extract input features - input window is [i : i + lookback]
        input_list = [[None, X], [None, states], [None, dynamic_adj]]
        for m, inputs in enumerate(input_list):
            if inputs[1] is not None:
                tmp = []
                for i, j in indices:
                    tmp.append(inputs[1][i:i + lookback_window_size])
                input_list[m][0] = torch.stack(tmp) if len(tmp) else torch.Tensor([[[]]])
                if permute:
                    inputs[0] = inputs[0].transpose(1, 2)
        
        input_list[2][0] = input_list[2][0].squeeze() if input_list[2][0] is not None else None
        
        if region_idx is not None:
            input_list[0][0] = input_list[0][0][:, region_idx, :]
            targets = targets[:, region_idx, :]
            input_list[1][0] = input_list[1][0][:, region_idx, :] if input_list[1][0] is not None else None
            input_list[2][0] = input_list[2][0][:, region_idx, :] if input_list[2][0] is not None else None
        
        return {
            'features': input_list[0][0],
            'targets': targets,
            'states': input_list[1][0],
            'dynamic_graph': input_list[2][0],
            'graph': adj
        }
    
    # ================== DataLoader Methods ==================
    
    def __getitem__(self, index):
        """Get item for DataLoader."""
        item = {'x': self.x[index]}
        if self.y is not None:
            item['y'] = self.y[index]
        if self.dynamic_graph is not None:
            item['dynamic_graph'] = self.dynamic_graph[index]
        if self.edge_index is not None:
            item['edge_index'] = self.edge_index
        if self.edge_weight is not None:
            item['edge_attr'] = self.edge_weight
        return item
    
    def __len__(self):
        return self.x.size(0) if self.x is not None else 0
    
    def __repr__(self):
        shape_str = f"x={tuple(self.x.shape)}" if self.x is not None else "x=None"
        if self.y is not None:
            shape_str += f", y={tuple(self.y.shape)}"
        if self.graph is not None:
            shape_str += f", graph={tuple(self.graph.shape)}"
        data_type = "Spatiotemporal" if self.is_spatiotemporal else "Temporal"
        return f"Dataset({data_type}, {shape_str})"
    
    # ================== Utility Methods ==================
    
    def to_time_series_data(self) -> TimeSeriesData:
        """Convert to TimeSeriesData object."""
        return TimeSeriesData(
            features=self.x,
            targets=self.y,
            timestamps=self.timestamps,
            regions=self.regions,
            graph=self.graph,
            dynamic_graph=self.dynamic_graph,
            states=self.states,
            feature_names=self.feature_names or [],
            target_names=self.target_names or [],
        )
    
    def download(self):
        """Download dataset (to be implemented by subclasses)."""
        pass
    
    def save(self, path: str):
        """Save dataset to file."""
        torch.save({
            'x': self.x,
            'y': self.y,
            'graph': self.graph,
            'dynamic_graph': self.dynamic_graph,
            'states': self.states,
            'timestamps': self.timestamps,
            'regions': self.regions,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'Dataset':
        """Load dataset from file."""
        data = torch.load(path, weights_only=False)
        return cls(
            x=data.get('x'),
            y=data.get('y'),
            graph=data.get('graph'),
            dynamic_graph=data.get('dynamic_graph'),
            states=data.get('states'),
            timestamps=data.get('timestamps'),
            regions=data.get('regions'),
            feature_names=data.get('feature_names'),
            target_names=data.get('target_names'),
        )








    
    # def get_split(self, inputs, idx1, idx2):
    #     """Split inputs at given indices."""
    #     if inputs is None:
    #         return None, None, None
        
    #     train = inputs[:idx1, ...]
    #     val = inputs[idx1:idx2, ...]
    #     test = inputs[idx2:, ...]
        
    #     return train, val, test
    
    # def generate_splits(self, train_rate=0.6, val_rate=0.1):
    #     """Generate train/val/test splits with proper normalization."""
    #     split_line1 = int(self.x.shape[0] * train_rate)
    #     split_line2 = int(self.x.shape[0] * (train_rate + val_rate))
        
    #     # Split raw data
    #     train_feat_raw, val_feat_raw, test_feat_raw = self.get_split(self.x, split_line1, split_line2)
    #     train_tgt_raw, val_tgt_raw, test_tgt_raw = self.get_split(self.y, split_line1, split_line2)
    #     train_states_raw, val_states_raw, test_states_raw = self.get_split(self.states, split_line1, split_line2)
    #     train_dyn_raw, val_dyn_raw, test_dyn_raw = self.get_split(self.dynamic_graph, split_line1, split_line2)
        
    #     # Build train dataset dict
    #     train_raw = {
    #         "features": train_feat_raw,
    #         "target": train_tgt_raw,
    #         "graph": self.graph,
    #         "dynamic_graph": train_dyn_raw,
    #         "states": train_states_raw
    #     }
        
    #     # Apply transforms to training data
    #     transformed_train, process_history = self.get_transformed(train_raw)
        
    #     adj_static = transformed_train.get('graph', self.graph)
        
    #     # Get normalization stats
    #     feat_mean = process_history.get('feat_mean')
    #     feat_std = process_history.get('feat_std')
    #     target_mean = process_history.get('target_mean')
    #     target_std = process_history.get('target_std')
        
    #     # Apply same normalization to val/test
    #     if feat_mean is not None and feat_std is not None:
    #         val_features = self._apply_normalization_with_stats(val_feat_raw, feat_mean, feat_std)
    #         test_features = self._apply_normalization_with_stats(test_feat_raw, feat_mean, feat_std)
    #     else:
    #         val_features = torch.Tensor(val_feat_raw) if val_feat_raw is not None else None
    #         test_features = torch.Tensor(test_feat_raw) if test_feat_raw is not None else None
        
    #     if target_mean is not None and target_std is not None:
    #         val_target = self._apply_normalization_with_stats(val_tgt_raw, target_mean, target_std)
    #         test_target = self._apply_normalization_with_stats(test_tgt_raw, target_mean, target_std)
    #     else:
    #         val_target = torch.Tensor(val_tgt_raw) if val_tgt_raw is not None else None
    #         test_target = torch.Tensor(test_tgt_raw) if test_tgt_raw is not None else None
        
    #     val_states = torch.Tensor(val_states_raw) if val_states_raw is not None else None
    #     test_states = torch.Tensor(test_states_raw) if test_states_raw is not None else None
        
    #     # Handle dynamic graph
    #     train_dyn_adj = transformed_train.get('dynamic_graph')
    #     if train_dyn_adj is not None:
    #         n_nodes = transformed_train['features'].shape[1] if len(transformed_train['features'].shape) >= 3 else transformed_train['features'].shape[2]
    #         train_dyn_adj = train_dyn_adj.view(-1, n_nodes, n_nodes)
    #         val_dyn_adj = torch.Tensor(val_dyn_raw).view(-1, n_nodes, n_nodes) if val_dyn_raw is not None else None
    #         test_dyn_adj = torch.Tensor(test_dyn_raw).view(-1, n_nodes, n_nodes) if test_dyn_raw is not None else None
    #     else:
    #         val_dyn_adj = None
    #         test_dyn_adj = None
        
    #     train_dataset = {
    #         "features": transformed_train['features'],
    #         "target": transformed_train['target'],
    #         "graph": adj_static,
    #         "dynamic_graph": train_dyn_adj,
    #         "states": transformed_train.get('states')
    #     }
    #     val_dataset = {
    #         "features": val_features,
    #         "target": val_target,
    #         "graph": adj_static,
    #         "dynamic_graph": val_dyn_adj,
    #         "states": val_states
    #     }
    #     test_dataset = {
    #         "features": test_features,
    #         "target": test_target,
    #         "graph": adj_static,
    #         "dynamic_graph": test_dyn_adj,
    #         "states": test_states
    #     }
        
    #     return train_dataset, val_dataset, test_dataset, process_history
    
    # def get_splits(
    #     self,
    #     train_rate=0.6,
    #     val_rate=0.2,
    #     lookback=None,
    #     horizon=None,
    #     ahead=None,
    #     k_fold=0,
    #     region_idx=None,
    #     permute=False
    # ):
    #     """Generate splits with sliding window dataset generation."""
    #     train_dataset, _, test_dataset, process_history = self.generate_splits(train_rate=train_rate, val_rate=0)
        
    #     train_splits = self.generate_dataset(
    #         X=train_dataset['features'],
    #         Y=train_dataset['target'],
    #         states=train_dataset['states'],
    #         dynamic_adj=train_dataset['dynamic_graph'],
    #         adj=train_dataset['graph'],
    #         lookback_window_size=lookback,
    #         horizon_size=horizon,
    #         ahead=ahead,
    #         permute=permute,
    #         region_idx=region_idx
    #     )
        
    #     test_splits = self.generate_dataset(
    #         X=test_dataset['features'],
    #         Y=test_dataset['target'],
    #         states=test_dataset['states'],
    #         dynamic_adj=test_dataset['dynamic_graph'],
    #         adj=test_dataset['graph'],
    #         lookback_window_size=lookback,
    #         horizon_size=horizon,
    #         ahead=ahead,
    #         permute=permute,
    #         region_idx=region_idx
    #     )
        
    #     if k_fold == 0:
    #         return train_splits, test_splits, process_history
        
    #     # Generate k-fold splits
    #     folds = []
    #     train_size = train_splits['features'].shape[0]
    #     val_fold_size = train_size * val_rate
    #     base_train_size = train_size - k_fold * val_fold_size
        
    #     for k in range(k_fold):
    #         train_end = int(base_train_size + k * val_fold_size)
    #         val_end = int(train_end + val_fold_size)
            
    #         fold_train = {
    #             "features": train_splits['features'][:train_end],
    #             "targets": train_splits['targets'][:train_end],
    #             "states": train_splits['states'][:train_end] if train_splits['states'] is not None else None,
    #             "dynamic_graph": train_splits['dynamic_graph'][:train_end] if train_splits['dynamic_graph'] is not None else None,
    #             "graph": train_splits['graph']
    #         }
            
    #         fold_val = {
    #             "features": train_splits['features'][train_end:val_end],
    #             "targets": train_splits['targets'][train_end:val_end],
    #             "states": train_splits['states'][train_end:val_end] if train_splits['states'] is not None else None,
    #             "dynamic_graph": train_splits['dynamic_graph'][train_end:val_end] if train_splits['dynamic_graph'] is not None else None,
    #             "graph": train_splits['graph']
    #         }
            
    #         folds.append({'train': fold_train, 'val': fold_val})
        
    #     return folds, test_splits, process_history
