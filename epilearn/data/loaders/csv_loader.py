"""
CSV data loader for EpiLearn.

Provides flexible CSV loading with user-specified column mappings for
timestamps, regions, features, and targets.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings

from .base import BaseLoader, DataLoaderRegistry
from ..core import LoadedData


class CSVLoader(BaseLoader):
    """
    Load time series data from CSV files with flexible column mapping.
    
    Supports both temporal (single time series) and spatiotemporal (multiple regions)
    data formats. Users specify which columns contain timestamps, regions, features,
    and targets.
    
    Example CSV format for spatiotemporal data:
        date,region,cases,deaths,population
        2020-01-01,RegionA,10,1,100000
        2020-01-01,RegionB,5,0,50000
        2020-01-02,RegionA,15,2,100000
        2020-01-02,RegionB,8,1,50000
    
    Example usage:
        loader = CSVLoader(
            file_path="data.csv",
            timestamp_col="date",
            region_col="region",
            feature_cols=["cases", "deaths", "population"],
            target_cols=["cases"]
        )
        data = loader.load()
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        timestamp_col: str,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        region_col: Optional[str] = None,
        graph_file: Optional[Union[str, Path]] = None,
        graph_source_col: str = "source",
        graph_target_col: str = "target",
        graph_weight_col: Optional[str] = None,
        parse_dates: Union[bool, str] = 'auto',
        date_format: Optional[str] = None,
        fillna_method: Optional[str] = None,
        fillna_value: Optional[float] = None,
        strict_numeric: bool = True,
        sort_timestamps: bool = True,
        **read_csv_kwargs
    ):
        """
        Initialize CSV loader with column specifications.
        
        Args:
            file_path: Path to the CSV file
            timestamp_col: Column name containing timestamps
            feature_cols: List of column names for features
            target_cols: List of column names for targets (optional, can be same as features)
            region_col: Column name for region identifiers (optional, for spatiotemporal data)
            graph_file: Optional path to CSV file defining graph edges
            graph_source_col: Column name for source node in graph file
            graph_target_col: Column name for target node in graph file  
            graph_weight_col: Optional column name for edge weights
            parse_dates: Whether to parse timestamp column as dates.
                         'auto' (default): auto-detect if column looks like dates
                         True: always parse as dates
                         False: never parse as dates (keep as-is)
            date_format: Date format string for parsing (optional)
            fillna_method: Method to fill NaN values ('ffill', 'bfill', 'interpolate')
            fillna_value: Value to fill NaN values (used if fillna_method not specified)
            strict_numeric: If True, raise on non-numeric feature/target values
            sort_timestamps: Whether to sort timestamps (default True). Uses numeric sort
                            if timestamps are numeric, otherwise lexicographic.
            **read_csv_kwargs: Additional arguments passed to pd.read_csv
        """
        self.file_path = Path(file_path)
        self.timestamp_col = timestamp_col
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols) if target_cols else None
        self.region_col = region_col
        self.graph_file = Path(graph_file) if graph_file else None
        self.graph_source_col = graph_source_col
        self.graph_target_col = graph_target_col
        self.graph_weight_col = graph_weight_col
        self.parse_dates = parse_dates
        self.date_format = date_format
        self.fillna_method = fillna_method
        self.fillna_value = fillna_value
        self.strict_numeric = strict_numeric
        self.sort_timestamps = sort_timestamps
        self.read_csv_kwargs = read_csv_kwargs
    
    def load(self) -> LoadedData:
        """
        Load and process the CSV file.
        
        Returns:
            LoadedData object with processed data
        """
        # Read CSV (without date parsing initially for auto-detection)
        df = self._read_csv()
        
        # Validate columns
        self._validate_columns(df)
        
        # Convert numeric columns
        df = self._convert_to_numeric(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Parse timestamps (handles auto-detection)
        df = self._parse_timestamps(df)
        
        # Get unique timestamps with proper sorting
        unique_timestamps = self._get_sorted_timestamps(df)
        
        if self.region_col:
            unique_regions = sorted(df[self.region_col].unique().tolist())
            features, targets = self._pivot_spatiotemporal(df, unique_timestamps, unique_regions)
            regions = unique_regions
        else:
            features, targets = self._extract_temporal(df, unique_timestamps)
            regions = None
        
        # Load graph if provided
        graph = self._load_graph(regions) if self.graph_file else None
        
        return LoadedData(
            features=features,
            targets=targets,
            timestamps=unique_timestamps,
            regions=regions,
            graph=graph,
            feature_names=self.feature_cols.copy(),
            target_names=self.target_cols.copy() if self.target_cols else [],
            metadata={
                'source_file': str(self.file_path),
                'timestamp_col': self.timestamp_col,
                'region_col': self.region_col,
            }
        )
    
    def _read_csv(self) -> pd.DataFrame:
        """Read the CSV file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        # Don't parse dates here - we'll do it in _parse_timestamps for auto-detection
        return pd.read_csv(
            self.file_path,
            **self.read_csv_kwargs
        )
    
    def _validate_columns(self, df: pd.DataFrame):
        """Validate that required columns exist."""
        required_cols = [self.timestamp_col] + self.feature_cols
        if self.region_col:
            required_cols.append(self.region_col)
        if self.target_cols:
            required_cols.extend(self.target_cols)
        
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert feature and target columns to numeric."""
        numeric_cols = self.feature_cols + (self.target_cols or [])
        issues = []
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            original = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for conversion failures
            bad_mask = df[col].isna() & original.notna()
            if bad_mask.any():
                issues.append((col, int(bad_mask.sum())))
        
        if issues:
            msg = "Found non-numeric entries: " + ", ".join([f"{c}({n})" for c, n in issues])
            if self.strict_numeric:
                raise ValueError(msg + " Please ensure these columns are purely numeric.")
            else:
                warnings.warn(msg + " These will be treated as NaN.")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        numeric_cols = list(set(self.feature_cols + (self.target_cols or [])))
        
        if self.fillna_method:
            if self.fillna_method == 'ffill':
                df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
            elif self.fillna_method == 'bfill':
                df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
            elif self.fillna_method == 'interpolate':
                df[numeric_cols] = df[numeric_cols].interpolate()
        elif self.fillna_value is not None:
            df[numeric_cols] = df[numeric_cols].fillna(self.fillna_value)
        
        return df
    
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and validate timestamps with auto-detection."""
        ts_col = self.timestamp_col
        
        # Determine whether to parse as dates
        should_parse_dates = self.parse_dates
        
        if should_parse_dates == 'auto':
            # Auto-detect: check if values look like dates
            sample = df[ts_col].dropna().head(10)
            
            # Check if already numeric (int/float)
            if pd.api.types.is_numeric_dtype(df[ts_col]):
                should_parse_dates = False
            else:
                # Try parsing a sample as dates
                try:
                    pd.to_datetime(sample, errors='raise')
                    should_parse_dates = True
                except:
                    should_parse_dates = False
        
        if should_parse_dates:
            if self.date_format:
                df[ts_col] = pd.to_datetime(df[ts_col], format=self.date_format)
            else:
                df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        
        return df
    
    def _get_sorted_timestamps(self, df: pd.DataFrame) -> List[Any]:
        """Get unique timestamps with proper sorting."""
        unique_ts = df[self.timestamp_col].unique()
        
        if not self.sort_timestamps:
            return list(unique_ts)
        
        # Determine sorting method based on data type
        if pd.api.types.is_numeric_dtype(unique_ts):
            # Numeric: sort numerically
            return sorted(unique_ts.tolist())
        elif pd.api.types.is_datetime64_any_dtype(unique_ts):
            # Datetime: sort chronologically
            return sorted(unique_ts.tolist())
        else:
            # Try to convert to numeric for sorting
            try:
                numeric_ts = pd.to_numeric(unique_ts, errors='raise')
                sorted_idx = numeric_ts.argsort()
                return [unique_ts[i] for i in sorted_idx]
            except:
                # Fall back to lexicographic
                return sorted(unique_ts.tolist())
    
    def _pivot_spatiotemporal(
        self, 
        df: pd.DataFrame, 
        timestamps: List[Any], 
        regions: List[Any]
    ) -> tuple:
        """
        Pivot dataframe to spatiotemporal format.
        
        Returns:
            features: ndarray of shape (T, N, F)
            targets: ndarray of shape (T, N, T_out) or (T, N) or None
        """
        T = len(timestamps)
        N = len(regions)
        F = len(self.feature_cols)
        
        # Create timestamp and region index mappings
        ts_to_idx = {t: i for i, t in enumerate(timestamps)}
        region_to_idx = {r: i for i, r in enumerate(regions)}
        
        # Initialize feature array
        features = np.full((T, N, F), np.nan, dtype=np.float32)
        
        # Fill features
        for idx, row in df.iterrows():
            t_idx = ts_to_idx.get(row[self.timestamp_col])
            r_idx = region_to_idx.get(row[self.region_col])
            if t_idx is not None and r_idx is not None:
                for f_idx, col in enumerate(self.feature_cols):
                    features[t_idx, r_idx, f_idx] = row[col]
        
        # Handle targets
        targets = None
        if self.target_cols:
            T_out = len(self.target_cols)
            targets = np.full((T, N, T_out) if T_out > 1 else (T, N), np.nan, dtype=np.float32)
            
            for idx, row in df.iterrows():
                t_idx = ts_to_idx.get(row[self.timestamp_col])
                r_idx = region_to_idx.get(row[self.region_col])
                if t_idx is not None and r_idx is not None:
                    if T_out > 1:
                        for t_idx_out, col in enumerate(self.target_cols):
                            targets[t_idx, r_idx, t_idx_out] = row[col]
                    else:
                        targets[t_idx, r_idx] = row[self.target_cols[0]]
        
        # Fill NaN values after pivoting if fillna_value is specified
        if self.fillna_value is not None:
            features = np.nan_to_num(features, nan=self.fillna_value)
            if targets is not None:
                targets = np.nan_to_num(targets, nan=self.fillna_value)
        
        # Check for missing data
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            msg = f"Features have {nan_count} missing values after pivoting."
            if self.strict_numeric:
                raise ValueError(msg + " Some time-region combinations may be missing.")
            else:
                warnings.warn(msg)
        
        return features, targets
    
    def _extract_temporal(
        self, 
        df: pd.DataFrame, 
        timestamps: List[Any]
    ) -> tuple:
        """
        Extract temporal (non-spatial) data.
        
        Returns:
            features: ndarray of shape (T, F)
            targets: ndarray of shape (T, T_out) or (T,) or None
        """
        # Sort by timestamp
        df = df.sort_values(self.timestamp_col)
        
        # Extract features
        features = df[self.feature_cols].values.astype(np.float32)
        
        # Extract targets
        targets = None
        if self.target_cols:
            # Keep the trailing target axis even for a single target column:
            # generate_dataset() indexes targets as (time, region, target).
            targets = df[self.target_cols].values.astype(np.float32)

        return features, targets
    
    def _load_graph(self, regions: Optional[List[Any]]) -> Optional[np.ndarray]:
        """Load graph from separate file."""
        if not self.graph_file or not self.graph_file.exists():
            return None
        
        if not regions:
            warnings.warn("Graph file provided but no regions defined. Skipping graph loading.")
            return None
        
        graph_df = pd.read_csv(self.graph_file)
        
        # Validate columns
        required = [self.graph_source_col, self.graph_target_col]
        missing = [c for c in required if c not in graph_df.columns]
        if missing:
            raise ValueError(f"Graph file missing columns: {missing}")
        
        # Create adjacency matrix
        N = len(regions)
        region_to_idx = {r: i for i, r in enumerate(regions)}
        graph = np.zeros((N, N), dtype=np.float32)
        
        for _, row in graph_df.iterrows():
            src = row[self.graph_source_col]
            tgt = row[self.graph_target_col]
            
            if src in region_to_idx and tgt in region_to_idx:
                src_idx = region_to_idx[src]
                tgt_idx = region_to_idx[tgt]
                
                if self.graph_weight_col and self.graph_weight_col in row:
                    weight = row[self.graph_weight_col]
                else:
                    weight = 1.0
                
                graph[src_idx, tgt_idx] = weight
        
        return graph


# Register the CSV loader
DataLoaderRegistry.register(['csv'], CSVLoader)
