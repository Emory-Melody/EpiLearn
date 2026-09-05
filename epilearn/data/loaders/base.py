"""
Base loader interface for EpiLearn data loaders.

All data loaders should inherit from BaseLoader and implement the load() method.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from ..core import LoadedData, TimeSeriesData


class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    Subclasses must implement the load() method which returns a LoadedData object.
    """
    
    @abstractmethod
    def load(self) -> LoadedData:
        """
        Load data from the source and return a LoadedData object.
        
        Returns:
            LoadedData object containing the loaded data
        """
        pass
    
    def load_as_time_series(self) -> TimeSeriesData:
        """
        Load data and convert directly to TimeSeriesData.
        
        Returns:
            TimeSeriesData object ready for use with Dataset
        """
        loaded = self.load()
        return loaded.to_time_series_data()


class DataLoaderRegistry:
    """Registry for data loaders, allowing automatic format detection."""
    
    _loaders: Dict[str, type] = {}
    
    @classmethod
    def register(cls, extensions: List[str], loader_class: type):
        """Register a loader class for given file extensions."""
        for ext in extensions:
            cls._loaders[ext.lower().lstrip('.')] = loader_class
    
    @classmethod
    def get_loader(cls, file_path: Union[str, Path], **kwargs) -> Optional[BaseLoader]:
        """
        Get appropriate loader for a file based on its extension.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            Loader instance or None if no loader found
        """
        path = Path(file_path)
        ext = path.suffix.lower().lstrip('.')
        
        loader_class = cls._loaders.get(ext)
        if loader_class:
            return loader_class(file_path, **kwargs)
        return None
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls._loaders.keys())
