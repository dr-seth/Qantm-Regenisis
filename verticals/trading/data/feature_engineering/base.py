"""
Feature Extractor Base Class

ARY-1084: Feature Engineering Pipeline
Provides abstract base class for all feature extractors.

Created: 2026-02-17
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class FeatureMetadata:
    """Metadata for a computed feature."""
    name: str
    description: str
    category: str
    lookback_period: int = 0
    is_normalized: bool = False
    value_range: Optional[tuple] = None


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    All feature extractors must implement:
    - extract(): Compute features from input DataFrame
    - get_feature_names(): Return list of feature names
    - get_feature_metadata(): Return metadata for all features
    """
    
    def __init__(self, prefix: str = ""):
        """
        Initialize feature extractor.
        
        Args:
            prefix: Optional prefix for feature names
        """
        self.prefix = prefix
        self._feature_cache: Dict[str, pd.Series] = {}
    
    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from input DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
                Required columns: open, high, low, close, volume
                Index: DatetimeIndex
        
        Returns:
            DataFrame with computed features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names produced by this extractor.
        
        Returns:
            List of feature name strings
        """
        pass
    
    def get_feature_metadata(self) -> List[FeatureMetadata]:
        """
        Get metadata for all features.
        
        Returns:
            List of FeatureMetadata objects
        """
        return [
            FeatureMetadata(
                name=name,
                description=f"Feature: {name}",
                category=self.__class__.__name__
            )
            for name in self.get_feature_names()
        ]
    
    def _add_prefix(self, name: str) -> str:
        """Add prefix to feature name if configured."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame has required columns.
        
        Args:
            df: Input DataFrame to validate
        
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
    
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series, 
                     fill_value: float = 0.0) -> pd.Series:
        """
        Safely divide two series, handling division by zero.
        
        Args:
            numerator: Numerator series
            denominator: Denominator series
            fill_value: Value to use when denominator is zero
        
        Returns:
            Result of division with zeros handled
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = result.replace([np.inf, -np.inf], fill_value)
            result = result.fillna(fill_value)
        return result
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._feature_cache.clear()
