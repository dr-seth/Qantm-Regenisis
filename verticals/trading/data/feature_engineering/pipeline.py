"""
Feature Engineering Pipeline

ARY-1084: Feature Engineering Pipeline
Orchestrates feature extraction from multiple extractors.

Created: 2026-02-17
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from .base import FeatureExtractor, FeatureMetadata
from .technical_indicators import TechnicalIndicatorExtractor
from .statistical_features import StatisticalFeatureExtractor
from .temporal_features import TemporalFeatureExtractor


@dataclass
class PipelineConfig:
    """Configuration for feature engineering pipeline."""
    include_technical: bool = True
    include_statistical: bool = True
    include_temporal: bool = True
    include_raw_ohlcv: bool = True
    drop_na: bool = True
    normalize_features: bool = False
    feature_prefix: str = ""


@dataclass
class PipelineMetrics:
    """Metrics from pipeline execution."""
    num_input_rows: int
    num_output_rows: int
    num_features: int
    extraction_time_ms: float
    rows_dropped: int
    features_by_category: Dict[str, int]


class FeatureEngineeringPipeline:
    """
    Orchestrates feature extraction from multiple extractors.
    
    Supports:
    - Batch mode: Process historical data efficiently
    - Streaming mode: Process real-time data with minimal latency
    - Feature versioning: Save/load feature definitions
    - Performance optimization: Vectorized operations
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        extractors: Optional[List[FeatureExtractor]] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize feature engineering pipeline.
        
        Args:
            extractors: List of feature extractors (default: all extractors)
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        if extractors is not None:
            self.extractors = extractors
        else:
            self.extractors = self._create_default_extractors()
        
        self._feature_names_cache: Optional[List[str]] = None
        self._last_metrics: Optional[PipelineMetrics] = None
    
    def _create_default_extractors(self) -> List[FeatureExtractor]:
        """Create default set of extractors based on config."""
        extractors = []
        prefix = self.config.feature_prefix
        
        if self.config.include_technical:
            extractors.append(TechnicalIndicatorExtractor(prefix=prefix))
        
        if self.config.include_statistical:
            extractors.append(StatisticalFeatureExtractor(prefix=prefix))
        
        if self.config.include_temporal:
            extractors.append(TemporalFeatureExtractor(prefix=prefix))
        
        return extractors
    
    def extract_batch(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Extract features in batch mode (historical data).
        
        Args:
            df: Input DataFrame with OHLCV data
                Required columns: open, high, low, close, volume
                Index: DatetimeIndex
            progress_callback: Optional callback for progress updates
        
        Returns:
            DataFrame with all computed features
        """
        start_time = time.time()
        num_input_rows = len(df)
        
        # Start with raw OHLCV if configured
        if self.config.include_raw_ohlcv:
            all_features = [df[['open', 'high', 'low', 'close', 'volume']].copy()]
        else:
            all_features = []
        
        # Extract features from each extractor
        features_by_category = {}
        for i, extractor in enumerate(self.extractors):
            extractor_name = type(extractor).__name__
            
            if progress_callback:
                progress_callback(f"Extracting {extractor_name}...", i / len(self.extractors))
            
            features = extractor.extract(df)
            all_features.append(features)
            features_by_category[extractor_name] = len(features.columns)
        
        # Combine all features
        result = pd.concat(all_features, axis=1)
        
        # Drop rows with NaN (from rolling windows)
        rows_before = len(result)
        if self.config.drop_na:
            result = result.dropna()
        rows_dropped = rows_before - len(result)
        
        # Normalize features if configured
        if self.config.normalize_features:
            result = self._normalize_features(result)
        
        # Calculate metrics
        extraction_time = (time.time() - start_time) * 1000
        self._last_metrics = PipelineMetrics(
            num_input_rows=num_input_rows,
            num_output_rows=len(result),
            num_features=len(result.columns),
            extraction_time_ms=extraction_time,
            rows_dropped=rows_dropped,
            features_by_category=features_by_category
        )
        
        if progress_callback:
            progress_callback("Complete", 1.0)
        
        # Clear feature names cache
        self._feature_names_cache = None
        
        return result
    
    def extract_streaming(
        self,
        df: pd.DataFrame,
        lookback: int = 200
    ) -> pd.Series:
        """
        Extract features in streaming mode (real-time data).
        
        Optimized for low latency by only computing features for the latest bar.
        
        Args:
            df: Recent historical data (e.g., last 200 bars) + latest bar
            lookback: Number of historical bars to use for computation
        
        Returns:
            Feature vector for the latest bar
        """
        # Use only the required lookback period
        if len(df) > lookback:
            df = df.iloc[-lookback:]
        
        # Extract features for all bars (needed for rolling calculations)
        features_df = self.extract_batch(df)
        
        # Return only the latest row
        if len(features_df) == 0:
            raise ValueError("No valid features computed. Increase lookback period.")
        
        return features_df.iloc[-1]
    
    def extract_streaming_batch(
        self,
        df: pd.DataFrame,
        lookback: int = 200,
        num_latest: int = 10
    ) -> pd.DataFrame:
        """
        Extract features for multiple latest bars in streaming mode.
        
        Args:
            df: Recent historical data + latest bars
            lookback: Number of historical bars for computation
            num_latest: Number of latest bars to return features for
        
        Returns:
            DataFrame with features for the latest num_latest bars
        """
        # Use only the required lookback period
        if len(df) > lookback + num_latest:
            df = df.iloc[-(lookback + num_latest):]
        
        # Extract features
        features_df = self.extract_batch(df)
        
        # Return only the latest rows
        return features_df.iloc[-num_latest:]
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Normalized DataFrame
        """
        result = df.copy()
        
        for col in result.columns:
            if result[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std
        
        return result
    
    def get_all_feature_names(self) -> List[str]:
        """
        Get all feature names from all extractors.
        
        Returns:
            List of all feature names
        """
        if self._feature_names_cache is not None:
            return self._feature_names_cache
        
        names = []
        
        # Raw OHLCV features
        if self.config.include_raw_ohlcv:
            names.extend(['open', 'high', 'low', 'close', 'volume'])
        
        # Features from extractors
        for extractor in self.extractors:
            names.extend(extractor.get_feature_names())
        
        self._feature_names_cache = names
        return names
    
    def get_all_feature_metadata(self) -> List[FeatureMetadata]:
        """
        Get metadata for all features.
        
        Returns:
            List of FeatureMetadata objects
        """
        metadata = []
        
        # Raw OHLCV metadata
        if self.config.include_raw_ohlcv:
            metadata.extend([
                FeatureMetadata('open', 'Opening price', 'raw', 0),
                FeatureMetadata('high', 'High price', 'raw', 0),
                FeatureMetadata('low', 'Low price', 'raw', 0),
                FeatureMetadata('close', 'Closing price', 'raw', 0),
                FeatureMetadata('volume', 'Trading volume', 'raw', 0),
            ])
        
        # Metadata from extractors
        for extractor in self.extractors:
            metadata.extend(extractor.get_feature_metadata())
        
        return metadata
    
    def get_last_metrics(self) -> Optional[PipelineMetrics]:
        """Get metrics from the last extraction."""
        return self._last_metrics
    
    def save_feature_definitions(self, path: Union[str, Path]) -> None:
        """
        Save feature definitions for versioning.
        
        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = self.get_all_feature_metadata()
        
        definitions = {
            'version': self.VERSION,
            'created_at': datetime.utcnow().isoformat(),
            'config': asdict(self.config),
            'extractors': [type(e).__name__ for e in self.extractors],
            'num_features': len(self.get_all_feature_names()),
            'feature_names': self.get_all_feature_names(),
            'feature_metadata': [
                {
                    'name': m.name,
                    'description': m.description,
                    'category': m.category,
                    'lookback_period': m.lookback_period,
                    'is_normalized': m.is_normalized,
                    'value_range': m.value_range
                }
                for m in metadata
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(definitions, f, indent=2)
    
    @classmethod
    def load_feature_definitions(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load feature definitions from file.
        
        Args:
            path: Path to JSON file
        
        Returns:
            Dictionary with feature definitions
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_required_lookback(self) -> int:
        """
        Get the maximum lookback period required by all features.
        
        Returns:
            Maximum lookback period in bars
        """
        max_lookback = 0
        
        for metadata in self.get_all_feature_metadata():
            if metadata.lookback_period > max_lookback:
                max_lookback = metadata.lookback_period
        
        # Add buffer for safety
        return max_lookback + 10
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate extracted features for quality issues.
        
        Args:
            df: DataFrame with extracted features
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            results['issues'].append({
                'type': 'nan_values',
                'columns': nan_cols.to_dict()
            })
        
        # Check for infinite values
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        inf_cols = inf_counts[inf_counts > 0]
        if len(inf_cols) > 0:
            results['is_valid'] = False
            results['issues'].append({
                'type': 'infinite_values',
                'columns': inf_cols.to_dict()
            })
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if len(constant_cols) > 0:
            results['issues'].append({
                'type': 'constant_columns',
                'columns': constant_cols
            })
        
        # Basic statistics
        results['statistics'] = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._feature_names_cache = None
        for extractor in self.extractors:
            extractor.clear_cache()


def create_pipeline(
    include_technical: bool = True,
    include_statistical: bool = True,
    include_temporal: bool = True,
    **kwargs
) -> FeatureEngineeringPipeline:
    """
    Factory function to create a feature engineering pipeline.
    
    Args:
        include_technical: Include technical indicators
        include_statistical: Include statistical features
        include_temporal: Include temporal features
        **kwargs: Additional config options
    
    Returns:
        Configured FeatureEngineeringPipeline
    """
    config = PipelineConfig(
        include_technical=include_technical,
        include_statistical=include_statistical,
        include_temporal=include_temporal,
        **kwargs
    )
    return FeatureEngineeringPipeline(config=config)
