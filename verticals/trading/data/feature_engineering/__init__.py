"""
Feature Engineering Pipeline — ARY-1084

Compute features for trading models from OHLCV market data.
Supports batch (historical) and streaming (real-time) modes.

Modules:
- base.py: FeatureExtractor abstract base class + FeatureMetadata
- technical_indicators.py: 60+ technical analysis indicators (TA-Lib)
- statistical_features.py: 40+ statistical features (returns, volatility, etc.)
- temporal_features.py: 38 temporal features (hour, day, session indicators)
- pipeline.py: FeatureEngineeringPipeline orchestrator + versioning
"""

from .base import FeatureExtractor, FeatureMetadata
from .technical_indicators import TechnicalIndicatorExtractor
from .statistical_features import StatisticalFeatureExtractor
from .temporal_features import TemporalFeatureExtractor
from .pipeline import (
    FeatureEngineeringPipeline,
    PipelineConfig,
    PipelineMetrics,
    create_pipeline,
)

__all__ = [
    "FeatureExtractor",
    "FeatureMetadata",
    "FeatureEngineeringPipeline",
    "PipelineConfig",
    "PipelineMetrics",
    "create_pipeline",
    "TechnicalIndicatorExtractor",
    "StatisticalFeatureExtractor",
    "TemporalFeatureExtractor",
]
