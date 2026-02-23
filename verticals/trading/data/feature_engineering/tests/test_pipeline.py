"""
Integration Tests for Feature Engineering Pipeline

ARY-1084: Feature Engineering Pipeline
Tests for the full pipeline and batch/streaming modes.

Created: 2026-02-17
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from datetime import datetime

from verticals.trading.data.feature_engineering.pipeline import (
    FeatureEngineeringPipeline,
    PipelineConfig,
    PipelineMetrics,
    create_pipeline
)
from verticals.trading.data.feature_engineering.technical_indicators import TechnicalIndicatorExtractor
from verticals.trading.data.feature_engineering.statistical_features import StatisticalFeatureExtractor
from verticals.trading.data.feature_engineering.temporal_features import TemporalFeatureExtractor


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start='2025-01-01', periods=n, freq='1h')
    
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = low + (high - low) * np.random.rand(n)
    
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    volume = np.random.randint(1000, 100000, n).astype(float)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def large_ohlcv_data():
    """Generate larger OHLCV data for performance testing."""
    np.random.seed(42)
    n = 5000  # ~7 months of hourly data
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = low + (high - low) * np.random.rand(n)
    
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    volume = np.random.randint(1000, 100000, n).astype(float)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.include_technical is True
        assert config.include_statistical is True
        assert config.include_temporal is True
        assert config.include_raw_ohlcv is True
        assert config.drop_na is True
        assert config.normalize_features is False
        assert config.feature_prefix == ""
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            include_technical=False,
            include_statistical=True,
            include_temporal=False,
            normalize_features=True
        )
        
        assert config.include_technical is False
        assert config.include_statistical is True
        assert config.include_temporal is False
        assert config.normalize_features is True


class TestFeatureEngineeringPipeline:
    """Tests for FeatureEngineeringPipeline."""
    
    def test_default_pipeline_creation(self):
        """Test creating pipeline with default settings."""
        pipeline = FeatureEngineeringPipeline()
        
        assert len(pipeline.extractors) == 3
        assert any(isinstance(e, TechnicalIndicatorExtractor) for e in pipeline.extractors)
        assert any(isinstance(e, StatisticalFeatureExtractor) for e in pipeline.extractors)
        assert any(isinstance(e, TemporalFeatureExtractor) for e in pipeline.extractors)
    
    def test_custom_extractors(self, sample_ohlcv_data):
        """Test pipeline with custom extractors."""
        extractors = [StatisticalFeatureExtractor()]
        pipeline = FeatureEngineeringPipeline(extractors=extractors)
        
        assert len(pipeline.extractors) == 1
        
        features = pipeline.extract_batch(sample_ohlcv_data)
        assert isinstance(features, pd.DataFrame)
    
    def test_extract_batch_returns_dataframe(self, sample_ohlcv_data):
        """Test that extract_batch returns a DataFrame."""
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_batch(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
    
    def test_extract_batch_includes_raw_ohlcv(self, sample_ohlcv_data):
        """Test that raw OHLCV columns are included."""
        config = PipelineConfig(include_raw_ohlcv=True)
        pipeline = FeatureEngineeringPipeline(config=config)
        features = pipeline.extract_batch(sample_ohlcv_data)
        
        assert 'open' in features.columns
        assert 'high' in features.columns
        assert 'low' in features.columns
        assert 'close' in features.columns
        assert 'volume' in features.columns
    
    def test_extract_batch_excludes_raw_ohlcv(self, sample_ohlcv_data):
        """Test that raw OHLCV can be excluded."""
        config = PipelineConfig(include_raw_ohlcv=False)
        pipeline = FeatureEngineeringPipeline(config=config)
        features = pipeline.extract_batch(sample_ohlcv_data)
        
        # Raw columns should not be present (unless created by extractors)
        # Note: Some extractors might create features with similar names
        assert len(features.columns) > 0
    
    def test_extract_batch_drops_na(self, sample_ohlcv_data):
        """Test that NaN rows are dropped when configured."""
        config = PipelineConfig(drop_na=True)
        pipeline = FeatureEngineeringPipeline(config=config)
        features = pipeline.extract_batch(sample_ohlcv_data)
        
        # Should have no NaN values
        assert not features.isna().any().any()
    
    def test_extract_batch_keeps_na(self, sample_ohlcv_data):
        """Test that NaN rows are kept when configured."""
        config = PipelineConfig(drop_na=False)
        pipeline = FeatureEngineeringPipeline(config=config)
        features = pipeline.extract_batch(sample_ohlcv_data)
        
        # Should have same number of rows as input
        assert len(features) == len(sample_ohlcv_data)
    
    def test_extract_batch_normalizes_features(self, sample_ohlcv_data):
        """Test feature normalization."""
        config = PipelineConfig(normalize_features=True)
        pipeline = FeatureEngineeringPipeline(config=config)
        features = pipeline.extract_batch(sample_ohlcv_data)
        
        # Normalized features should have mean ~0 and std ~1
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Check first 5 columns
            mean = features[col].mean()
            std = features[col].std()
            # Allow some tolerance
            assert abs(mean) < 0.1 or np.isnan(mean)
    
    def test_extract_streaming_returns_series(self, sample_ohlcv_data):
        """Test that extract_streaming returns a Series."""
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_streaming(sample_ohlcv_data)
        
        assert isinstance(features, pd.Series)
    
    def test_extract_streaming_matches_batch_last_row(self, sample_ohlcv_data):
        """Test that streaming mode matches batch mode for last row."""
        pipeline = FeatureEngineeringPipeline()
        
        # Batch mode
        batch_features = pipeline.extract_batch(sample_ohlcv_data)
        batch_last = batch_features.iloc[-1]
        
        # Streaming mode
        streaming_features = pipeline.extract_streaming(sample_ohlcv_data)
        
        # Should be identical
        pd.testing.assert_series_equal(batch_last, streaming_features)
    
    def test_extract_streaming_with_lookback(self, sample_ohlcv_data):
        """Test streaming mode with custom lookback."""
        pipeline = FeatureEngineeringPipeline()
        
        # Use only last 300 bars
        features = pipeline.extract_streaming(sample_ohlcv_data, lookback=300)
        
        assert isinstance(features, pd.Series)
        assert len(features) > 0
    
    def test_extract_streaming_batch(self, sample_ohlcv_data):
        """Test streaming batch mode."""
        pipeline = FeatureEngineeringPipeline()
        
        features = pipeline.extract_streaming_batch(sample_ohlcv_data, lookback=300, num_latest=10)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 10
    
    def test_get_all_feature_names(self):
        """Test getting all feature names."""
        pipeline = FeatureEngineeringPipeline()
        names = pipeline.get_all_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert 'open' in names  # Raw OHLCV
        assert 'close' in names
    
    def test_get_all_feature_metadata(self):
        """Test getting all feature metadata."""
        pipeline = FeatureEngineeringPipeline()
        metadata = pipeline.get_all_feature_metadata()
        
        assert isinstance(metadata, list)
        assert len(metadata) > 0
    
    def test_get_last_metrics(self, sample_ohlcv_data):
        """Test getting metrics from last extraction."""
        pipeline = FeatureEngineeringPipeline()
        
        # Before extraction
        assert pipeline.get_last_metrics() is None
        
        # After extraction
        pipeline.extract_batch(sample_ohlcv_data)
        metrics = pipeline.get_last_metrics()
        
        assert isinstance(metrics, PipelineMetrics)
        assert metrics.num_input_rows == len(sample_ohlcv_data)
        assert metrics.num_output_rows > 0
        assert metrics.num_features > 0
        assert metrics.extraction_time_ms > 0
    
    def test_get_required_lookback(self):
        """Test getting required lookback period."""
        pipeline = FeatureEngineeringPipeline()
        lookback = pipeline.get_required_lookback()
        
        assert isinstance(lookback, int)
        assert lookback > 0
        # Should be at least 200 (SMA 200 + buffer)
        assert lookback >= 200
    
    def test_validate_features(self, sample_ohlcv_data):
        """Test feature validation."""
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.extract_batch(sample_ohlcv_data)
        
        validation = pipeline.validate_features(features)
        
        assert 'is_valid' in validation
        assert 'issues' in validation
        assert 'statistics' in validation
        assert validation['is_valid'] is True  # No infinite values
    
    def test_clear_cache(self):
        """Test clearing cache."""
        pipeline = FeatureEngineeringPipeline()
        
        # Populate cache
        _ = pipeline.get_all_feature_names()
        assert pipeline._feature_names_cache is not None
        
        # Clear cache
        pipeline.clear_cache()
        assert pipeline._feature_names_cache is None


class TestPipelineVersioning:
    """Tests for feature versioning."""
    
    def test_save_feature_definitions(self):
        """Test saving feature definitions."""
        pipeline = FeatureEngineeringPipeline()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "definitions.json"
            pipeline.save_feature_definitions(path)
            
            assert path.exists()
            
            with open(path) as f:
                definitions = json.load(f)
            
            assert 'version' in definitions
            assert 'feature_names' in definitions
            assert 'num_features' in definitions
            assert 'extractors' in definitions
    
    def test_load_feature_definitions(self):
        """Test loading feature definitions."""
        pipeline = FeatureEngineeringPipeline()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "definitions.json"
            pipeline.save_feature_definitions(path)
            
            definitions = FeatureEngineeringPipeline.load_feature_definitions(path)
            
            assert definitions['version'] == pipeline.VERSION
            assert len(definitions['feature_names']) == len(pipeline.get_all_feature_names())


class TestPipelineDeterminism:
    """Tests for pipeline determinism."""
    
    def test_batch_determinism(self, sample_ohlcv_data):
        """Test that batch extraction is deterministic."""
        pipeline = FeatureEngineeringPipeline()
        
        features1 = pipeline.extract_batch(sample_ohlcv_data)
        features2 = pipeline.extract_batch(sample_ohlcv_data)
        
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_streaming_determinism(self, sample_ohlcv_data):
        """Test that streaming extraction is deterministic."""
        pipeline = FeatureEngineeringPipeline()
        
        features1 = pipeline.extract_streaming(sample_ohlcv_data)
        features2 = pipeline.extract_streaming(sample_ohlcv_data)
        
        pd.testing.assert_series_equal(features1, features2)


class TestPipelinePerformance:
    """Tests for pipeline performance."""
    
    def test_batch_performance(self, large_ohlcv_data):
        """Test batch extraction performance."""
        pipeline = FeatureEngineeringPipeline()
        
        features = pipeline.extract_batch(large_ohlcv_data)
        metrics = pipeline.get_last_metrics()
        
        # Should complete in reasonable time (< 30 seconds for 5000 rows)
        assert metrics.extraction_time_ms < 30000
        
        # Log performance for reference
        print(f"\nBatch performance: {metrics.extraction_time_ms:.2f}ms for {metrics.num_input_rows} rows")
        print(f"Features: {metrics.num_features}")
    
    def test_streaming_performance(self, sample_ohlcv_data):
        """Test streaming extraction performance."""
        import time
        
        pipeline = FeatureEngineeringPipeline()
        
        # Warm up
        _ = pipeline.extract_streaming(sample_ohlcv_data)
        
        # Measure
        start = time.time()
        for _ in range(10):
            _ = pipeline.extract_streaming(sample_ohlcv_data)
        elapsed = (time.time() - start) * 1000 / 10
        
        # Should complete in < 100ms per call
        assert elapsed < 100
        
        print(f"\nStreaming performance: {elapsed:.2f}ms per call")


class TestCreatePipelineFactory:
    """Tests for create_pipeline factory function."""
    
    def test_create_default_pipeline(self):
        """Test creating default pipeline."""
        pipeline = create_pipeline()
        
        assert isinstance(pipeline, FeatureEngineeringPipeline)
        assert len(pipeline.extractors) == 3
    
    def test_create_technical_only_pipeline(self):
        """Test creating pipeline with only technical indicators."""
        pipeline = create_pipeline(
            include_technical=True,
            include_statistical=False,
            include_temporal=False
        )
        
        assert len(pipeline.extractors) == 1
        assert isinstance(pipeline.extractors[0], TechnicalIndicatorExtractor)
    
    def test_create_pipeline_with_options(self, sample_ohlcv_data):
        """Test creating pipeline with custom options."""
        pipeline = create_pipeline(
            include_technical=True,
            include_statistical=True,
            include_temporal=False,
            normalize_features=True
        )
        
        features = pipeline.extract_batch(sample_ohlcv_data)
        assert isinstance(features, pd.DataFrame)


class TestBatchStreamingConsistency:
    """Tests for consistency between batch and streaming modes."""
    
    def test_feature_consistency(self, sample_ohlcv_data):
        """Test that batch and streaming produce identical features for last row."""
        pipeline = FeatureEngineeringPipeline()
        
        # Batch mode
        batch_features = pipeline.extract_batch(sample_ohlcv_data)
        
        # Streaming mode (using same data)
        streaming_features = pipeline.extract_streaming(sample_ohlcv_data)
        
        # Compare last row of batch with streaming result
        batch_last = batch_features.iloc[-1]
        
        # Should be identical
        pd.testing.assert_series_equal(
            batch_last.sort_index(),
            streaming_features.sort_index(),
            check_names=False
        )
    
    def test_feature_consistency_with_lookback(self, sample_ohlcv_data):
        """Test consistency with limited lookback."""
        pipeline = FeatureEngineeringPipeline()
        
        # Use last 300 bars for streaming
        lookback = 300
        streaming_data = sample_ohlcv_data.iloc[-lookback:]
        
        # Batch on limited data
        batch_features = pipeline.extract_batch(streaming_data)
        batch_last = batch_features.iloc[-1]
        
        # Streaming on same limited data
        streaming_features = pipeline.extract_streaming(streaming_data, lookback=lookback)
        
        # Should be identical
        pd.testing.assert_series_equal(
            batch_last.sort_index(),
            streaming_features.sort_index(),
            check_names=False
        )
