"""
Unit Tests for Feature Extractors

ARY-1084: Feature Engineering Pipeline
Tests for individual feature extractors.

Created: 2026-02-17
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from verticals.trading.data.feature_engineering.base import FeatureExtractor, FeatureMetadata
from verticals.trading.data.feature_engineering.technical_indicators import TechnicalIndicatorExtractor
from verticals.trading.data.feature_engineering.statistical_features import StatisticalFeatureExtractor
from verticals.trading.data.feature_engineering.temporal_features import TemporalFeatureExtractor


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    
    # Generate realistic price data
    dates = pd.date_range(start='2025-01-01', periods=n, freq='1h')
    
    # Random walk for close prices
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = low + (high - low) * np.random.rand(n)
    
    # Ensure OHLC relationships are valid
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    # Generate volume
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
def small_ohlcv_data():
    """Generate small OHLCV data for edge case testing."""
    dates = pd.date_range(start='2025-01-01', periods=50, freq='1h')
    
    df = pd.DataFrame({
        'open': np.linspace(100, 110, 50),
        'high': np.linspace(101, 111, 50),
        'low': np.linspace(99, 109, 50),
        'close': np.linspace(100.5, 110.5, 50),
        'volume': np.ones(50) * 10000
    }, index=dates)
    
    return df


class TestFeatureExtractorBase:
    """Tests for FeatureExtractor base class."""
    
    def test_validate_input_missing_columns(self, sample_ohlcv_data):
        """Test that validation fails with missing columns."""
        # Create a concrete implementation for testing
        class TestExtractor(FeatureExtractor):
            def extract(self, df):
                self._validate_input(df)
                return pd.DataFrame(index=df.index)
            
            def get_feature_names(self):
                return []
        
        extractor = TestExtractor()
        
        # Remove a required column
        df = sample_ohlcv_data.drop(columns=['volume'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            extractor.extract(df)
    
    def test_validate_input_wrong_index(self, sample_ohlcv_data):
        """Test that validation fails with non-datetime index."""
        class TestExtractor(FeatureExtractor):
            def extract(self, df):
                self._validate_input(df)
                return pd.DataFrame(index=df.index)
            
            def get_feature_names(self):
                return []
        
        extractor = TestExtractor()
        
        # Reset index to integer
        df = sample_ohlcv_data.reset_index(drop=True)
        
        with pytest.raises(ValueError, match="DatetimeIndex"):
            extractor.extract(df)
    
    def test_prefix_functionality(self):
        """Test that prefix is correctly applied to feature names."""
        class TestExtractor(FeatureExtractor):
            def extract(self, df):
                return pd.DataFrame(index=df.index)
            
            def get_feature_names(self):
                return [self._add_prefix('feature1'), self._add_prefix('feature2')]
        
        # Without prefix
        extractor1 = TestExtractor()
        assert extractor1.get_feature_names() == ['feature1', 'feature2']
        
        # With prefix
        extractor2 = TestExtractor(prefix='test')
        assert extractor2.get_feature_names() == ['test_feature1', 'test_feature2']
    
    def test_safe_divide(self, sample_ohlcv_data):
        """Test safe division handles edge cases."""
        class TestExtractor(FeatureExtractor):
            def extract(self, df):
                return pd.DataFrame(index=df.index)
            
            def get_feature_names(self):
                return []
        
        extractor = TestExtractor()
        
        numerator = pd.Series([1, 2, 3, 4, 5])
        denominator = pd.Series([1, 0, 2, 0, 1])
        
        result = extractor._safe_divide(numerator, denominator, fill_value=0.0)
        
        assert result[0] == 1.0
        assert result[1] == 0.0  # Division by zero handled
        assert result[2] == 1.5
        assert result[3] == 0.0  # Division by zero handled
        assert result[4] == 5.0


class TestTechnicalIndicatorExtractor:
    """Tests for TechnicalIndicatorExtractor."""
    
    def test_extract_returns_dataframe(self, sample_ohlcv_data):
        """Test that extract returns a DataFrame."""
        extractor = TechnicalIndicatorExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
    
    def test_extract_has_expected_features(self, sample_ohlcv_data):
        """Test that extract produces expected number of features."""
        extractor = TechnicalIndicatorExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        # Should have 20+ technical indicators
        assert len(features.columns) >= 20
    
    def test_feature_names_match_columns(self, sample_ohlcv_data):
        """Test that get_feature_names matches extracted columns."""
        extractor = TechnicalIndicatorExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        expected_names = set(extractor.get_feature_names())
        actual_names = set(features.columns)
        
        assert expected_names == actual_names
    
    def test_rsi_bounds(self, sample_ohlcv_data):
        """Test that RSI is bounded between 0 and 100."""
        extractor = TechnicalIndicatorExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        rsi = features['rsi_14'].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()
    
    def test_bollinger_band_relationships(self, sample_ohlcv_data):
        """Test Bollinger Band relationships."""
        extractor = TechnicalIndicatorExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        # Upper > Middle > Lower
        valid_rows = features[['bb_upper', 'bb_middle', 'bb_lower']].dropna()
        assert (valid_rows['bb_upper'] >= valid_rows['bb_middle']).all()
        assert (valid_rows['bb_middle'] >= valid_rows['bb_lower']).all()
    
    def test_feature_metadata(self):
        """Test that feature metadata is properly defined."""
        extractor = TechnicalIndicatorExtractor()
        metadata = extractor.get_feature_metadata()
        
        assert len(metadata) > 0
        assert all(isinstance(m, FeatureMetadata) for m in metadata)
        assert all(m.name in extractor.get_feature_names() for m in metadata)
    
    def test_prefix_applied(self, sample_ohlcv_data):
        """Test that prefix is applied to all features."""
        extractor = TechnicalIndicatorExtractor(prefix='tech')
        features = extractor.extract(sample_ohlcv_data)
        
        assert all(col.startswith('tech_') for col in features.columns)


class TestStatisticalFeatureExtractor:
    """Tests for StatisticalFeatureExtractor."""
    
    def test_extract_returns_dataframe(self, sample_ohlcv_data):
        """Test that extract returns a DataFrame."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
    
    def test_extract_has_expected_features(self, sample_ohlcv_data):
        """Test that extract produces expected number of features."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        # Should have 15+ statistical features
        assert len(features.columns) >= 15
    
    def test_feature_names_match_columns(self, sample_ohlcv_data):
        """Test that get_feature_names matches extracted columns."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        expected_names = set(extractor.get_feature_names())
        actual_names = set(features.columns)
        
        assert expected_names == actual_names
    
    def test_returns_calculation(self, small_ohlcv_data):
        """Test that returns are calculated correctly."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(small_ohlcv_data)
        
        # Manual calculation of 1-period return
        expected_return = small_ohlcv_data['close'].pct_change(1)
        actual_return = features['return_1']
        
        pd.testing.assert_series_equal(expected_return, actual_return, check_names=False)
    
    def test_volatility_positive(self, sample_ohlcv_data):
        """Test that volatility is always positive."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        vol = features['volatility_20'].dropna()
        assert (vol >= 0).all()
    
    def test_autocorrelation_bounds(self, sample_ohlcv_data):
        """Test that autocorrelation is bounded between -1 and 1."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        autocorr = features['autocorr_1'].dropna()
        assert (autocorr >= -1).all()
        assert (autocorr <= 1).all()
    
    def test_close_position_bounds(self, sample_ohlcv_data):
        """Test that close position is bounded between 0 and 1."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        pos = features['close_position'].dropna()
        assert (pos >= 0).all()
        assert (pos <= 1).all()


class TestTemporalFeatureExtractor:
    """Tests for TemporalFeatureExtractor."""
    
    def test_extract_returns_dataframe(self, sample_ohlcv_data):
        """Test that extract returns a DataFrame."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
    
    def test_hour_bounds(self, sample_ohlcv_data):
        """Test that hour is bounded between 0 and 23."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        hour = features['hour']
        assert (hour >= 0).all()
        assert (hour <= 23).all()
    
    def test_day_of_week_bounds(self, sample_ohlcv_data):
        """Test that day of week is bounded between 0 and 6."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        dow = features['day_of_week']
        assert (dow >= 0).all()
        assert (dow <= 6).all()
    
    def test_cyclical_encoding_bounds(self, sample_ohlcv_data):
        """Test that cyclical encodings are bounded between -1 and 1."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        for col in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']:
            assert (features[col] >= -1).all()
            assert (features[col] <= 1).all()
    
    def test_binary_indicators(self, sample_ohlcv_data):
        """Test that binary indicators are 0 or 1."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_ohlcv_data)
        
        binary_cols = ['is_weekend', 'is_monday', 'is_friday', 'is_month_start', 'is_month_end']
        for col in binary_cols:
            assert features[col].isin([0, 1]).all()
    
    def test_weekend_detection(self):
        """Test that weekend is correctly detected."""
        # Create data spanning a weekend
        dates = pd.date_range(start='2025-01-03', periods=5, freq='D')  # Fri, Sat, Sun, Mon, Tue
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5
        }, index=dates)
        
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(df)
        
        # Friday (index 0) should not be weekend
        assert features['is_weekend'].iloc[0] == 0
        # Saturday (index 1) should be weekend
        assert features['is_weekend'].iloc[1] == 1
        # Sunday (index 2) should be weekend
        assert features['is_weekend'].iloc[2] == 1
        # Monday (index 3) should not be weekend
        assert features['is_weekend'].iloc[3] == 0


class TestFeatureDeterminism:
    """Tests for feature determinism (same input -> same output)."""
    
    def test_technical_determinism(self, sample_ohlcv_data):
        """Test that technical features are deterministic."""
        extractor = TechnicalIndicatorExtractor()
        
        features1 = extractor.extract(sample_ohlcv_data)
        features2 = extractor.extract(sample_ohlcv_data)
        
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_statistical_determinism(self, sample_ohlcv_data):
        """Test that statistical features are deterministic."""
        extractor = StatisticalFeatureExtractor()
        
        features1 = extractor.extract(sample_ohlcv_data)
        features2 = extractor.extract(sample_ohlcv_data)
        
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_temporal_determinism(self, sample_ohlcv_data):
        """Test that temporal features are deterministic."""
        extractor = TemporalFeatureExtractor()
        
        features1 = extractor.extract(sample_ohlcv_data)
        features2 = extractor.extract(sample_ohlcv_data)
        
        pd.testing.assert_frame_equal(features1, features2)
