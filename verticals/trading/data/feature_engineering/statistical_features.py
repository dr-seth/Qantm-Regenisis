"""
Statistical Feature Extractor

ARY-1084: Feature Engineering Pipeline
Extracts 15+ statistical features from price data.

Created: 2026-02-17
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

from .base import FeatureExtractor, FeatureMetadata


class StatisticalFeatureExtractor(FeatureExtractor):
    """
    Extracts statistical features from OHLCV data.
    
    Implements 15+ features across categories:
    - Returns: Simple returns, log returns, multi-period returns
    - Volatility: Rolling std, Parkinson, Garman-Klass
    - Distribution: Skewness, kurtosis
    - Autocorrelation: Lag-based correlations
    - Volume: Relative volume, volume volatility
    - Price ratios: High-low, close-open
    """
    
    def __init__(self, prefix: str = ""):
        """
        Initialize statistical feature extractor.
        
        Args:
            prefix: Optional prefix for feature names
        """
        super().__init__(prefix)
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical features from OHLCV data.
        
        Args:
            df: DataFrame with open, high, low, close, volume columns
        
        Returns:
            DataFrame with statistical features
        """
        self._validate_input(df)
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']
        
        # === RETURN FEATURES ===
        
        # Simple returns (percentage change)
        features[self._add_prefix('return_1')] = close.pct_change(1)
        features[self._add_prefix('return_5')] = close.pct_change(5)
        features[self._add_prefix('return_10')] = close.pct_change(10)
        features[self._add_prefix('return_20')] = close.pct_change(20)
        
        # Log returns
        features[self._add_prefix('log_return')] = np.log(close / close.shift(1))
        features[self._add_prefix('log_return_5')] = np.log(close / close.shift(5))
        features[self._add_prefix('log_return_10')] = np.log(close / close.shift(10))
        
        # Cumulative returns
        features[self._add_prefix('cum_return_5')] = close / close.shift(5) - 1
        features[self._add_prefix('cum_return_20')] = close / close.shift(20) - 1
        
        # === VOLATILITY FEATURES ===
        
        # Rolling standard deviation of returns
        log_ret = np.log(close / close.shift(1))
        features[self._add_prefix('volatility_10')] = log_ret.rolling(10).std() * np.sqrt(252)
        features[self._add_prefix('volatility_20')] = log_ret.rolling(20).std() * np.sqrt(252)
        features[self._add_prefix('volatility_50')] = log_ret.rolling(50).std() * np.sqrt(252)
        
        # Parkinson volatility (uses high-low range)
        hl_ratio = np.log(high / low)
        features[self._add_prefix('parkinson_vol_10')] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(10).mean()
        ) * np.sqrt(252)
        features[self._add_prefix('parkinson_vol_20')] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(20).mean()
        ) * np.sqrt(252)
        
        # Garman-Klass volatility (uses OHLC)
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_price) ** 2
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        features[self._add_prefix('garman_klass_vol_10')] = np.sqrt(gk_var.rolling(10).mean()) * np.sqrt(252)
        features[self._add_prefix('garman_klass_vol_20')] = np.sqrt(gk_var.rolling(20).mean()) * np.sqrt(252)
        
        # Volatility ratio (short-term / long-term)
        features[self._add_prefix('vol_ratio_10_50')] = (
            features[self._add_prefix('volatility_10')] / 
            features[self._add_prefix('volatility_50')]
        )
        
        # === DISTRIBUTION FEATURES ===
        
        # Skewness (asymmetry of returns)
        features[self._add_prefix('skewness_20')] = log_ret.rolling(20).skew()
        features[self._add_prefix('skewness_50')] = log_ret.rolling(50).skew()
        
        # Kurtosis (tail heaviness of returns)
        features[self._add_prefix('kurtosis_20')] = log_ret.rolling(20).kurt()
        features[self._add_prefix('kurtosis_50')] = log_ret.rolling(50).kurt()
        
        # === AUTOCORRELATION FEATURES ===
        
        # Autocorrelation at different lags
        features[self._add_prefix('autocorr_1')] = log_ret.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        features[self._add_prefix('autocorr_5')] = log_ret.rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )
        features[self._add_prefix('autocorr_10')] = log_ret.rolling(30).apply(
            lambda x: x.autocorr(lag=10) if len(x) > 10 else 0, raw=False
        )
        
        # === VOLUME FEATURES ===
        
        # Volume ratio (current / average)
        features[self._add_prefix('volume_ratio_10')] = volume / volume.rolling(10).mean()
        features[self._add_prefix('volume_ratio_20')] = volume / volume.rolling(20).mean()
        
        # Volume standard deviation
        features[self._add_prefix('volume_std_10')] = volume.rolling(10).std() / volume.rolling(10).mean()
        features[self._add_prefix('volume_std_20')] = volume.rolling(20).std() / volume.rolling(20).mean()
        
        # Volume trend
        features[self._add_prefix('volume_trend')] = (
            volume.rolling(5).mean() / volume.rolling(20).mean()
        )
        
        # Price-volume correlation
        features[self._add_prefix('price_volume_corr')] = close.rolling(20).corr(volume)
        
        # === PRICE RATIO FEATURES ===
        
        # High-low ratio (intrabar volatility proxy)
        features[self._add_prefix('high_low_ratio')] = high / low
        features[self._add_prefix('high_low_pct')] = (high - low) / close
        
        # Close-open ratio (intrabar direction)
        features[self._add_prefix('close_open_ratio')] = close / open_price
        features[self._add_prefix('close_open_pct')] = (close - open_price) / open_price
        
        # Close position within bar
        features[self._add_prefix('close_position')] = (close - low) / (high - low + 1e-10)
        
        # Gap (open vs previous close)
        features[self._add_prefix('gap')] = open_price / close.shift(1) - 1
        
        # === TREND FEATURES ===
        
        # Linear regression slope
        features[self._add_prefix('trend_slope_10')] = self._rolling_slope(close, 10)
        features[self._add_prefix('trend_slope_20')] = self._rolling_slope(close, 20)
        
        # R-squared of linear fit
        features[self._add_prefix('trend_r2_10')] = self._rolling_r2(close, 10)
        features[self._add_prefix('trend_r2_20')] = self._rolling_r2(close, 20)
        
        # === Z-SCORE FEATURES ===
        
        # Price z-score (how many std from mean)
        features[self._add_prefix('zscore_20')] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        features[self._add_prefix('zscore_50')] = (close - close.rolling(50).mean()) / close.rolling(50).std()
        
        # Volume z-score
        features[self._add_prefix('volume_zscore_20')] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        
        return features
    
    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear regression slope."""
        def calc_slope(y):
            if len(y) < window:
                return np.nan
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        return series.rolling(window).apply(calc_slope, raw=True)
    
    def _rolling_r2(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling R-squared of linear fit."""
        def calc_r2(y):
            if len(y) < window:
                return np.nan
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot == 0:
                return 0
            return 1 - (ss_res / ss_tot)
        
        return series.rolling(window).apply(calc_r2, raw=True)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all statistical feature names."""
        names = [
            # Returns
            'return_1', 'return_5', 'return_10', 'return_20',
            'log_return', 'log_return_5', 'log_return_10',
            'cum_return_5', 'cum_return_20',
            # Volatility
            'volatility_10', 'volatility_20', 'volatility_50',
            'parkinson_vol_10', 'parkinson_vol_20',
            'garman_klass_vol_10', 'garman_klass_vol_20',
            'vol_ratio_10_50',
            # Distribution
            'skewness_20', 'skewness_50',
            'kurtosis_20', 'kurtosis_50',
            # Autocorrelation
            'autocorr_1', 'autocorr_5', 'autocorr_10',
            # Volume
            'volume_ratio_10', 'volume_ratio_20',
            'volume_std_10', 'volume_std_20',
            'volume_trend', 'price_volume_corr',
            # Price ratios
            'high_low_ratio', 'high_low_pct',
            'close_open_ratio', 'close_open_pct',
            'close_position', 'gap',
            # Trend
            'trend_slope_10', 'trend_slope_20',
            'trend_r2_10', 'trend_r2_20',
            # Z-scores
            'zscore_20', 'zscore_50', 'volume_zscore_20'
        ]
        return [self._add_prefix(name) for name in names]
    
    def get_feature_metadata(self) -> List[FeatureMetadata]:
        """Get metadata for all statistical features."""
        metadata = [
            # Returns
            FeatureMetadata('return_1', 'Simple return (1 period)', 'returns', 1),
            FeatureMetadata('return_5', 'Simple return (5 periods)', 'returns', 5),
            FeatureMetadata('return_10', 'Simple return (10 periods)', 'returns', 10),
            FeatureMetadata('return_20', 'Simple return (20 periods)', 'returns', 20),
            FeatureMetadata('log_return', 'Log return (1 period)', 'returns', 1),
            FeatureMetadata('log_return_5', 'Log return (5 periods)', 'returns', 5),
            FeatureMetadata('log_return_10', 'Log return (10 periods)', 'returns', 10),
            FeatureMetadata('cum_return_5', 'Cumulative return (5 periods)', 'returns', 5),
            FeatureMetadata('cum_return_20', 'Cumulative return (20 periods)', 'returns', 20),
            # Volatility
            FeatureMetadata('volatility_10', 'Annualized volatility (10 periods)', 'volatility', 10),
            FeatureMetadata('volatility_20', 'Annualized volatility (20 periods)', 'volatility', 20),
            FeatureMetadata('volatility_50', 'Annualized volatility (50 periods)', 'volatility', 50),
            FeatureMetadata('parkinson_vol_10', 'Parkinson volatility (10 periods)', 'volatility', 10),
            FeatureMetadata('parkinson_vol_20', 'Parkinson volatility (20 periods)', 'volatility', 20),
            FeatureMetadata('garman_klass_vol_10', 'Garman-Klass volatility (10 periods)', 'volatility', 10),
            FeatureMetadata('garman_klass_vol_20', 'Garman-Klass volatility (20 periods)', 'volatility', 20),
            FeatureMetadata('vol_ratio_10_50', 'Volatility ratio (10/50)', 'volatility', 50),
            # Distribution
            FeatureMetadata('skewness_20', 'Return skewness (20 periods)', 'distribution', 20),
            FeatureMetadata('skewness_50', 'Return skewness (50 periods)', 'distribution', 50),
            FeatureMetadata('kurtosis_20', 'Return kurtosis (20 periods)', 'distribution', 20),
            FeatureMetadata('kurtosis_50', 'Return kurtosis (50 periods)', 'distribution', 50),
            # Autocorrelation
            FeatureMetadata('autocorr_1', 'Autocorrelation lag 1', 'autocorrelation', 20, value_range=(-1, 1)),
            FeatureMetadata('autocorr_5', 'Autocorrelation lag 5', 'autocorrelation', 20, value_range=(-1, 1)),
            FeatureMetadata('autocorr_10', 'Autocorrelation lag 10', 'autocorrelation', 30, value_range=(-1, 1)),
            # Volume
            FeatureMetadata('volume_ratio_10', 'Volume ratio (10 period avg)', 'volume', 10),
            FeatureMetadata('volume_ratio_20', 'Volume ratio (20 period avg)', 'volume', 20),
            FeatureMetadata('volume_std_10', 'Volume coefficient of variation (10)', 'volume', 10),
            FeatureMetadata('volume_std_20', 'Volume coefficient of variation (20)', 'volume', 20),
            FeatureMetadata('volume_trend', 'Volume trend (5/20 ratio)', 'volume', 20),
            FeatureMetadata('price_volume_corr', 'Price-volume correlation', 'volume', 20, value_range=(-1, 1)),
            # Price ratios
            FeatureMetadata('high_low_ratio', 'High/Low ratio', 'price_ratio', 1),
            FeatureMetadata('high_low_pct', 'High-Low as % of close', 'price_ratio', 1),
            FeatureMetadata('close_open_ratio', 'Close/Open ratio', 'price_ratio', 1),
            FeatureMetadata('close_open_pct', 'Close-Open as % of open', 'price_ratio', 1),
            FeatureMetadata('close_position', 'Close position in bar range', 'price_ratio', 1, value_range=(0, 1)),
            FeatureMetadata('gap', 'Gap from previous close', 'price_ratio', 1),
            # Trend
            FeatureMetadata('trend_slope_10', 'Linear regression slope (10)', 'trend', 10),
            FeatureMetadata('trend_slope_20', 'Linear regression slope (20)', 'trend', 20),
            FeatureMetadata('trend_r2_10', 'Linear fit R-squared (10)', 'trend', 10, value_range=(0, 1)),
            FeatureMetadata('trend_r2_20', 'Linear fit R-squared (20)', 'trend', 20, value_range=(0, 1)),
            # Z-scores
            FeatureMetadata('zscore_20', 'Price z-score (20 periods)', 'zscore', 20),
            FeatureMetadata('zscore_50', 'Price z-score (50 periods)', 'zscore', 50),
            FeatureMetadata('volume_zscore_20', 'Volume z-score (20 periods)', 'zscore', 20),
        ]
        
        # Add prefix to names
        for m in metadata:
            m.name = self._add_prefix(m.name)
        
        return metadata
