"""
Temporal Feature Extractor

ARY-1084: Feature Engineering Pipeline
Extracts time-based features from datetime index.

Created: 2026-02-17
"""

from typing import List
import pandas as pd
import numpy as np

from .base import FeatureExtractor, FeatureMetadata


class TemporalFeatureExtractor(FeatureExtractor):
    """
    Extracts temporal features from datetime index.
    
    Implements features for:
    - Hour of day (with cyclical encoding)
    - Day of week (with cyclical encoding)
    - Day of month
    - Month of year (with cyclical encoding)
    - Market session indicators (US market hours)
    - Quarter and year features
    """
    
    def __init__(self, prefix: str = "", timezone: str = "UTC"):
        """
        Initialize temporal feature extractor.
        
        Args:
            prefix: Optional prefix for feature names
            timezone: Timezone for market session calculations
        """
        super().__init__(prefix)
        self.timezone = timezone
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from DataFrame index.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with temporal features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        features = pd.DataFrame(index=df.index)
        idx = df.index
        
        # === HOUR FEATURES ===
        
        # Raw hour (0-23)
        features[self._add_prefix('hour')] = idx.hour
        
        # Cyclical encoding for hour (preserves continuity: 23 is close to 0)
        features[self._add_prefix('hour_sin')] = np.sin(2 * np.pi * idx.hour / 24)
        features[self._add_prefix('hour_cos')] = np.cos(2 * np.pi * idx.hour / 24)
        
        # Minute of day (0-1439)
        features[self._add_prefix('minute_of_day')] = idx.hour * 60 + idx.minute
        features[self._add_prefix('minute_of_day_sin')] = np.sin(2 * np.pi * features[self._add_prefix('minute_of_day')] / 1440)
        features[self._add_prefix('minute_of_day_cos')] = np.cos(2 * np.pi * features[self._add_prefix('minute_of_day')] / 1440)
        
        # === DAY OF WEEK FEATURES ===
        
        # Raw day of week (0=Monday, 6=Sunday)
        features[self._add_prefix('day_of_week')] = idx.dayofweek
        
        # Cyclical encoding for day of week
        features[self._add_prefix('dow_sin')] = np.sin(2 * np.pi * idx.dayofweek / 7)
        features[self._add_prefix('dow_cos')] = np.cos(2 * np.pi * idx.dayofweek / 7)
        
        # Weekend indicator
        features[self._add_prefix('is_weekend')] = (idx.dayofweek >= 5).astype(int)
        
        # Monday/Friday effects (known market anomalies)
        features[self._add_prefix('is_monday')] = (idx.dayofweek == 0).astype(int)
        features[self._add_prefix('is_friday')] = (idx.dayofweek == 4).astype(int)
        
        # === DAY OF MONTH FEATURES ===
        
        # Raw day of month (1-31)
        features[self._add_prefix('day_of_month')] = idx.day
        
        # Cyclical encoding (approximate, using 30 days)
        features[self._add_prefix('dom_sin')] = np.sin(2 * np.pi * idx.day / 30)
        features[self._add_prefix('dom_cos')] = np.cos(2 * np.pi * idx.day / 30)
        
        # Month start/end effects
        features[self._add_prefix('is_month_start')] = idx.is_month_start.astype(int)
        features[self._add_prefix('is_month_end')] = idx.is_month_end.astype(int)
        
        # Week of month (1-5)
        features[self._add_prefix('week_of_month')] = (idx.day - 1) // 7 + 1
        
        # === MONTH FEATURES ===
        
        # Raw month (1-12)
        features[self._add_prefix('month')] = idx.month
        
        # Cyclical encoding for month
        features[self._add_prefix('month_sin')] = np.sin(2 * np.pi * idx.month / 12)
        features[self._add_prefix('month_cos')] = np.cos(2 * np.pi * idx.month / 12)
        
        # Quarter (1-4)
        features[self._add_prefix('quarter')] = idx.quarter
        
        # Quarter start/end
        features[self._add_prefix('is_quarter_start')] = idx.is_quarter_start.astype(int)
        features[self._add_prefix('is_quarter_end')] = idx.is_quarter_end.astype(int)
        
        # === YEAR FEATURES ===
        
        # Day of year (1-366)
        features[self._add_prefix('day_of_year')] = idx.dayofyear
        
        # Cyclical encoding for day of year
        features[self._add_prefix('doy_sin')] = np.sin(2 * np.pi * idx.dayofyear / 365)
        features[self._add_prefix('doy_cos')] = np.cos(2 * np.pi * idx.dayofyear / 365)
        
        # Week of year (1-52)
        features[self._add_prefix('week_of_year')] = idx.isocalendar().week.values
        
        # Year start/end
        features[self._add_prefix('is_year_start')] = idx.is_year_start.astype(int)
        features[self._add_prefix('is_year_end')] = idx.is_year_end.astype(int)
        
        # === MARKET SESSION FEATURES (US Market) ===
        
        # US market hours (9:30 AM - 4:00 PM ET)
        # Assuming UTC, US market is 14:30 - 21:00 UTC (EST) or 13:30 - 20:00 UTC (EDT)
        hour = idx.hour
        minute = idx.minute
        time_decimal = hour + minute / 60
        
        # Regular trading hours (approximate for UTC)
        features[self._add_prefix('is_us_market_hours')] = (
            (time_decimal >= 14.5) & (time_decimal < 21) &
            (idx.dayofweek < 5)
        ).astype(int)
        
        # Pre-market (4:00 AM - 9:30 AM ET = 9:00 - 14:30 UTC)
        features[self._add_prefix('is_us_premarket')] = (
            (time_decimal >= 9) & (time_decimal < 14.5) &
            (idx.dayofweek < 5)
        ).astype(int)
        
        # Post-market (4:00 PM - 8:00 PM ET = 21:00 - 01:00 UTC)
        features[self._add_prefix('is_us_postmarket')] = (
            ((time_decimal >= 21) | (time_decimal < 1)) &
            (idx.dayofweek < 5)
        ).astype(int)
        
        # Asian session (Tokyo: 00:00 - 09:00 UTC)
        features[self._add_prefix('is_asian_session')] = (
            (time_decimal >= 0) & (time_decimal < 9)
        ).astype(int)
        
        # European session (London: 08:00 - 17:00 UTC)
        features[self._add_prefix('is_european_session')] = (
            (time_decimal >= 8) & (time_decimal < 17)
        ).astype(int)
        
        # Session overlap (London-NY: 13:00 - 17:00 UTC)
        features[self._add_prefix('is_session_overlap')] = (
            (time_decimal >= 13) & (time_decimal < 17) &
            (idx.dayofweek < 5)
        ).astype(int)
        
        # === CRYPTO-SPECIFIC (24/7 market) ===
        
        # Time since midnight UTC (normalized 0-1)
        features[self._add_prefix('time_of_day_normalized')] = time_decimal / 24
        
        # Weekend trading (crypto trades 24/7)
        features[self._add_prefix('is_crypto_weekend')] = features[self._add_prefix('is_weekend')]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all temporal feature names."""
        names = [
            # Hour
            'hour', 'hour_sin', 'hour_cos',
            'minute_of_day', 'minute_of_day_sin', 'minute_of_day_cos',
            # Day of week
            'day_of_week', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_monday', 'is_friday',
            # Day of month
            'day_of_month', 'dom_sin', 'dom_cos',
            'is_month_start', 'is_month_end', 'week_of_month',
            # Month
            'month', 'month_sin', 'month_cos',
            'quarter', 'is_quarter_start', 'is_quarter_end',
            # Year
            'day_of_year', 'doy_sin', 'doy_cos',
            'week_of_year', 'is_year_start', 'is_year_end',
            # Market sessions
            'is_us_market_hours', 'is_us_premarket', 'is_us_postmarket',
            'is_asian_session', 'is_european_session', 'is_session_overlap',
            # Crypto
            'time_of_day_normalized', 'is_crypto_weekend'
        ]
        return [self._add_prefix(name) for name in names]
    
    def get_feature_metadata(self) -> List[FeatureMetadata]:
        """Get metadata for all temporal features."""
        metadata = [
            # Hour
            FeatureMetadata('hour', 'Hour of day (0-23)', 'temporal', 0, value_range=(0, 23)),
            FeatureMetadata('hour_sin', 'Hour cyclical sine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('hour_cos', 'Hour cyclical cosine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('minute_of_day', 'Minute of day (0-1439)', 'temporal', 0, value_range=(0, 1439)),
            FeatureMetadata('minute_of_day_sin', 'Minute of day sine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('minute_of_day_cos', 'Minute of day cosine encoding', 'temporal', 0, value_range=(-1, 1)),
            # Day of week
            FeatureMetadata('day_of_week', 'Day of week (0=Mon, 6=Sun)', 'temporal', 0, value_range=(0, 6)),
            FeatureMetadata('dow_sin', 'Day of week sine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('dow_cos', 'Day of week cosine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('is_weekend', 'Weekend indicator', 'temporal', 0, value_range=(0, 1)),
            FeatureMetadata('is_monday', 'Monday indicator', 'temporal', 0, value_range=(0, 1)),
            FeatureMetadata('is_friday', 'Friday indicator', 'temporal', 0, value_range=(0, 1)),
            # Day of month
            FeatureMetadata('day_of_month', 'Day of month (1-31)', 'temporal', 0, value_range=(1, 31)),
            FeatureMetadata('dom_sin', 'Day of month sine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('dom_cos', 'Day of month cosine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('is_month_start', 'Month start indicator', 'temporal', 0, value_range=(0, 1)),
            FeatureMetadata('is_month_end', 'Month end indicator', 'temporal', 0, value_range=(0, 1)),
            FeatureMetadata('week_of_month', 'Week of month (1-5)', 'temporal', 0, value_range=(1, 5)),
            # Month
            FeatureMetadata('month', 'Month (1-12)', 'temporal', 0, value_range=(1, 12)),
            FeatureMetadata('month_sin', 'Month sine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('month_cos', 'Month cosine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('quarter', 'Quarter (1-4)', 'temporal', 0, value_range=(1, 4)),
            FeatureMetadata('is_quarter_start', 'Quarter start indicator', 'temporal', 0, value_range=(0, 1)),
            FeatureMetadata('is_quarter_end', 'Quarter end indicator', 'temporal', 0, value_range=(0, 1)),
            # Year
            FeatureMetadata('day_of_year', 'Day of year (1-366)', 'temporal', 0, value_range=(1, 366)),
            FeatureMetadata('doy_sin', 'Day of year sine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('doy_cos', 'Day of year cosine encoding', 'temporal', 0, value_range=(-1, 1)),
            FeatureMetadata('week_of_year', 'Week of year (1-52)', 'temporal', 0, value_range=(1, 52)),
            FeatureMetadata('is_year_start', 'Year start indicator', 'temporal', 0, value_range=(0, 1)),
            FeatureMetadata('is_year_end', 'Year end indicator', 'temporal', 0, value_range=(0, 1)),
            # Market sessions
            FeatureMetadata('is_us_market_hours', 'US market hours indicator', 'session', 0, value_range=(0, 1)),
            FeatureMetadata('is_us_premarket', 'US pre-market indicator', 'session', 0, value_range=(0, 1)),
            FeatureMetadata('is_us_postmarket', 'US post-market indicator', 'session', 0, value_range=(0, 1)),
            FeatureMetadata('is_asian_session', 'Asian session indicator', 'session', 0, value_range=(0, 1)),
            FeatureMetadata('is_european_session', 'European session indicator', 'session', 0, value_range=(0, 1)),
            FeatureMetadata('is_session_overlap', 'London-NY overlap indicator', 'session', 0, value_range=(0, 1)),
            # Crypto
            FeatureMetadata('time_of_day_normalized', 'Time of day (0-1)', 'temporal', 0, value_range=(0, 1)),
            FeatureMetadata('is_crypto_weekend', 'Crypto weekend indicator', 'temporal', 0, value_range=(0, 1)),
        ]
        
        # Add prefix to names
        for m in metadata:
            m.name = self._add_prefix(m.name)
        
        return metadata
