# Feature Engineering Pipeline

**ARY-1084: Feature Engineering Pipeline**  
**Last Updated:** 2026-02-23

## Overview

The Feature Engineering Pipeline extracts trading features from OHLCV market data. It supports both batch processing for historical data and streaming mode for real-time applications.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 FeatureEngineeringPipeline                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Technical     │  │   Statistical   │  │  Temporal   │ │
│  │   Indicators    │  │    Features     │  │  Features   │ │
│  │   (60+ feat)    │  │   (40+ feat)    │  │  (38 feat)  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           └────────────────────┼───────────────────┘        │
│                                ▼                            │
│                    ┌───────────────────┐                    │
│                    │  Combined Output  │                    │
│                    │   (140+ features) │                    │
│                    └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install trading dependencies
pip install -r requirements-trading.txt

# TA-Lib requires system library
# Linux: sudo apt-get install ta-lib
# macOS: brew install ta-lib
```

## Quick Start

### Batch Mode (Historical Data)

```python
import pandas as pd
from verticals.trading.data.feature_engineering import (
    FeatureEngineeringPipeline,
    create_pipeline
)

# Load historical data
df = pd.read_parquet('data/historical/BTCUSDT/1h/data.parquet')

# Create pipeline
pipeline = FeatureEngineeringPipeline()

# Extract features
features = pipeline.extract_batch(df)
print(f"Extracted {len(features.columns)} features")

# Save features
features.to_parquet('data/features/BTCUSDT_1h_features.parquet')
```

### Streaming Mode (Real-time Data)

```python
from verticals.trading.data.feature_engineering import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline()

# Get required lookback period
lookback = pipeline.get_required_lookback()  # ~210 bars

def on_new_bar(historical_data, new_bar):
    """Process new bar in real-time."""
    # Append new bar to historical data
    data = pd.concat([historical_data.iloc[-lookback:], new_bar])
    
    # Extract features for latest bar only
    features = pipeline.extract_streaming(data)
    
    return features
```

## Feature Extractors

### TechnicalIndicatorExtractor

Extracts 60+ technical analysis indicators using TA-Lib.

**Categories:**
- **Trend**: SMA, EMA, MACD, ADX, Parabolic SAR, Aroon
- **Momentum**: RSI, Stochastic, Williams %R, CCI, MFI, ROC, Ultimate Oscillator
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, AD Line, Chaikin Money Flow, VWAP
- **Patterns**: Doji, Hammer, Engulfing, Morning/Evening Star

```python
from verticals.trading.data.feature_engineering import TechnicalIndicatorExtractor

extractor = TechnicalIndicatorExtractor()
features = extractor.extract(df)

# Get feature names
print(extractor.get_feature_names())
```

### StatisticalFeatureExtractor

Extracts 40+ statistical features from price data.

**Categories:**
- **Returns**: Simple, log, cumulative returns at multiple periods
- **Volatility**: Rolling std, Parkinson, Garman-Klass estimators
- **Distribution**: Skewness, kurtosis
- **Autocorrelation**: Lag-based correlations
- **Volume**: Relative volume, volume volatility
- **Trend**: Linear regression slope, R-squared

```python
from verticals.trading.data.feature_engineering import StatisticalFeatureExtractor

extractor = StatisticalFeatureExtractor()
features = extractor.extract(df)
```

### TemporalFeatureExtractor

Extracts 38 time-based features from datetime index.

**Categories:**
- **Hour/Minute**: Raw and cyclical encoding
- **Day of Week**: Raw, cyclical, weekend/Monday/Friday indicators
- **Day/Month/Year**: Various temporal markers
- **Market Sessions**: US, Asian, European session indicators

```python
from verticals.trading.data.feature_engineering import TemporalFeatureExtractor

extractor = TemporalFeatureExtractor()
features = extractor.extract(df)
```

## Pipeline Configuration

```python
from verticals.trading.data.feature_engineering import (
    FeatureEngineeringPipeline,
    PipelineConfig
)

config = PipelineConfig(
    include_technical=True,      # Include technical indicators
    include_statistical=True,    # Include statistical features
    include_temporal=True,       # Include temporal features
    include_raw_ohlcv=True,      # Include raw OHLCV columns
    drop_na=True,                # Drop rows with NaN values
    normalize_features=False,    # Z-score normalization
    feature_prefix=""            # Prefix for feature names
)

pipeline = FeatureEngineeringPipeline(config=config)
```

## Feature Versioning

Save and load feature definitions for reproducibility:

```python
# Save feature definitions
pipeline.save_feature_definitions('data/features/v1.0.0/definitions.json')

# Load feature definitions
definitions = FeatureEngineeringPipeline.load_feature_definitions(
    'data/features/v1.0.0/definitions.json'
)
print(f"Version: {definitions['version']}")
print(f"Features: {definitions['num_features']}")
```

## Feature Validation

Validate extracted features for quality issues:

```python
features = pipeline.extract_batch(df)
validation = pipeline.validate_features(features)

if not validation['is_valid']:
    print("Issues found:")
    for issue in validation['issues']:
        print(f"  - {issue['type']}: {issue}")
```

## Performance Metrics

Access metrics from the last extraction:

```python
features = pipeline.extract_batch(df)
metrics = pipeline.get_last_metrics()

print(f"Input rows: {metrics.num_input_rows}")
print(f"Output rows: {metrics.num_output_rows}")
print(f"Features: {metrics.num_features}")
print(f"Time: {metrics.extraction_time_ms:.2f}ms")
print(f"Rows dropped: {metrics.rows_dropped}")
```

## Performance Benchmarks

| Mode | Data Size | Time | Notes |
|------|-----------|------|-------|
| Batch | 5,000 rows | <5s | Full pipeline |
| Batch | 50,000 rows | <30s | Full pipeline |
| Streaming | 200 rows | <100ms | Per-bar latency |

## Feature Catalog

### Technical Indicators (60+ features)

| Feature | Description | Lookback |
|---------|-------------|----------|
| `sma_10`, `sma_20`, `sma_50`, `sma_200` | Simple Moving Averages | 10-200 |
| `ema_10`, `ema_20`, `ema_50` | Exponential Moving Averages | 10-50 |
| `macd`, `macd_signal`, `macd_hist` | MACD Indicator | 35 |
| `rsi_7`, `rsi_14`, `rsi_21` | Relative Strength Index | 7-21 |
| `bb_upper`, `bb_middle`, `bb_lower` | Bollinger Bands | 20 |
| `atr_7`, `atr_14` | Average True Range | 7-14 |
| `adx`, `plus_di`, `minus_di` | ADX Indicator | 14 |
| `stoch_k`, `stoch_d` | Stochastic Oscillator | 14 |
| `obv`, `ad`, `cmf` | Volume Indicators | 1-20 |

### Statistical Features (40+ features)

| Feature | Description | Lookback |
|---------|-------------|----------|
| `return_1`, `return_5`, `return_10`, `return_20` | Simple Returns | 1-20 |
| `log_return`, `log_return_5`, `log_return_10` | Log Returns | 1-10 |
| `volatility_10`, `volatility_20`, `volatility_50` | Annualized Volatility | 10-50 |
| `parkinson_vol_10`, `parkinson_vol_20` | Parkinson Volatility | 10-20 |
| `skewness_20`, `skewness_50` | Return Skewness | 20-50 |
| `kurtosis_20`, `kurtosis_50` | Return Kurtosis | 20-50 |
| `autocorr_1`, `autocorr_5`, `autocorr_10` | Autocorrelation | 20-30 |
| `zscore_20`, `zscore_50` | Price Z-Score | 20-50 |

### Temporal Features (38 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `hour`, `hour_sin`, `hour_cos` | Hour of day | 0-23 |
| `day_of_week`, `dow_sin`, `dow_cos` | Day of week | 0-6 |
| `is_weekend`, `is_monday`, `is_friday` | Day indicators | 0-1 |
| `is_us_market_hours` | US market session | 0-1 |
| `is_asian_session`, `is_european_session` | Global sessions | 0-1 |

## Custom Extractors

Create custom feature extractors by extending the base class:

```python
from verticals.trading.data.feature_engineering import FeatureExtractor

class CustomExtractor(FeatureExtractor):
    def extract(self, df):
        self._validate_input(df)
        features = pd.DataFrame(index=df.index)
        
        # Add custom features
        features['custom_feature'] = df['close'].rolling(10).mean()
        
        return features
    
    def get_feature_names(self):
        return ['custom_feature']

# Use with pipeline
pipeline = FeatureEngineeringPipeline(extractors=[
    TechnicalIndicatorExtractor(),
    CustomExtractor()
])
```

## Testing

Run tests:

```bash
# Unit tests
pytest verticals/trading/data/feature_engineering/tests/test_extractors.py -v

# Integration tests
pytest verticals/trading/data/feature_engineering/tests/test_pipeline.py -v

# All tests with coverage
pytest verticals/trading/data/feature_engineering/tests/ -v --cov=verticals.trading.data.feature_engineering
```

## Dependencies

- **pandas** >= 2.1.4
- **numpy** >= 1.26.2
- **scipy** >= 1.11.4
- **TA-Lib** >= 0.4.28

## Related Issues

- **ARY-5**: Historical Data Collection (blocked by)
- **ARY-6**: Real-time Data Feeds (blocked by)
- **ARY-8**: Data Versioning (relates to)
- **ARY-9 - ARY-13**: Nano Model Development (blocks)

## AARA / MCP / FastAPI Integration

All feature engineering capabilities are accessible through the AARA single point of entry.

### MCP Tools (via `aara_orchestrate`)

| Tool | Description |
|------|-------------|
| `trading_feature_extract_batch` | Batch extraction from historical OHLCV data |
| `trading_feature_extract_streaming` | Streaming extraction for latest bar (<100ms) |
| `trading_feature_list` | List all features with metadata |
| `trading_feature_validate` | Validate extracted features for quality |
| `trading_feature_save_definitions` | Save feature definitions for versioning |

### REST API (internal port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/trading-features/extract-batch` | Batch extraction |
| POST | `/api/v1/trading-features/extract-streaming` | Streaming extraction |
| GET | `/api/v1/trading-features/list` | List features |
| POST | `/api/v1/trading-features/validate` | Validate features |
| POST | `/api/v1/trading-features/save-definitions` | Save definitions |

### NL Routing Keywords

`extract feature`, `feature engineer`, `trading feature`, `technical indicator`, `statistical feature`, `streaming feature`, `realtime feature`, `list feature`, `feature catalog`, `validate feature`, `feature quality`, `save feature definition`, `feature version`

## Changelog

### v1.1.0 (2026-02-23)
- AARA engine registration (5 tools in `_BUILTIN_TOOLS`)
- MCP tool definitions (5 tools)
- FastAPI REST endpoints (5 routes)
- NL routing keywords (13 keywords)
- Updated `__init__.py` exports

### v1.0.0 (2026-02-17)
- Initial implementation
- TechnicalIndicatorExtractor with 60+ indicators
- StatisticalFeatureExtractor with 40+ features
- TemporalFeatureExtractor with 38 features
- FeatureEngineeringPipeline with batch/streaming modes
- Feature versioning and validation
- Unit and integration tests
