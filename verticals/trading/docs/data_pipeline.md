# Data Pipeline Guide

**Last Updated**: February 2026

## Overview

The data pipeline collects, processes, and stores market data for trading models and strategies. It supports both historical and real-time data flows.

## Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Collectors │───►│Preprocessors│───►│  Feature    │───►│   Storage   │
│             │    │             │    │ Engineering │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  Raw Data          Clean Data         Features           Versioned
  (OHLCV)          (Normalized)       (Indicators)         Data
```

## Data Collectors

### Historical Data

Collect historical OHLCV data from exchanges:

```python
from verticals.trading.data.collectors.historical import HistoricalCollector

collector = HistoricalCollector(
    exchange="binance",
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="1h"
)

data = collector.fetch(
    start="2024-01-01",
    end="2024-12-31"
)
```

### Real-time Data

Stream real-time market data:

```python
from verticals.trading.data.collectors.realtime import RealtimeCollector

collector = RealtimeCollector(
    exchange="binance",
    symbols=["BTC/USDT", "ETH/USDT"]
)

async for tick in collector.stream():
    process_tick(tick)
```

### Exchange-Specific Collectors

- **Binance**: `binance_collector.py`
- **Interactive Brokers**: `ib_collector.py`

## Preprocessors

### OHLCV Processor

Clean and normalize OHLCV data:

```python
from verticals.trading.data.preprocessors.ohlcv_processor import OHLCVProcessor

processor = OHLCVProcessor()

clean_data = processor.process(raw_data)
```

Processing steps:
1. Remove duplicates
2. Handle missing values
3. Validate price/volume ranges
4. Normalize timestamps to UTC
5. Resample to target timeframe

### Order Book Processor

Process order book snapshots:

```python
from verticals.trading.data.preprocessors.orderbook_processor import OrderBookProcessor

processor = OrderBookProcessor()

processed = processor.process(orderbook_data)
```

## Feature Engineering

### Technical Indicators

Compute standard technical indicators:

```python
from verticals.trading.data.feature_engineering.technical_indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr
)

features = pd.DataFrame()
features["rsi"] = compute_rsi(data["close"], period=14)
features["macd"], features["signal"] = compute_macd(data["close"])
features["bb_upper"], features["bb_lower"] = compute_bollinger_bands(data["close"])
features["atr"] = compute_atr(data["high"], data["low"], data["close"])
```

### Statistical Features

Compute statistical features:

```python
from verticals.trading.data.feature_engineering.statistical_features import (
    compute_returns,
    compute_volatility,
    compute_skewness,
    compute_kurtosis
)

features["returns"] = compute_returns(data["close"])
features["volatility"] = compute_volatility(data["close"], window=20)
features["skewness"] = compute_skewness(data["close"], window=20)
```

### Microstructure Features

Compute market microstructure features:

```python
from verticals.trading.data.feature_engineering.microstructure_features import (
    compute_spread,
    compute_depth,
    compute_order_imbalance
)

features["spread"] = compute_spread(orderbook)
features["depth"] = compute_depth(orderbook)
features["imbalance"] = compute_order_imbalance(orderbook)
```

## Data Storage

### Data Store

Store and retrieve versioned data:

```python
from verticals.trading.data.storage.data_store import DataStore

store = DataStore(path="./data")

# Save data
store.save(
    data=features,
    symbol="BTC/USDT",
    timeframe="1h",
    version="v1.0.0"
)

# Load data
data = store.load(
    symbol="BTC/USDT",
    timeframe="1h",
    start="2024-01-01",
    end="2024-12-31"
)
```

### Version Manager

Track data versions and lineage:

```python
from verticals.trading.data.storage.version_manager import VersionManager

manager = VersionManager()

# Create new version
version = manager.create_version(
    data_id="btc_usdt_1h",
    source="binance",
    processing_steps=["ohlcv_processor", "technical_indicators"]
)

# Get lineage
lineage = manager.get_lineage(version)
```

## Data Quality

### Validation Checks

- **Completeness**: No missing timestamps
- **Consistency**: Prices within valid ranges
- **Timeliness**: Data freshness within threshold
- **Accuracy**: Cross-validation with multiple sources

### Quality Metrics

```python
from verticals.trading.data.storage.data_store import DataQualityChecker

checker = DataQualityChecker()

report = checker.check(data)
print(f"Completeness: {report.completeness:.2%}")
print(f"Missing values: {report.missing_count}")
print(f"Outliers: {report.outlier_count}")
```

## Configuration

Configure data pipeline in `configs/`:

```yaml
data:
  historical_source: remote
  historical_path: ./data/historical
  realtime_enabled: true
  realtime_buffer_size: 1000
  
  quality_checks:
    enabled: true
    max_gap_seconds: 60
    min_volume_threshold: 1000
```
