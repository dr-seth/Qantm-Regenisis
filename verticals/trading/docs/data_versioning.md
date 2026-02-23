# Data Versioning and Lineage

**ARY-1085: Data Versioning and Lineage**  
**Last Updated:** 2026-02-17

## Overview

The Data Versioning system provides versioned storage for trading data with full lineage tracking. It enables reproducible ML pipelines by tracking the complete history of data transformations from raw data to model training.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DataVersionStore                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   SQLite DB     │  │  Parquet Files  │                  │
│  │   (Metadata)    │  │  (Data Storage) │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           ▼                    ▼                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Lineage Graph                          │   │
│  │   raw_1.0.0 → features_1.0.0 → labels_1.0.0        │   │
│  │       ↓                                             │   │
│  │   raw_1.0.1 → features_1.0.1                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install trading dependencies
pip install -r requirements-trading.txt

# Required packages: pandas, numpy, pyarrow
```

## Quick Start

### Creating Versions

```python
import pandas as pd
from verticals.trading.data.storage import DataVersionStore

# Initialize store
store = DataVersionStore(
    storage_root='data/versioned',
    db_path='data/versions.db'
)

# Load raw data
raw_df = pd.read_parquet('data/historical/BTCUSDT/1h/data.parquet')

# Create raw data version
raw_version = store.create_version(
    df=raw_df,
    dataset_name='BTCUSDT_1h',
    data_type='raw',
    source='binance'
)
print(f"Created: {raw_version.version_id}")  # "1.0.0"
```

### Creating Derived Versions with Lineage

```python
from verticals.trading.data.feature_engineering import FeatureEngineeringPipeline

# Engineer features
pipeline = FeatureEngineeringPipeline()
features_df = pipeline.extract_batch(raw_df)

# Create features version with lineage
features_version = store.create_version(
    df=features_df,
    dataset_name='BTCUSDT_1h',
    data_type='features',
    source='feature_pipeline_v1.0.0',
    parent_version_id=raw_version.version_id,
    transformation='feature_engineering'
)
print(f"Created: {features_version.version_id}")  # "1.0.0"
```

### Loading Data

```python
# Load by version ID
df = store.load_data("1.0.0")

# Get latest version
latest = store.get_latest_version("BTCUSDT_1h", data_type="features")
df = store.load_data(latest.version_id)
```

### Querying Versions

```python
# List all versions
all_versions = store.list_versions()

# Filter by dataset
btc_versions = store.list_versions(dataset_name="BTCUSDT_1h")

# Filter by type
feature_versions = store.list_versions(data_type="features")

# Get specific version
version = store.get_version("1.0.0")
```

## Data Types

| Type | Description | Example Source |
|------|-------------|----------------|
| `raw` | Raw OHLCV data from exchanges | binance, yfinance |
| `features` | Engineered features | feature_pipeline_v1.0.0 |
| `labels` | Training labels | label_generator |
| `predictions` | Model predictions | nano_model_v1.0.0 |

## Lineage Tracking

### Upstream Lineage (Ancestors)

```python
# Get all ancestors of a version
lineage = store.get_lineage("labels_1.0.0")

for record in lineage:
    print(f"Depth {record['depth']}: {record['input_version']} → {record['output_version']}")
    print(f"  Transformation: {record['transformation']}")
```

### Downstream Lineage (Descendants)

```python
# Find all versions derived from raw data
downstream = store.get_downstream("raw_1.0.0")
print(f"Affected versions: {downstream}")
```

### Lineage Example

```
raw_1.0.0 (binance)
    │
    ├── feature_engineering
    ▼
features_1.0.0 (feature_pipeline_v1.0.0)
    │
    ├── label_generation
    ▼
labels_1.0.0 (label_generator)
    │
    ├── model_training
    ▼
model_1.0.0 (nano_model_trainer)
```

## Data Quality Scoring

Each version includes an automatic quality score (0.0 to 1.0):

```python
version = store.create_version(df=df, ...)
print(f"Quality: {version.quality_score:.2f}")
```

### Quality Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Missing Values | 50% | Ratio of NaN cells |
| Duplicates | 30% | Ratio of duplicate rows |
| Outliers | 20% | Values > 5 std devs |

## Storage

### Parquet with Compression

Data is stored in Parquet format with Snappy compression:

```python
store = DataVersionStore(
    storage_root='data/versioned',
    db_path='data/versions.db',
    compression='snappy'  # or 'gzip', 'zstd'
)
```

### Storage Structure

```
data/versioned/
├── BTCUSDT_1h/
│   ├── raw/
│   │   ├── 1.0.0.parquet
│   │   └── 1.0.1.parquet
│   ├── features/
│   │   └── 1.0.0.parquet
│   └── labels/
│       └── 1.0.0.parquet
└── ETHUSDT_1h/
    └── raw/
        └── 1.0.0.parquet
```

### Storage Statistics

```python
stats = store.get_storage_stats()
print(f"Total versions: {stats['total_versions']}")
print(f"Total datasets: {stats['total_datasets']}")
print(f"Total records: {stats['total_records']}")
print(f"Storage size: {stats['storage_size_mb']:.2f} MB")
```

## Integration with Core LineageStore

The DataVersionStore integrates with the core `LineageStore` for unified provenance:

```python
from packages.core.lineage_store import LineageStore

# Create with LineageStore integration
lineage_store = LineageStore(db_path="lineage.db")
version_store = DataVersionStore(
    storage_root='data/versioned',
    db_path='data/versions.db',
    lineage_store=lineage_store
)

# Versions are automatically logged to LineageStore
version = version_store.create_version(...)

# Query unified lineage
events = lineage_store.get_lineage(version.version_id)
```

## DataVersion Schema

```python
@dataclass
class DataVersion:
    version_id: str          # Semantic version (e.g., "1.0.0")
    dataset_name: str        # Dataset name (e.g., "BTCUSDT_1h")
    data_type: str           # Type (raw, features, labels, predictions)
    source: str              # Data source or pipeline
    created_at: datetime     # Creation timestamp
    num_records: int         # Number of records
    start_date: datetime     # Data range start
    end_date: datetime       # Data range end
    schema_version: str      # Schema version
    quality_score: float     # Quality score (0.0-1.0)
    storage_path: str        # Path to Parquet file
    metadata: Dict           # Additional metadata
    parent_version_id: str   # Parent version for lineage
    data_hash: str           # SHA256 hash of data
```

## Version Deletion

```python
# Delete version (metadata only)
store.delete_version("1.0.0")

# Delete version and data file
store.delete_version("1.0.0", delete_data=True)

# Note: Cannot delete versions with downstream dependencies
```

## Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| Create version | <5s | 1 year of 1-minute data |
| Load data | <5s | 1 year of 1-minute data |
| Query versions | <100ms | With indexes |
| Lineage traversal | <100ms | Recursive CTE |

### Compression Ratios

| Compression | Ratio | Speed |
|-------------|-------|-------|
| Snappy | 3-5x | Fast |
| Gzip | 5-8x | Medium |
| Zstd | 6-10x | Fast |

## Testing

```bash
# Run unit tests
pytest verticals/trading/data/storage/tests/test_data_version.py -v

# Run with coverage
pytest verticals/trading/data/storage/tests/ -v --cov=verticals.trading.data.storage
```

## API Reference

### DataVersionStore

| Method | Description |
|--------|-------------|
| `create_version(df, ...)` | Create new version |
| `get_version(version_id)` | Get version by ID |
| `load_data(version_id)` | Load data for version |
| `list_versions(...)` | List versions with filters |
| `get_latest_version(...)` | Get latest version |
| `get_lineage(version_id)` | Get upstream lineage |
| `get_downstream(version_id)` | Get downstream versions |
| `delete_version(version_id)` | Delete version |
| `get_storage_stats()` | Get storage statistics |

## Related Issues

- **ARY-5**: Historical Data Collection (blocked by)
- **ARY-7**: Feature Engineering (blocked by)
- **ARY-9 - ARY-13**: Nano Model Development (blocks)
- **ARY-14**: Nano Model Registry Integration (relates to)

## Changelog

### v1.0.0 (2026-02-17)
- Initial implementation
- DataVersion dataclass
- DataVersionStore with SQLite backend
- Parquet storage with Snappy compression
- Semantic versioning (auto-increment)
- Lineage tracking (parent-child)
- Recursive lineage queries
- Data quality scoring
- Integration with core LineageStore
- Unit and integration tests
