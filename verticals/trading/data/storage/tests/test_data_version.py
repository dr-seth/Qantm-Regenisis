"""
Unit and Integration Tests for Data Version Store

ARY-1085: Data Versioning and Lineage
Tests for DataVersion and DataVersionStore classes.

Created: 2026-02-17
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from verticals.trading.data.storage.data_version import (
    DataVersion,
    DataLineage,
    DataVersionStore,
    _compute_data_hash,
    _generate_id,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range(start='2025-01-01', periods=n, freq='1h')
    
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_price = low + (high - low) * np.random.rand(n)
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
def sample_features_data(sample_ohlcv_data):
    """Generate sample features data."""
    df = sample_ohlcv_data.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi_14'] = 50 + np.random.randn(len(df)) * 10
    df['volatility'] = df['close'].rolling(20).std()
    return df.dropna()


@pytest.fixture
def version_store(temp_storage):
    """Create a DataVersionStore instance."""
    storage_root = Path(temp_storage) / "versioned"
    db_path = Path(temp_storage) / "versions.db"
    
    store = DataVersionStore(
        storage_root=storage_root,
        db_path=db_path
    )
    yield store
    store.close()


class TestDataVersion:
    """Tests for DataVersion dataclass."""
    
    def test_create_data_version(self):
        """Test creating a DataVersion object."""
        version = DataVersion(
            version_id="1.0.0",
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance",
            created_at=datetime.utcnow(),
            num_records=1000,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
        )
        
        assert version.version_id == "1.0.0"
        assert version.dataset_name == "BTCUSDT_1h"
        assert version.data_type == "raw"
        assert version.num_records == 1000
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        version = DataVersion(
            version_id="1.0.0",
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance",
            created_at=datetime(2025, 1, 15, 12, 0, 0),
            num_records=1000,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            metadata={"key": "value"},
        )
        
        data = version.to_dict()
        
        assert data["version_id"] == "1.0.0"
        assert data["dataset_name"] == "BTCUSDT_1h"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "version_id": "1.0.0",
            "dataset_name": "BTCUSDT_1h",
            "data_type": "raw",
            "source": "binance",
            "created_at": "2025-01-15T12:00:00",
            "num_records": 1000,
            "start_date": "2025-01-01T00:00:00",
            "end_date": "2025-01-31T00:00:00",
            "quality_score": 0.95,
        }
        
        version = DataVersion.from_dict(data)
        
        assert version.version_id == "1.0.0"
        assert version.quality_score == 0.95
        assert isinstance(version.created_at, datetime)


class TestDataVersionStore:
    """Tests for DataVersionStore."""
    
    def test_create_version(self, version_store, sample_ohlcv_data):
        """Test creating a new data version."""
        version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        assert version.version_id == "1.0.0"
        assert version.dataset_name == "BTCUSDT_1h"
        assert version.data_type == "raw"
        assert version.num_records == len(sample_ohlcv_data)
        assert version.quality_score > 0
        assert Path(version.storage_path).exists()
    
    def test_create_multiple_versions(self, version_store, sample_ohlcv_data):
        """Test creating multiple versions increments version ID."""
        v1 = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        v2 = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        v3 = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        assert v1.version_id == "1.0.0"
        assert v2.version_id == "1.0.1"
        assert v3.version_id == "1.0.2"
    
    def test_get_version(self, version_store, sample_ohlcv_data):
        """Test retrieving a version by ID."""
        created = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        retrieved = version_store.get_version(created.version_id)
        
        assert retrieved is not None
        assert retrieved.version_id == created.version_id
        assert retrieved.dataset_name == created.dataset_name
    
    def test_get_version_not_found(self, version_store):
        """Test retrieving non-existent version returns None."""
        result = version_store.get_version("nonexistent")
        assert result is None
    
    def test_load_data(self, version_store, sample_ohlcv_data):
        """Test loading data for a version."""
        version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        loaded = version_store.load_data(version.version_id)
        
        assert len(loaded) == len(sample_ohlcv_data)
        pd.testing.assert_frame_equal(loaded, sample_ohlcv_data)
    
    def test_load_data_not_found(self, version_store):
        """Test loading data for non-existent version raises error."""
        with pytest.raises(ValueError, match="Version not found"):
            version_store.load_data("nonexistent")
    
    def test_list_versions(self, version_store, sample_ohlcv_data):
        """Test listing versions."""
        # Create multiple versions
        version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="ETHUSDT_1h",
            data_type="raw",
            source="binance"
        )
        version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="features",
            source="pipeline"
        )
        
        # List all
        all_versions = version_store.list_versions()
        assert len(all_versions) == 3
        
        # Filter by dataset
        btc_versions = version_store.list_versions(dataset_name="BTCUSDT_1h")
        assert len(btc_versions) == 2
        
        # Filter by type
        raw_versions = version_store.list_versions(data_type="raw")
        assert len(raw_versions) == 2
    
    def test_get_latest_version(self, version_store, sample_ohlcv_data):
        """Test getting the latest version."""
        version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        v2 = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        latest = version_store.get_latest_version("BTCUSDT_1h")
        
        assert latest is not None
        assert latest.version_id == v2.version_id


class TestDataLineage:
    """Tests for lineage tracking."""
    
    def test_create_version_with_parent(self, version_store, sample_ohlcv_data, sample_features_data):
        """Test creating a version with parent lineage."""
        # Create raw data version
        raw_version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        # Create features version with lineage
        features_version = version_store.create_version(
            df=sample_features_data,
            dataset_name="BTCUSDT_1h",
            data_type="features",
            source="feature_pipeline_v1.0.0",
            parent_version_id=raw_version.version_id,
            transformation="feature_engineering"
        )
        
        assert features_version.parent_version_id == raw_version.version_id
    
    def test_get_lineage(self, version_store, sample_ohlcv_data, sample_features_data):
        """Test getting lineage for a version."""
        # Create chain: raw -> features
        raw_version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        features_version = version_store.create_version(
            df=sample_features_data,
            dataset_name="BTCUSDT_1h",
            data_type="features",
            source="feature_pipeline",
            parent_version_id=raw_version.version_id,
            transformation="feature_engineering"
        )
        
        # Get lineage for features version
        lineage = version_store.get_lineage(features_version.version_id)
        
        assert len(lineage) == 1
        assert lineage[0]['input_version'] == raw_version.version_id
        assert lineage[0]['transformation'] == "feature_engineering"
    
    def test_get_downstream(self, version_store, sample_ohlcv_data, sample_features_data):
        """Test getting downstream versions."""
        # Create chain: raw -> features
        raw_version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        features_version = version_store.create_version(
            df=sample_features_data,
            dataset_name="BTCUSDT_1h",
            data_type="features",
            source="feature_pipeline",
            parent_version_id=raw_version.version_id,
            transformation="feature_engineering"
        )
        
        # Get downstream from raw
        downstream = version_store.get_downstream(raw_version.version_id)
        
        assert len(downstream) == 1
        assert features_version.version_id in downstream
    
    def test_multi_level_lineage(self, version_store, sample_ohlcv_data, sample_features_data):
        """Test multi-level lineage chain."""
        # Create chain: raw -> features -> labels
        raw = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        features = version_store.create_version(
            df=sample_features_data,
            dataset_name="BTCUSDT_1h",
            data_type="features",
            source="feature_pipeline",
            parent_version_id=raw.version_id,
            transformation="feature_engineering"
        )
        
        labels_df = sample_features_data.copy()
        labels_df['label'] = (labels_df['close'].shift(-1) > labels_df['close']).astype(int)
        
        labels = version_store.create_version(
            df=labels_df.dropna(),
            dataset_name="BTCUSDT_1h",
            data_type="labels",
            source="label_generator",
            parent_version_id=features.version_id,
            transformation="label_generation"
        )
        
        # Get full lineage for labels
        lineage = version_store.get_lineage(labels.version_id)
        
        assert len(lineage) == 2
        
        # Get all downstream from raw
        downstream = version_store.get_downstream(raw.version_id)
        assert len(downstream) == 2


class TestDataQuality:
    """Tests for data quality scoring."""
    
    def test_quality_score_perfect_data(self, version_store, sample_ohlcv_data):
        """Test quality score for clean data."""
        version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        # Clean data should have high quality score
        assert version.quality_score > 0.9
    
    def test_quality_score_missing_values(self, version_store):
        """Test quality score with missing values."""
        df = pd.DataFrame({
            'open': [100, np.nan, 102, np.nan, 104],
            'high': [101, 102, np.nan, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 2000, 3000, 4000, 5000]
        }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        
        version = version_store.create_version(
            df=df,
            dataset_name="TEST",
            data_type="raw",
            source="test"
        )
        
        # Data with missing values should have lower quality score
        assert version.quality_score < 1.0
    
    def test_quality_score_duplicates(self, version_store):
        """Test quality score with duplicate rows."""
        dates = pd.date_range('2025-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'open': [100, 100, 100, 103, 104],
            'high': [101, 101, 101, 104, 105],
            'low': [99, 99, 99, 102, 103],
            'close': [100, 100, 100, 103, 104],
            'volume': [1000, 1000, 1000, 4000, 5000]
        }, index=dates)
        
        version = version_store.create_version(
            df=df,
            dataset_name="TEST",
            data_type="raw",
            source="test"
        )
        
        # Data with duplicates should have lower quality score
        assert version.quality_score < 1.0


class TestStorageStats:
    """Tests for storage statistics."""
    
    def test_get_storage_stats(self, version_store, sample_ohlcv_data):
        """Test getting storage statistics."""
        # Create some versions
        version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="ETHUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        stats = version_store.get_storage_stats()
        
        assert stats["total_versions"] == 2
        assert stats["total_datasets"] == 2
        assert stats["total_records"] == len(sample_ohlcv_data) * 2
        assert stats["storage_size_mb"] > 0


class TestDeleteVersion:
    """Tests for version deletion."""
    
    def test_delete_version(self, version_store, sample_ohlcv_data):
        """Test deleting a version."""
        version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        result = version_store.delete_version(version.version_id)
        
        assert result is True
        assert version_store.get_version(version.version_id) is None
    
    def test_delete_version_with_data(self, version_store, sample_ohlcv_data):
        """Test deleting a version with data file."""
        version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        storage_path = Path(version.storage_path)
        assert storage_path.exists()
        
        version_store.delete_version(version.version_id, delete_data=True)
        
        assert not storage_path.exists()
    
    def test_delete_version_with_downstream(self, version_store, sample_ohlcv_data, sample_features_data):
        """Test that deleting version with downstream dependencies fails."""
        raw = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        version_store.create_version(
            df=sample_features_data,
            dataset_name="BTCUSDT_1h",
            data_type="features",
            source="pipeline",
            parent_version_id=raw.version_id,
            transformation="feature_engineering"
        )
        
        with pytest.raises(ValueError, match="downstream dependencies"):
            version_store.delete_version(raw.version_id)


class TestDataHash:
    """Tests for data hashing."""
    
    def test_compute_data_hash(self, sample_ohlcv_data):
        """Test computing data hash."""
        hash1 = _compute_data_hash(sample_ohlcv_data)
        hash2 = _compute_data_hash(sample_ohlcv_data)
        
        assert hash1 == hash2
        assert len(hash1) == 16
    
    def test_different_data_different_hash(self, sample_ohlcv_data):
        """Test that different data produces different hash."""
        df2 = sample_ohlcv_data.copy()
        df2.iloc[0, 0] = 999999
        
        hash1 = _compute_data_hash(sample_ohlcv_data)
        hash2 = _compute_data_hash(df2)
        
        assert hash1 != hash2
    
    def test_version_has_data_hash(self, version_store, sample_ohlcv_data):
        """Test that created version has data hash."""
        version = version_store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        assert version.data_hash is not None
        assert len(version.data_hash) == 16


class TestCompression:
    """Tests for storage compression."""
    
    def test_parquet_compression(self, temp_storage, sample_ohlcv_data):
        """Test that Parquet files are compressed."""
        storage_root = Path(temp_storage) / "versioned"
        db_path = Path(temp_storage) / "versions.db"
        
        store = DataVersionStore(
            storage_root=storage_root,
            db_path=db_path,
            compression="snappy"
        )
        
        version = store.create_version(
            df=sample_ohlcv_data,
            dataset_name="BTCUSDT_1h",
            data_type="raw",
            source="binance"
        )
        
        # Check file exists and is smaller than uncompressed would be
        storage_path = Path(version.storage_path)
        assert storage_path.exists()
        
        # Parquet with compression should be efficient
        file_size = storage_path.stat().st_size
        assert file_size > 0
        
        store.close()
