"""
Data Version Store

ARY-1085: Data Versioning and Lineage
Provides versioned storage for trading data with lineage tracking.

Created: 2026-02-17
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from packages.core.lineage_store import LineageStore
    from packages.core.types import LineageEvent, LineageEventType
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False
    LineageStore = None
    LineageEvent = None
    LineageEventType = None


def _generate_id(prefix: str = "dv") -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _compute_data_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of DataFrame contents."""
    # Use pandas hash for efficiency
    data_bytes = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:16]


@dataclass
class DataVersion:
    """
    Represents a versioned snapshot of trading data.
    
    Attributes:
        version_id: Unique semantic version (e.g., "1.0.0", "1.0.1")
        dataset_name: Name of the dataset (e.g., "BTCUSDT_1h")
        data_type: Type of data (raw, features, labels, predictions)
        source: Data source (e.g., "binance", "feature_pipeline_v1.0.0")
        created_at: When this version was created
        num_records: Number of records in the dataset
        start_date: Start date of data range
        end_date: End date of data range
        schema_version: Version of the data schema
        quality_score: Data quality score (0.0 to 1.0)
        storage_path: Path to the stored Parquet file
        metadata: Additional metadata
        parent_version_id: ID of parent version (for lineage)
        data_hash: Hash of the data contents
    """
    version_id: str
    dataset_name: str
    data_type: str
    source: str
    created_at: datetime
    num_records: int
    start_date: datetime
    end_date: datetime
    schema_version: str = "1.0.0"
    quality_score: float = 1.0
    storage_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version_id: Optional[str] = None
    data_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version_id": self.version_id,
            "dataset_name": self.dataset_name,
            "data_type": self.data_type,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "num_records": self.num_records,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "schema_version": self.schema_version,
            "quality_score": self.quality_score,
            "storage_path": self.storage_path,
            "metadata": self.metadata,
            "parent_version_id": self.parent_version_id,
            "data_hash": self.data_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataVersion":
        """Deserialize from dictionary."""
        return cls(
            version_id=data["version_id"],
            dataset_name=data["dataset_name"],
            data_type=data["data_type"],
            source=data["source"],
            created_at=datetime.fromisoformat(data["created_at"]),
            num_records=data["num_records"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            schema_version=data.get("schema_version", "1.0.0"),
            quality_score=data.get("quality_score", 1.0),
            storage_path=data.get("storage_path", ""),
            metadata=data.get("metadata", {}),
            parent_version_id=data.get("parent_version_id"),
            data_hash=data.get("data_hash"),
        )


@dataclass
class DataLineage:
    """
    Represents a lineage relationship between data versions.
    
    Attributes:
        output_version_id: ID of the output/derived version
        input_version_id: ID of the input/source version
        transformation: Description of the transformation applied
        created_at: When this lineage was recorded
        metadata: Additional metadata about the transformation
    """
    output_version_id: str
    input_version_id: str
    transformation: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataVersionStore:
    """
    Versioned data storage with lineage tracking.
    
    Provides:
    - Semantic versioning for datasets
    - Parquet storage with Snappy compression
    - SQLite metadata database
    - Lineage tracking (parent-child relationships)
    - Data quality scoring
    - Integration with core LineageStore
    
    Example:
        >>> store = DataVersionStore(
        ...     storage_root='data/versioned',
        ...     db_path='data/versions.db'
        ... )
        >>> 
        >>> # Create a new version
        >>> version = store.create_version(
        ...     df=raw_data,
        ...     dataset_name='BTCUSDT_1h',
        ...     data_type='raw',
        ...     source='binance'
        ... )
        >>> 
        >>> # Load data
        >>> df = store.load_data(version.version_id)
        >>> 
        >>> # Query versions
        >>> versions = store.list_versions(dataset_name='BTCUSDT_1h')
    """
    
    def __init__(
        self,
        storage_root: Union[str, Path] = "data/versioned",
        db_path: Union[str, Path] = "data/versions.db",
        lineage_store: Optional[Any] = None,
        compression: str = "snappy",
    ):
        """
        Initialize the data version store.
        
        Args:
            storage_root: Root directory for Parquet files
            db_path: Path to SQLite database
            lineage_store: Optional LineageStore for integration
            compression: Parquet compression (snappy, gzip, zstd)
        """
        self.storage_root = Path(storage_root)
        self.db_path = Path(db_path)
        self.lineage_store = lineage_store
        self.compression = compression
        
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        
        self._init_storage()
        self._init_db()
    
    def _init_storage(self) -> None:
        """Initialize storage directories."""
        self.storage_root.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self) -> None:
        """Initialize the database connection and schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        self._conn.executescript("""
            -- Data versions table
            CREATE TABLE IF NOT EXISTS data_versions (
                version_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                num_records INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                schema_version TEXT DEFAULT '1.0.0',
                quality_score REAL DEFAULT 1.0,
                storage_path TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                parent_version_id TEXT,
                data_hash TEXT,
                FOREIGN KEY (parent_version_id) REFERENCES data_versions(version_id)
            );
            
            -- Data lineage table
            CREATE TABLE IF NOT EXISTS data_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_version_id TEXT NOT NULL,
                input_version_id TEXT NOT NULL,
                transformation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (output_version_id) REFERENCES data_versions(version_id),
                FOREIGN KEY (input_version_id) REFERENCES data_versions(version_id),
                UNIQUE(output_version_id, input_version_id)
            );
            
            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_versions_dataset ON data_versions(dataset_name);
            CREATE INDEX IF NOT EXISTS idx_versions_type ON data_versions(data_type);
            CREATE INDEX IF NOT EXISTS idx_versions_created ON data_versions(created_at);
            CREATE INDEX IF NOT EXISTS idx_versions_parent ON data_versions(parent_version_id);
            CREATE INDEX IF NOT EXISTS idx_lineage_output ON data_lineage(output_version_id);
            CREATE INDEX IF NOT EXISTS idx_lineage_input ON data_lineage(input_version_id);
        """)
        self._conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
    
    def __enter__(self) -> "DataVersionStore":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def create_version(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        data_type: str,
        source: str,
        parent_version_id: Optional[str] = None,
        transformation: Optional[str] = None,
        schema_version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataVersion:
        """
        Create a new data version.
        
        Args:
            df: DataFrame to store
            dataset_name: Name of the dataset (e.g., "BTCUSDT_1h")
            data_type: Type of data (raw, features, labels, predictions)
            source: Data source or pipeline that created this data
            parent_version_id: ID of parent version for lineage
            transformation: Description of transformation from parent
            schema_version: Version of the data schema
            metadata: Additional metadata
        
        Returns:
            DataVersion object for the created version
        """
        with self._lock:
            # Generate version ID
            version_id = self._generate_version_id(dataset_name, data_type)
            
            # Compute data hash
            data_hash = _compute_data_hash(df)
            
            # Compute quality score
            quality_score = self._compute_quality_score(df)
            
            # Determine date range
            if isinstance(df.index, pd.DatetimeIndex):
                start_date = df.index.min().to_pydatetime()
                end_date = df.index.max().to_pydatetime()
            else:
                start_date = datetime.utcnow()
                end_date = datetime.utcnow()
            
            # Create storage path
            storage_dir = self.storage_root / dataset_name / data_type
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_path = storage_dir / f"{version_id}.parquet"
            
            # Save data to Parquet with compression
            df.to_parquet(
                storage_path,
                compression=self.compression,
                index=True
            )
            
            # Create version object
            version = DataVersion(
                version_id=version_id,
                dataset_name=dataset_name,
                data_type=data_type,
                source=source,
                created_at=datetime.utcnow(),
                num_records=len(df),
                start_date=start_date,
                end_date=end_date,
                schema_version=schema_version,
                quality_score=quality_score,
                storage_path=str(storage_path),
                metadata=metadata or {},
                parent_version_id=parent_version_id,
                data_hash=data_hash,
            )
            
            # Store in database
            self._conn.execute("""
                INSERT INTO data_versions (
                    version_id, dataset_name, data_type, source, created_at,
                    num_records, start_date, end_date, schema_version,
                    quality_score, storage_path, metadata, parent_version_id, data_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id,
                version.dataset_name,
                version.data_type,
                version.source,
                version.created_at.isoformat(),
                version.num_records,
                version.start_date.isoformat(),
                version.end_date.isoformat(),
                version.schema_version,
                version.quality_score,
                version.storage_path,
                json.dumps(version.metadata),
                version.parent_version_id,
                version.data_hash,
            ))
            
            # Record lineage if parent exists
            if parent_version_id and transformation:
                self._record_lineage(
                    output_version_id=version_id,
                    input_version_id=parent_version_id,
                    transformation=transformation,
                )
            
            self._conn.commit()
            
            # Log to core LineageStore if available
            self._log_to_lineage_store(version, parent_version_id, transformation)
            
            return version
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """
        Get a specific data version by ID.
        
        Args:
            version_id: Version ID to retrieve
        
        Returns:
            DataVersion object or None if not found
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM data_versions WHERE version_id = ?",
                (version_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_version(row)
    
    def load_data(self, version_id: str) -> pd.DataFrame:
        """
        Load data for a specific version.
        
        Args:
            version_id: Version ID to load
        
        Returns:
            DataFrame with the versioned data
        
        Raises:
            ValueError: If version not found
            FileNotFoundError: If data file is missing
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")
        
        storage_path = Path(version.storage_path)
        if not storage_path.exists():
            raise FileNotFoundError(f"Data file not found: {storage_path}")
        
        return pd.read_parquet(storage_path)
    
    def list_versions(
        self,
        dataset_name: Optional[str] = None,
        data_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DataVersion]:
        """
        List versions matching criteria.
        
        Args:
            dataset_name: Filter by dataset name
            data_type: Filter by data type
            source: Filter by source
            limit: Maximum number of results
            offset: Offset for pagination
        
        Returns:
            List of DataVersion objects
        """
        with self._lock:
            query = "SELECT * FROM data_versions WHERE 1=1"
            params: List[Any] = []
            
            if dataset_name:
                query += " AND dataset_name = ?"
                params.append(dataset_name)
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type)
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = self._conn.execute(query, params)
            
            versions = []
            for row in cursor.fetchall():
                versions.append(self._row_to_version(row))
            
            return versions
    
    def get_latest_version(
        self,
        dataset_name: str,
        data_type: Optional[str] = None,
    ) -> Optional[DataVersion]:
        """
        Get the latest version of a dataset.
        
        Args:
            dataset_name: Dataset name
            data_type: Optional data type filter
        
        Returns:
            Latest DataVersion or None
        """
        versions = self.list_versions(
            dataset_name=dataset_name,
            data_type=data_type,
            limit=1
        )
        return versions[0] if versions else None
    
    def get_lineage(self, version_id: str, max_depth: int = -1) -> List[Dict[str, Any]]:
        """
        Get full lineage for a data version (upstream).
        
        Args:
            version_id: Version ID to trace
            max_depth: Maximum traversal depth (-1 for unlimited)
        
        Returns:
            List of lineage records with depth information
        """
        with self._lock:
            cursor = self._conn.execute("""
                WITH RECURSIVE lineage_tree AS (
                    SELECT 
                        output_version_id, 
                        input_version_id, 
                        transformation,
                        0 as depth
                    FROM data_lineage
                    WHERE output_version_id = ?
                    
                    UNION ALL
                    
                    SELECT 
                        dl.output_version_id, 
                        dl.input_version_id, 
                        dl.transformation,
                        lt.depth + 1
                    FROM data_lineage dl
                    JOIN lineage_tree lt ON dl.output_version_id = lt.input_version_id
                    WHERE (? < 0 OR lt.depth < ?)
                )
                SELECT * FROM lineage_tree
            """, (version_id, max_depth, max_depth))
            
            lineage = []
            for row in cursor.fetchall():
                lineage.append({
                    'output_version': row[0],
                    'input_version': row[1],
                    'transformation': row[2],
                    'depth': row[3]
                })
            
            return lineage
    
    def get_downstream(self, version_id: str, max_depth: int = -1) -> List[str]:
        """
        Get all versions derived from this one (downstream).
        
        Args:
            version_id: Source version ID
            max_depth: Maximum traversal depth (-1 for unlimited)
        
        Returns:
            List of downstream version IDs
        """
        with self._lock:
            cursor = self._conn.execute("""
                WITH RECURSIVE downstream_tree AS (
                    SELECT 
                        output_version_id,
                        0 as depth
                    FROM data_lineage
                    WHERE input_version_id = ?
                    
                    UNION ALL
                    
                    SELECT 
                        dl.output_version_id,
                        dt.depth + 1
                    FROM data_lineage dl
                    JOIN downstream_tree dt ON dl.input_version_id = dt.output_version_id
                    WHERE (? < 0 OR dt.depth < ?)
                )
                SELECT DISTINCT output_version_id FROM downstream_tree
            """, (version_id, max_depth, max_depth))
            
            return [row[0] for row in cursor.fetchall()]
    
    def delete_version(self, version_id: str, delete_data: bool = False) -> bool:
        """
        Delete a data version.
        
        Args:
            version_id: Version ID to delete
            delete_data: Also delete the Parquet file
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            version = self.get_version(version_id)
            if not version:
                return False
            
            # Check for downstream dependencies
            downstream = self.get_downstream(version_id, max_depth=1)
            if downstream:
                raise ValueError(
                    f"Cannot delete version {version_id}: "
                    f"has downstream dependencies: {downstream}"
                )
            
            # Delete from database
            self._conn.execute(
                "DELETE FROM data_lineage WHERE output_version_id = ? OR input_version_id = ?",
                (version_id, version_id)
            )
            self._conn.execute(
                "DELETE FROM data_versions WHERE version_id = ?",
                (version_id,)
            )
            self._conn.commit()
            
            # Delete data file if requested
            if delete_data:
                storage_path = Path(version.storage_path)
                if storage_path.exists():
                    storage_path.unlink()
            
            return True
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        with self._lock:
            cursor = self._conn.execute("""
                SELECT 
                    COUNT(*) as total_versions,
                    COUNT(DISTINCT dataset_name) as total_datasets,
                    SUM(num_records) as total_records,
                    AVG(quality_score) as avg_quality
                FROM data_versions
            """)
            row = cursor.fetchone()
            
            # Calculate storage size
            total_size = 0
            for path in self.storage_root.rglob("*.parquet"):
                total_size += path.stat().st_size
            
            return {
                "total_versions": row[0] or 0,
                "total_datasets": row[1] or 0,
                "total_records": row[2] or 0,
                "avg_quality": row[3] or 0.0,
                "storage_size_mb": total_size / (1024 * 1024),
            }
    
    def _generate_version_id(self, dataset_name: str, data_type: str) -> str:
        """Generate unique semantic version ID."""
        cursor = self._conn.execute("""
            SELECT version_id FROM data_versions 
            WHERE dataset_name = ? AND data_type = ?
            ORDER BY created_at DESC LIMIT 1
        """, (dataset_name, data_type))
        
        row = cursor.fetchone()
        if not row:
            return "1.0.0"
        
        # Increment patch version
        latest = row[0]
        try:
            major, minor, patch = map(int, latest.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except (ValueError, AttributeError):
            # Fallback to UUID-based ID
            return _generate_id("dv")
    
    def _compute_quality_score(self, df: pd.DataFrame) -> float:
        """
        Compute data quality score (0.0 to 1.0).
        
        Factors:
        - Missing values (50% weight)
        - Duplicate rows (30% weight)
        - Outliers (20% weight)
        """
        score = 1.0
        
        if len(df) == 0:
            return 0.0
        
        # Penalize missing values
        total_cells = len(df) * len(df.columns)
        if total_cells > 0:
            missing_ratio = df.isnull().sum().sum() / total_cells
            score -= missing_ratio * 0.5
        
        # Penalize duplicate rows
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 0.3
        
        # Penalize outliers (>5 std devs) in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_count = 0
            total_numeric = 0
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0 and col_data.std() > 0:
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    outlier_count += (z_scores > 5).sum()
                    total_numeric += len(col_data)
            
            if total_numeric > 0:
                outlier_ratio = outlier_count / total_numeric
                score -= outlier_ratio * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _record_lineage(
        self,
        output_version_id: str,
        input_version_id: str,
        transformation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record lineage relationship in database."""
        self._conn.execute("""
            INSERT OR IGNORE INTO data_lineage (
                output_version_id, input_version_id, transformation, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            output_version_id,
            input_version_id,
            transformation,
            datetime.utcnow().isoformat(),
            json.dumps(metadata or {}),
        ))
    
    def _log_to_lineage_store(
        self,
        version: DataVersion,
        parent_version_id: Optional[str],
        transformation: Optional[str],
    ) -> None:
        """Log version creation to core LineageStore."""
        if not LINEAGE_AVAILABLE or not self.lineage_store:
            return
        
        try:
            # Determine event type based on data type
            if version.data_type == "raw":
                event_type = LineageEventType.DATA_INGESTION
            else:
                event_type = LineageEventType.DATA_TRANSFORMATION
            
            parent_ids = [parent_version_id] if parent_version_id else []
            
            event = LineageEvent(
                event_type=event_type,
                domain="trading.data",
                actor="DataVersionStore",
                artifact_id=version.version_id,
                parent_ids=parent_ids,
                payload={
                    "dataset_name": version.dataset_name,
                    "data_type": version.data_type,
                    "source": version.source,
                    "num_records": version.num_records,
                    "quality_score": version.quality_score,
                    "data_hash": version.data_hash,
                    "transformation": transformation,
                },
            )
            
            self.lineage_store.log_event(event)
        except Exception:
            pass  # Don't fail version creation if lineage logging fails
    
    def _row_to_version(self, row: sqlite3.Row) -> DataVersion:
        """Convert database row to DataVersion object."""
        return DataVersion(
            version_id=row["version_id"],
            dataset_name=row["dataset_name"],
            data_type=row["data_type"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            num_records=row["num_records"],
            start_date=datetime.fromisoformat(row["start_date"]),
            end_date=datetime.fromisoformat(row["end_date"]),
            schema_version=row["schema_version"],
            quality_score=row["quality_score"],
            storage_path=row["storage_path"],
            metadata=json.loads(row["metadata"]),
            parent_version_id=row["parent_version_id"],
            data_hash=row["data_hash"],
        )
