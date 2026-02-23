"""
QuantConnect Integration

AARA-QuantConnect bridge for algorithmic trading:
- auth.py: SHA-256 hashed token authentication
- client.py: Full API v2 client with all endpoints
- connector.py: High-level connector interface
- models.py: Pydantic request/response models

Reference: https://www.quantconnect.com/docs/v2/our-platform/api-reference
"""

from .auth import QuantConnectAuth
from .client import QuantConnectClient, QCResponse, QCEndpoint, Project, Backtest
from .connector import QuantConnectConnector, BacktestResult, AlgorithmDeployment, ConnectionState
from .models import (
    Language,
    BacktestStatus,
    LiveAlgorithmStatus,
    Resolution,
    OrderType,
    OrderStatus,
    CreateProjectRequest,
    CreateBacktestRequest,
    CreateLiveRequest,
)

__all__ = [
    # Auth
    'QuantConnectAuth',
    # Client
    'QuantConnectClient',
    'QCResponse',
    'QCEndpoint',
    'Project',
    'Backtest',
    # Connector
    'QuantConnectConnector',
    'BacktestResult',
    'AlgorithmDeployment',
    'ConnectionState',
    # Models
    'Language',
    'BacktestStatus',
    'LiveAlgorithmStatus',
    'Resolution',
    'OrderType',
    'OrderStatus',
    'CreateProjectRequest',
    'CreateBacktestRequest',
    'CreateLiveRequest',
]
