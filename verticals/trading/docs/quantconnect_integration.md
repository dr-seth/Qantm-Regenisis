# QuantConnect API Integration

**Last Updated:** 2026-02-23

## Overview

This module provides full integration with the QuantConnect API v2 for algorithmic trading operations. It supports:

- **Authentication**: SHA-256 hashed token authentication
- **Project Management**: Create, list, and manage algorithm projects
- **Backtesting**: Run and monitor backtests
- **Live Trading**: Deploy, monitor, and control live algorithms
- **Data Access**: Download historical market data

## Quick Start

### Environment Setup

Set the following environment variables:

```bash
export QC_USER_ID=462895
export QC_API_TOKEN=your_api_token_here
export QC_ORGANIZATION_ID=6158162de34ee050731d19d95b86e4de  # Optional
```

### Basic Usage

```python
from verticals.trading.integration.quantconnect import (
    QuantConnectClient,
    QuantConnectConnector,
)

# Using the low-level client
client = QuantConnectClient()

# Verify authentication
if client.authenticate():
    print("✓ Connected to QuantConnect")
    
    # List projects
    projects = client.list_projects()
    for project in projects:
        print(f"  - {project.name} (ID: {project.project_id})")

# Using the high-level connector
connector = QuantConnectConnector()
await connector.connect()

# Run a backtest
result = await connector.run_backtest(
    project_id=12345,
    name="RSI Strategy Test"
)
print(f"Sharpe Ratio: {result.sharpe_ratio}")
```

## Architecture

```
quantconnect/
├── __init__.py          # Public exports
├── auth.py              # SHA-256 authentication
├── client.py            # API v2 client
├── connector.py         # High-level connector
├── models.py            # Pydantic models
└── tests/
    ├── test_auth.py     # Auth tests
    └── test_client.py   # Client tests
```

## Authentication

QuantConnect uses timestamped SHA-256 hashed tokens for authentication. Each request includes:

1. **Authorization Header**: Base64-encoded `user_id:sha256(api_token:timestamp)`
2. **Timestamp Header**: Unix timestamp used in the hash

```python
from verticals.trading.integration.quantconnect import QuantConnectAuth

auth = QuantConnectAuth(user_id=462895, api_token="your_token")
headers = auth.get_headers()

# Headers structure:
# {
#     'Authorization': 'Basic <base64_encoded_auth>',
#     'Timestamp': '1708700000'
# }
```

## API Client

The `QuantConnectClient` provides methods for all API endpoints:

### Projects

```python
# List all projects
projects = client.list_projects()

# Get specific project
project = client.get_project(project_id=12345)

# Create new project
response = client.create_project(name="My Strategy", language="Python")

# Delete project
response = client.delete_project(project_id=12345)
```

### Files

```python
# List files in project
response = client.list_files(project_id=12345)

# Get file contents
response = client.get_file(project_id=12345, file_name="main.py")

# Create/update file
response = client.create_file(
    project_id=12345,
    name="main.py",
    content="class MyAlgorithm(QCAlgorithm):..."
)
```

### Compilation

```python
# Compile project
response = client.compile_project(project_id=12345)
compile_id = response.data['compileId']

# Check compile status
status = client.get_compile_status(project_id=12345, compile_id=compile_id)
```

### Backtesting

```python
# Create backtest
response = client.create_backtest(
    project_id=12345,
    compile_id="compile-abc123",
    name="Test Run"
)

# Get backtest results
backtest = client.get_backtest(project_id=12345, backtest_id="bt-xyz")

# List all backtests
backtests = client.list_backtests(project_id=12345)
```

### Live Trading

```python
# Deploy live algorithm
response = client.create_live_algorithm(
    project_id=12345,
    compile_id="compile-abc123",
    node_id="L-MICRO",
    brokerage="InteractiveBrokers",
    brokerage_data={
        "account": "DU12345",
        "environment": "paper"
    }
)

# Get live status
status = client.get_live_algorithm(project_id=12345)

# Stop live algorithm
response = client.stop_live_algorithm(project_id=12345)

# Liquidate and stop
response = client.liquidate_live_algorithm(project_id=12345)
```

### Orders & Holdings

```python
# Get orders
orders = client.get_orders(project_id=12345)

# Get current holdings
holdings = client.get_holdings(project_id=12345)
```

## High-Level Connector

The `QuantConnectConnector` provides a simplified async interface:

```python
from verticals.trading.integration.quantconnect import QuantConnectConnector

connector = QuantConnectConnector()

# Connect and authenticate
await connector.connect()

# Run backtest with automatic compilation
result = await connector.run_backtest(
    project_id=12345,
    name="RSI Strategy v2",
    wait_for_completion=True
)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Total Trades: {result.total_trades}")

# Deploy live
deployment = await connector.deploy_live(
    project_id=12345,
    brokerage="Alpaca",
    brokerage_config={
        "key": "your_alpaca_key",
        "secret": "your_alpaca_secret",
        "paper": True
    }
)

# Event callbacks
def on_backtest_complete(result):
    print(f"Backtest {result.name} completed!")

connector.on('backtest_complete', on_backtest_complete)
```

## Data Models

### BacktestResult

```python
@dataclass
class BacktestResult:
    backtest_id: str
    name: str
    status: str
    sharpe_ratio: Optional[float]
    total_return: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    total_trades: int
    start_date: Optional[datetime]
    end_date: Optional[datetime]
```

### AlgorithmDeployment

```python
@dataclass
class AlgorithmDeployment:
    project_id: int
    deploy_id: str
    name: str
    status: str
    launched: datetime
    brokerage: str
```

## Configuration

Configuration in `configs/development.yaml`:

```yaml
platforms:
  quantconnect:
    enabled: true
    user_id_env: QC_USER_ID
    api_token_env: QC_API_TOKEN
    organization_id_env: QC_ORGANIZATION_ID
    base_url: https://www.quantconnect.com/api/v2
    timeout: 30
    default_language: Python
    auto_compile: true
```

## Error Handling

All API responses are wrapped in `QCResponse`:

```python
response = client.create_project("Test")

if response.success:
    project_id = response.data['projects'][0]['projectId']
else:
    print(f"Error: {response.errors}")
    print(f"Status Code: {response.status_code}")
```

Common error codes:
- **401**: Invalid or expired authentication
- **403**: Insufficient permissions
- **404**: Resource not found
- **429**: Rate limit exceeded
- **500**: Server error

## Testing

Run tests:

```bash
# Unit tests (no API calls)
pytest verticals/trading/integration/quantconnect/tests/ -v

# Integration tests (requires credentials)
pytest verticals/trading/integration/quantconnect/tests/ -v -m integration
```

## Related Issues

- **ARY-1083**: Real-time Data Feed Infrastructure
- **ARY-15**: AARA-QuantConnect Integration
- **ARY-20**: Paper Trading

## API Reference

Base URL: `https://www.quantconnect.com/api/v2`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/authenticate` | POST | Verify authentication |
| `/projects` | GET | List projects |
| `/projects` | POST | Create project |
| `/projects` | DELETE | Delete project |
| `/files` | GET | List/get files |
| `/files` | POST | Create file |
| `/files/update` | POST | Update file |
| `/compile` | POST | Compile project |
| `/backtests` | POST | Create backtest |
| `/backtests` | GET | Get backtest status |
| `/live` | POST | Deploy live |
| `/live` | GET | Get live status |
| `/live/stop` | POST | Stop live |
| `/live/liquidate` | POST | Liquidate |
| `/orders` | GET | Get orders |
| `/holdings` | GET | Get holdings |
