# Deployment Guide

**Last Updated**: February 2026

## Overview

This guide covers deploying trading strategies from development through live trading.

## Deployment Stages

```
Development → Backtesting → Paper Trading → Live Trading
```

### 1. Development

Local development with testnet APIs and simulated data.

**Configuration**: `configs/development.yaml`

```bash
# Run with development config
export TRADING_ENV=development
python scripts/backtest_strategy.py --strategy my_strategy ...
```

### 2. Backtesting

Validate strategy on historical data.

```bash
python scripts/backtest_strategy.py \
    --strategy simple_momentum \
    --symbols BTC/USDT,ETH/USDT \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --initial-capital 100000
```

**Validation Criteria**:
- Sharpe Ratio > 1.0
- Max Drawdown < 20%
- Win Rate > 40%
- Profit Factor > 1.5

### 3. Paper Trading

Test with real market data but simulated execution.

**Configuration**: `configs/paper_trading.yaml`

```bash
python scripts/deploy_live.py \
    --strategy simple_momentum \
    --mode paper \
    --platform binance
```

**Validation Period**: Minimum 2 weeks

**Success Criteria**:
- Performance matches backtest expectations
- No technical issues
- Risk limits respected

### 4. Live Trading

Deploy with real money.

**Configuration**: `configs/production.yaml`

```bash
python scripts/deploy_live.py \
    --strategy simple_momentum \
    --mode live \
    --platform binance \
    --confirm
```

## Configuration

### Development

```yaml
environment: development
data:
  use_testnet: true
platforms:
  binance:
    testnet: true
trading:
  risk:
    max_position_size: 0.01
monitoring:
  enabled: false
```

### Paper Trading

```yaml
environment: paper_trading
data:
  use_testnet: false
  realtime_enabled: true
platforms:
  binance:
    paper_trading: true
trading:
  risk:
    max_position_size: 0.05
monitoring:
  enabled: true
```

### Production

```yaml
environment: production
data:
  use_testnet: false
  realtime_enabled: true
platforms:
  binance:
    paper_trading: false
trading:
  risk:
    max_position_size: 0.02
    max_daily_loss: 0.02
  safety_kernel:
    enabled: true
monitoring:
  enabled: true
  alert_channels:
    - slack
    - pagerduty
```

## Safety Checklist

Before live deployment:

- [ ] Strategy passed backtesting validation
- [ ] Strategy passed paper trading validation (2+ weeks)
- [ ] Risk parameters configured appropriately
- [ ] Safety Kernel integration tested
- [ ] Monitoring and alerting configured
- [ ] Emergency stop procedures documented
- [ ] API keys secured and IP-whitelisted
- [ ] Backup and recovery procedures in place

## Monitoring

### Metrics

Key metrics to monitor:

- **PnL**: Real-time profit/loss
- **Positions**: Open positions and exposure
- **Orders**: Pending and filled orders
- **Latency**: Order execution latency
- **Errors**: API errors and failures

### Alerts

Configure alerts for:

- Daily loss exceeds threshold
- Drawdown exceeds threshold
- Connection lost
- High latency
- Model degradation

### Dashboard

Access the monitoring dashboard:

```bash
# Start dashboard
python -m verticals.trading.monitoring.dashboard

# Access at http://localhost:8080
```

## Emergency Procedures

### Emergency Stop

Immediately halt all trading:

```bash
# Via script
python scripts/emergency_stop.py --confirm

# Via Safety Kernel API
curl -X POST http://safety-kernel:8002/emergency_stop
```

### Manual Position Close

Close all positions:

```bash
python scripts/close_all_positions.py --confirm
```

### Rollback

Revert to previous strategy version:

```bash
python scripts/rollback_strategy.py --strategy simple_momentum --version v1.0.0
```

## Scaling

### Horizontal Scaling

Run multiple strategy instances:

```yaml
# docker-compose.yml
services:
  strategy-btc:
    image: trading-vertical
    environment:
      STRATEGY: simple_momentum
      SYMBOLS: BTC/USDT
  
  strategy-eth:
    image: trading-vertical
    environment:
      STRATEGY: simple_momentum
      SYMBOLS: ETH/USDT
```

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Data Collector | 1 | 512MB | 10GB |
| Strategy Engine | 2 | 2GB | 1GB |
| Monitoring | 1 | 1GB | 10GB |

## Maintenance

### Regular Tasks

- **Daily**: Review performance metrics
- **Weekly**: Check model accuracy, retrain if needed
- **Monthly**: Review and update risk parameters
- **Quarterly**: Full strategy review and optimization

### Updates

```bash
# Update trading vertical
git pull origin main
pip install -e ".[trading]"

# Restart services
docker-compose restart
```
