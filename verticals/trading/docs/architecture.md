# Trading Vertical Architecture

**Last Updated**: February 2026

## Overview

The Trading vertical follows a layered architecture that separates concerns between data collection, model inference, strategy execution, and platform integration.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Monitoring Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Metrics   │  │  Alerting   │  │  Dashboard  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Execution Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Executor   │──│  Position   │──│    Risk     │──► Safety    │
│  │             │  │  Manager    │  │  Enforcer   │   Kernel     │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Strategy Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Momentum   │  │    Mean     │  │  Discovery  │              │
│  │ Strategies  │  │  Reversion  │  │  Generated  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                               │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│  │   Price   │ │ Volatility│ │  Regime   │ │   Risk    │        │
│  │ Prediction│ │ Forecast  │ │  Class.   │ │Assessment │        │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│  │ Collectors│─│Preprocessor│─│ Feature   │─│  Storage  │        │
│  │           │ │           │ │Engineering│ │           │        │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Platform Layer                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │  QuantConnect │  │ Interactive   │  │    Binance    │        │
│  │               │  │   Brokers     │  │               │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### Data Layer

Responsible for collecting, cleaning, and storing market data.

- **Collectors**: Fetch data from exchanges and data providers
- **Preprocessors**: Clean, normalize, and validate data
- **Feature Engineering**: Compute technical indicators and statistical features
- **Storage**: Version and store data with lineage tracking

### Model Layer

Contains nano models for various prediction and classification tasks.

- **Price Prediction**: Forecast price movements at various timeframes
- **Volatility Forecast**: Predict realized volatility
- **Regime Classification**: Identify market regimes
- **Risk Assessment**: Estimate portfolio risk metrics

### Strategy Layer

Implements trading strategies that consume model outputs.

- **Momentum**: Trend-following strategies
- **Mean Reversion**: Statistical arbitrage strategies
- **Discovery Generated**: Strategies evolved by AARA Discovery Engine

### Execution Layer

Manages live trading execution with safety controls.

- **Executor**: Orchestrates signal generation and order submission
- **Position Manager**: Tracks and manages open positions
- **Risk Enforcer**: Validates trades against risk limits

### Platform Layer

Provides unified interfaces to trading platforms.

- **QuantConnect**: Cloud-based algorithmic trading
- **Interactive Brokers**: Multi-asset brokerage
- **Binance**: Cryptocurrency exchange

### Monitoring Layer

Real-time performance monitoring and alerting.

- **Metrics**: Collect and expose performance metrics
- **Alerting**: Send alerts on threshold breaches
- **Dashboard**: Web-based monitoring interface

## AARA Integration

The Trading vertical integrates with core AARA components:

### Discovery Engine

- Generates new trading strategies through evolutionary optimization
- Validates strategies via backtesting before deployment
- Maintains lineage tracking for all generated strategies

### Safety Kernel

- Validates all trades before execution
- Enforces risk limits and position constraints
- Provides emergency stop functionality

### Lineage Store

- Tracks data-to-model relationships
- Enables data removal compliance (GDPR, etc.)
- Supports model retraining on data changes

## Data Flow

1. **Data Collection**: Market data collected from exchanges
2. **Preprocessing**: Data cleaned and normalized
3. **Feature Engineering**: Technical indicators computed
4. **Model Inference**: Nano models generate predictions
5. **Signal Generation**: Strategies generate trading signals
6. **Risk Validation**: Signals validated against risk limits
7. **Order Execution**: Orders submitted to trading platforms
8. **Position Tracking**: Positions updated on fills
9. **Monitoring**: Performance metrics collected and displayed

## Configuration

Configuration is environment-specific:

- `development.yaml`: Local development with testnet APIs
- `paper_trading.yaml`: Paper trading with real market data
- `production.yaml`: Live trading with real money

See [Deployment Guide](deployment.md) for configuration details.
