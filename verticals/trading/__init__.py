"""
Trading Vertical - ARYA Algorithmic Trading Systems

This vertical implements AARA's algorithmic trading capabilities, enabling
autonomous trading across crypto, equities, and forex markets through
multiple platform integrations.

Key Features:
- Multi-asset class trading (crypto, equities, forex, futures)
- Real-time market data streaming and processing
- Nano model-based price prediction and risk assessment
- Backtesting infrastructure with realistic execution simulation
- Platform integrations: QuantConnect, Interactive Brokers, Binance
- AARA Discovery Engine integration for strategy evolution
- Safety Kernel integration for risk enforcement

Architecture:
- nano_models/: Trading-specific nano models (price, volatility, regime, risk)
- strategies/: Trading strategy implementations and discovery-generated strategies
- data/: Data collection, preprocessing, and feature engineering pipelines
- integration/: Platform connectors (QuantConnect, IB, Binance)
- backtesting/: Strategy validation and performance analysis
- execution/: Live trading execution framework
- monitoring/: Real-time performance monitoring and alerting

Dependencies:
- ccxt: Cryptocurrency exchange API
- ib_insync: Interactive Brokers API wrapper
- pandas: Data manipulation
- numpy: Numerical computing
- ta: Technical analysis library

AARA Integration:
- Uses GLASSBOX-compliant nano models for safety-critical decisions
- Full lineage tracking for all models and data
- Safety Kernel integration for risk enforcement
- Discovery Engine integration for strategy evolution
"""

__version__ = "0.1.0"
