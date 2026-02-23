"""
Trading Nano Models

This package contains all trading-specific nano models organized by category:

- price_prediction/: NM-TRADE-001 to 005 - Forecast price movements at various timeframes
- volatility_forecast/: NM-TRADE-006 to 008 - Predict realized volatility
- regime_classification/: NM-TRADE-009 to 010 - Identify market regimes
- execution_optimization/: NM-TRADE-011 to 012 - Optimize order execution
- risk_assessment/: NM-TRADE-013 to 015 - Estimate portfolio risk metrics

All nano models follow GLASSBOX/CDAI compliance requirements for transparency
and auditability. Safety-critical risk decisions use deterministic models.

Usage:
    from verticals.trading.nano_models import registry
    
    # Get a registered nano model
    model = registry.get_model("NM-TRADE-001")
    
    # List all available models
    models = registry.list_models()
"""

from .base import BaseNanoModel, NanoModelConfig

__all__ = ["BaseNanoModel", "NanoModelConfig"]
