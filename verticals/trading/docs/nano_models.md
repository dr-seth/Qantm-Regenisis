# Trading Nano Models Guide

**Last Updated**: February 2026

## Overview

Trading nano models are specialized, lightweight models designed for specific prediction tasks in the trading domain. They follow GLASSBOX/CDAI compliance requirements for transparency and auditability.

## Model Categories

### Price Prediction (NM-TRADE-001 to 005)

Forecast price movements at various timeframes.

| Model ID | Timeframe | Input Features | Output |
|----------|-----------|----------------|--------|
| NM-TRADE-001 | 1 minute | OHLCV, technical indicators | Price direction + magnitude |
| NM-TRADE-002 | 5 minutes | OHLCV, technical indicators | Price direction + magnitude |
| NM-TRADE-003 | 15 minutes | OHLCV, technical indicators | Price direction + magnitude |
| NM-TRADE-004 | 1 hour | OHLCV, technical indicators | Price direction + magnitude |
| NM-TRADE-005 | 4 hours | OHLCV, technical indicators | Price direction + magnitude |

### Volatility Forecast (NM-TRADE-006 to 008)

Predict realized volatility for position sizing and risk management.

| Model ID | Timeframe | Input Features | Output |
|----------|-----------|----------------|--------|
| NM-TRADE-006 | 1 hour | Historical volatility, volume | Realized volatility |
| NM-TRADE-007 | 4 hours | Historical volatility, volume | Realized volatility |
| NM-TRADE-008 | 1 day | Historical volatility, volume | Realized volatility |

### Regime Classification (NM-TRADE-009 to 010)

Identify market regimes for strategy selection.

| Model ID | Timeframe | Regimes |
|----------|-----------|---------|
| NM-TRADE-009 | Short-term | Trending Up, Trending Down, Ranging, High Volatility |
| NM-TRADE-010 | Medium-term | Bull Market, Bear Market, Consolidation |

### Execution Optimization (NM-TRADE-011 to 012)

Optimize order execution to minimize slippage and market impact.

| Model ID | Asset Class | Optimization Target |
|----------|-------------|---------------------|
| NM-TRADE-011 | Crypto | Optimal order size, timing |
| NM-TRADE-012 | Equities | Optimal order size, timing |

### Risk Assessment (NM-TRADE-013 to 015)

Estimate portfolio risk metrics. These models use GLASSBOX-compliant architectures.

| Model ID | Metric | Compliance |
|----------|--------|------------|
| NM-TRADE-013 | Value at Risk (VaR) | GLASSBOX |
| NM-TRADE-014 | Expected Shortfall (CVaR) | GLASSBOX |
| NM-TRADE-015 | Drawdown Probability | GLASSBOX |

## Creating a New Nano Model

### 1. Define the Model Class

```python
from verticals.trading.nano_models.base import (
    BaseNanoModel,
    NanoModelConfig,
    PredictionResult,
    ModelArchitecture,
    ComplianceCategory
)
import numpy as np

class MyPriceModel(BaseNanoModel):
    """Custom price prediction model."""
    
    def __init__(self, config: NanoModelConfig):
        super().__init__(config)
        self._model = None
    
    def predict(self, features: np.ndarray) -> PredictionResult:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        prediction = self._model.predict(features)
        
        return PredictionResult(
            prediction=prediction,
            confidence=self._calculate_confidence(prediction),
            timestamp=datetime.utcnow(),
            model_id=self.config.model_id,
            model_version=self.config.version,
            input_hash=self._hash_input(features)
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        # Training implementation
        self._is_trained = True
        return {"accuracy": 0.65, "loss": 0.35}
    
    def explain(self, features: np.ndarray):
        # Required for GLASSBOX compliance
        return {
            "feature_importance": {...},
            "decision_path": [...],
            "confidence_factors": [...]
        }
```

### 2. Register the Model

```python
from verticals.trading.nano_models.registry import register_model

config = NanoModelConfig(
    model_id="NM-TRADE-NEW",
    name="My Price Model",
    description="Custom price prediction model",
    architecture=ModelArchitecture.LINEAR,
    compliance_category=ComplianceCategory.CDAI,
    input_features=["close", "volume", "rsi", "macd"],
    output_features=["price_direction", "magnitude"]
)

@register_model("NM-TRADE-NEW", config=config)
class MyPriceModel(BaseNanoModel):
    ...
```

### 3. Add Tests

Create tests in the appropriate `tests/` directory:

```python
# nano_models/price_prediction/tests/test_my_model.py

import pytest
import numpy as np
from verticals.trading.nano_models.price_prediction.my_model import MyPriceModel

def test_model_prediction():
    model = MyPriceModel(config)
    # Train with sample data
    X = np.random.randn(100, 4)
    y = np.random.randn(100, 2)
    model.train(X, y)
    
    # Test prediction
    features = np.random.randn(1, 4)
    result = model.predict(features)
    
    assert result.prediction is not None
    assert 0 <= result.confidence <= 1
```

## Compliance Requirements

### GLASSBOX Compliance

For safety-critical models (risk assessment):

- Must use transparent architectures (rules, formulas, linear)
- Must implement `explain()` method
- Must provide complete audit trails
- Must be deterministic

### CDAI Compliance

For all trading models:

- Must be deterministic (same input → same output)
- Must track data lineage
- Must support data removal

## Training Pipeline

1. **Data Preparation**: Load and preprocess training data
2. **Feature Engineering**: Compute required features
3. **Training**: Train model with validation split
4. **Evaluation**: Evaluate on held-out test set
5. **Registration**: Register trained model with lineage
6. **Deployment**: Deploy to production

Use the training script:

```bash
python scripts/train_nano_models.py --model NM-TRADE-001 --data-path ./data
```

## Model Versioning

Models are versioned using semantic versioning:

- **Major**: Breaking changes to input/output format
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, no feature changes

All versions are tracked in the model registry with full lineage.
