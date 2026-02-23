"""
NM-TRADE-001: 1-Minute Price Prediction Nano Model

This nano model predicts price direction and magnitude for the next 1-minute candle.
Uses a linear model architecture for CDAI compliance.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from ..base import (
    BaseNanoModel,
    ComplianceCategory,
    ModelArchitecture,
    NanoModelConfig,
    PredictionResult,
)
from ..registry import register_model


# Default configuration for NM-TRADE-001
DEFAULT_CONFIG = NanoModelConfig(
    model_id="NM-TRADE-001",
    name="1-Minute Price Prediction",
    description="Predicts price direction and magnitude for the next 1-minute candle",
    architecture=ModelArchitecture.LINEAR,
    compliance_category=ComplianceCategory.CDAI,
    version="1.0.0",
    input_features=[
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_position",
        "atr_14",
        "returns_1",
        "returns_5",
        "returns_10",
    ],
    output_features=["direction", "magnitude"],
    hyperparameters={
        "regularization": 0.01,
        "normalize_inputs": True,
    },
    metadata={
        "timeframe": "1m",
        "asset_classes": ["crypto", "equities"],
    },
)


@register_model("NM-TRADE-001", config=DEFAULT_CONFIG)
class OneMinutePricePredictor(BaseNanoModel):
    """
    1-Minute Price Prediction Model (NM-TRADE-001)
    
    This model uses a linear regression approach to predict price movements
    for the next 1-minute candle. It is CDAI-compliant, meaning it produces
    deterministic outputs and supports full lineage tracking.
    
    Input Features:
        - close: Current close price (normalized)
        - volume: Current volume (normalized)
        - rsi_14: 14-period RSI
        - macd: MACD line value
        - macd_signal: MACD signal line value
        - bb_position: Position within Bollinger Bands (-1 to 1)
        - atr_14: 14-period ATR (normalized)
        - returns_1: 1-period return
        - returns_5: 5-period return
        - returns_10: 10-period return
    
    Output:
        - direction: Predicted price direction (-1, 0, 1)
        - magnitude: Predicted price change magnitude (normalized)
    
    Example:
        model = OneMinutePricePredictor(DEFAULT_CONFIG)
        model.train(X_train, y_train)
        
        result = model.predict(features)
        print(f"Direction: {result.prediction[0]}")
        print(f"Magnitude: {result.prediction[1]}")
        print(f"Confidence: {result.confidence}")
    """
    
    def __init__(self, config: NanoModelConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._input_mean: np.ndarray | None = None
        self._input_std: np.ndarray | None = None
        self._direction_threshold = 0.001  # 0.1% threshold for direction
    
    def predict(self, features: np.ndarray) -> PredictionResult:
        """
        Predict price direction and magnitude.
        
        Args:
            features: Input features as numpy array of shape (n_samples, n_features)
            
        Returns:
            PredictionResult with direction and magnitude predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize inputs
        if self.config.hyperparameters.get("normalize_inputs", True):
            features = (features - self._input_mean) / (self._input_std + 1e-8)
        
        # Linear prediction
        raw_prediction = features @ self._weights + self._bias
        
        # Extract direction and magnitude
        magnitude = raw_prediction[:, 0]
        direction = np.where(
            magnitude > self._direction_threshold, 1,
            np.where(magnitude < -self._direction_threshold, -1, 0)
        )
        
        # Calculate confidence based on prediction magnitude
        confidence = float(np.clip(np.abs(magnitude).mean() * 10, 0, 1))
        
        prediction = np.column_stack([direction, magnitude])
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            model_id=self.config.model_id,
            model_version=self.config.version,
            input_hash=self._hash_input(features),
            metadata={
                "direction_threshold": self._direction_threshold,
                "raw_magnitude": float(magnitude.mean()),
            }
        )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the linear model using ordinary least squares with regularization.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples, 1) - price returns
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary of training metrics
        """
        # Validate compliance
        self.validate_compliance()
        
        # Split data
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize inputs
        self._input_mean = X_train.mean(axis=0)
        self._input_std = X_train.std(axis=0)
        
        if self.config.hyperparameters.get("normalize_inputs", True):
            X_train_norm = (X_train - self._input_mean) / (self._input_std + 1e-8)
            X_val_norm = (X_val - self._input_mean) / (self._input_std + 1e-8)
        else:
            X_train_norm = X_train
            X_val_norm = X_val
        
        # Ensure y is 2D
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
        
        # Ridge regression (OLS with L2 regularization)
        reg = self.config.hyperparameters.get("regularization", 0.01)
        n_features = X_train_norm.shape[1]
        
        # Closed-form solution: w = (X'X + λI)^(-1) X'y
        XtX = X_train_norm.T @ X_train_norm
        XtX_reg = XtX + reg * np.eye(n_features)
        Xty = X_train_norm.T @ y_train
        
        self._weights = np.linalg.solve(XtX_reg, Xty)
        self._bias = y_train.mean(axis=0) - X_train_norm.mean(axis=0) @ self._weights
        
        # Calculate metrics
        train_pred = X_train_norm @ self._weights + self._bias
        val_pred = X_val_norm @ self._weights + self._bias
        
        train_mse = float(np.mean((train_pred - y_train) ** 2))
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        
        # Direction accuracy
        train_dir_acc = float(np.mean(
            np.sign(train_pred) == np.sign(y_train)
        ))
        val_dir_acc = float(np.mean(
            np.sign(val_pred) == np.sign(y_val)
        ))
        
        self._is_trained = True
        self._training_timestamp = datetime.utcnow()
        
        return {
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_direction_accuracy": train_dir_acc,
            "val_direction_accuracy": val_dir_acc,
            "n_train_samples": len(train_idx),
            "n_val_samples": len(val_idx),
        }
    
    def explain(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Explain the model's prediction for given features.
        
        Provides feature importance and contribution breakdown for
        GLASSBOX/CDAI compliance.
        
        Args:
            features: Input features to explain
            
        Returns:
            Dictionary containing explanation details
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before explanation")
        
        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize inputs
        if self.config.hyperparameters.get("normalize_inputs", True):
            features_norm = (features - self._input_mean) / (self._input_std + 1e-8)
        else:
            features_norm = features
        
        # Calculate feature contributions
        contributions = features_norm * self._weights.T
        
        # Feature importance (absolute weight magnitude)
        importance = np.abs(self._weights.flatten())
        importance_normalized = importance / importance.sum()
        
        # Create feature importance dict
        feature_importance = {
            name: float(imp)
            for name, imp in zip(self.config.input_features, importance_normalized)
        }
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Feature contributions for this prediction
        feature_contributions = {
            name: float(contrib)
            for name, contrib in zip(
                self.config.input_features,
                contributions.flatten()
            )
        }
        
        return {
            "model_type": "linear_regression",
            "feature_importance": feature_importance,
            "feature_contributions": feature_contributions,
            "bias": float(self._bias[0]) if self._bias is not None else 0.0,
            "decision_rule": (
                f"prediction = sum(feature * weight) + bias, "
                f"direction = sign(prediction) if |prediction| > {self._direction_threshold}"
            ),
            "compliance": {
                "category": self.config.compliance_category.value,
                "architecture": self.config.architecture.value,
                "deterministic": True,
                "explainable": True,
            }
        }
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        import json
        
        model_data = {
            "config": {
                "model_id": self.config.model_id,
                "version": self.config.version,
            },
            "weights": self._weights.tolist() if self._weights is not None else None,
            "bias": self._bias.tolist() if self._bias is not None else None,
            "input_mean": self._input_mean.tolist() if self._input_mean is not None else None,
            "input_std": self._input_std.tolist() if self._input_std is not None else None,
            "is_trained": self._is_trained,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
        }
        
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "OneMinutePricePredictor":
        """Load a model from disk."""
        import json
        
        with open(path, "r") as f:
            model_data = json.load(f)
        
        model = cls(DEFAULT_CONFIG)
        model._weights = np.array(model_data["weights"]) if model_data["weights"] else None
        model._bias = np.array(model_data["bias"]) if model_data["bias"] else None
        model._input_mean = np.array(model_data["input_mean"]) if model_data["input_mean"] else None
        model._input_std = np.array(model_data["input_std"]) if model_data["input_std"] else None
        model._is_trained = model_data["is_trained"]
        
        if model_data["training_timestamp"]:
            model._training_timestamp = datetime.fromisoformat(model_data["training_timestamp"])
        
        return model
