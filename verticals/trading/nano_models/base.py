"""
Base Nano Model for Trading Vertical

This module provides the base class for all trading nano models, ensuring
consistent interfaces, GLASSBOX/CDAI compliance, and lineage tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class ComplianceCategory(Enum):
    """Compliance categories for nano models."""
    
    GLASSBOX = "glassbox"  # Full transparency + audit trail
    CDAI = "cdai"  # Deterministic, non-probabilistic
    STANDARD = "standard"  # No restrictions


class ModelArchitecture(Enum):
    """Supported model architectures for trading nano models."""
    
    # GLASSBOX-compliant (full transparency)
    RULES_BASED = "rules_based"
    FORMULA = "formula"
    LINEAR = "linear"
    LOOKUP_TABLE = "lookup_table"
    STATE_MACHINE = "state_machine"
    SIGNAL_PROCESSING = "signal_processing"
    
    # CDAI-compliant (deterministic)
    DECISION_TREE = "decision_tree"
    
    # Standard (probabilistic allowed)
    NEURAL_NETWORK = "neural_network"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"


@dataclass
class NanoModelConfig:
    """Configuration for a trading nano model."""
    
    model_id: str
    name: str
    description: str
    architecture: ModelArchitecture
    compliance_category: ComplianceCategory
    version: str = "1.0.0"
    input_features: List[str] = field(default_factory=list)
    output_features: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result from a nano model prediction."""
    
    prediction: np.ndarray
    confidence: float
    timestamp: datetime
    model_id: str
    model_version: str
    input_hash: str  # Hash of input data for lineage tracking
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNanoModel(ABC):
    """
    Base class for all trading nano models.
    
    All trading nano models must inherit from this class and implement
    the required abstract methods. This ensures consistent interfaces
    across all models and enables GLASSBOX/CDAI compliance tracking.
    
    Example:
        class PricePredictionModel(BaseNanoModel):
            def __init__(self, config: NanoModelConfig):
                super().__init__(config)
                self._model = self._build_model()
            
            def predict(self, features: np.ndarray) -> PredictionResult:
                prediction = self._model.predict(features)
                return PredictionResult(
                    prediction=prediction,
                    confidence=self._calculate_confidence(prediction),
                    timestamp=datetime.utcnow(),
                    model_id=self.config.model_id,
                    model_version=self.config.version,
                    input_hash=self._hash_input(features)
                )
    """
    
    def __init__(self, config: NanoModelConfig):
        """
        Initialize the nano model with configuration.
        
        Args:
            config: Model configuration including architecture and compliance settings
        """
        self.config = config
        self._is_trained = False
        self._training_timestamp: Optional[datetime] = None
        self._lineage_id: Optional[str] = None
    
    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self.config.model_id
    
    @property
    def is_trained(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_trained
    
    @property
    def is_glassbox_compliant(self) -> bool:
        """Return whether the model is GLASSBOX compliant."""
        return self.config.compliance_category == ComplianceCategory.GLASSBOX
    
    @property
    def is_cdai_compliant(self) -> bool:
        """Return whether the model is CDAI compliant."""
        return self.config.compliance_category in [
            ComplianceCategory.GLASSBOX,
            ComplianceCategory.CDAI
        ]
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionResult:
        """
        Make a prediction using the trained model.
        
        Args:
            features: Input features as numpy array
            
        Returns:
            PredictionResult containing prediction and metadata
            
        Raises:
            ValueError: If model is not trained
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model on provided data.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def explain(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Explain the model's prediction for given features.
        
        Required for GLASSBOX compliance. Must provide human-readable
        explanation of how the prediction was made.
        
        Args:
            features: Input features to explain
            
        Returns:
            Dictionary containing explanation details
        """
        pass
    
    def validate_compliance(self) -> bool:
        """
        Validate that the model meets its compliance requirements.
        
        Returns:
            True if model meets compliance requirements
            
        Raises:
            ValueError: If model violates compliance requirements
        """
        # GLASSBOX models must be fully transparent
        if self.config.compliance_category == ComplianceCategory.GLASSBOX:
            glassbox_architectures = {
                ModelArchitecture.RULES_BASED,
                ModelArchitecture.FORMULA,
                ModelArchitecture.LINEAR,
                ModelArchitecture.LOOKUP_TABLE,
                ModelArchitecture.STATE_MACHINE,
                ModelArchitecture.SIGNAL_PROCESSING,
            }
            if self.config.architecture not in glassbox_architectures:
                raise ValueError(
                    f"Architecture {self.config.architecture} is not GLASSBOX compliant. "
                    f"Use one of: {glassbox_architectures}"
                )
        
        # CDAI models must be deterministic
        if self.config.compliance_category == ComplianceCategory.CDAI:
            cdai_architectures = {
                ModelArchitecture.RULES_BASED,
                ModelArchitecture.FORMULA,
                ModelArchitecture.LINEAR,
                ModelArchitecture.LOOKUP_TABLE,
                ModelArchitecture.STATE_MACHINE,
                ModelArchitecture.SIGNAL_PROCESSING,
                ModelArchitecture.DECISION_TREE,
            }
            if self.config.architecture not in cdai_architectures:
                raise ValueError(
                    f"Architecture {self.config.architecture} is not CDAI compliant. "
                    f"Use one of: {cdai_architectures}"
                )
        
        return True
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    @classmethod
    def load(cls, path: str) -> "BaseNanoModel":
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def _hash_input(self, features: np.ndarray) -> str:
        """Generate a hash of input features for lineage tracking."""
        import hashlib
        return hashlib.sha256(features.tobytes()).hexdigest()[:16]
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id='{self.config.model_id}', "
            f"architecture={self.config.architecture.value}, "
            f"trained={self._is_trained})"
        )
