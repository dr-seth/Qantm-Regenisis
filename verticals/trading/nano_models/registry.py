"""
Nano Model Registry for Trading Vertical

This module provides registration and discovery of trading nano models,
enabling consistent access patterns and lineage tracking integration.
"""

from typing import Dict, List, Optional, Type

from .base import BaseNanoModel, NanoModelConfig


class NanoModelRegistry:
    """
    Registry for trading nano models.
    
    Provides centralized registration, discovery, and instantiation
    of nano models with lineage tracking support.
    
    Example:
        registry = NanoModelRegistry()
        
        # Register a model class
        registry.register("NM-TRADE-001", PricePrediction1Min)
        
        # Get a model instance
        model = registry.get_model("NM-TRADE-001")
        
        # List all registered models
        models = registry.list_models()
    """
    
    _instance: Optional["NanoModelRegistry"] = None
    
    def __new__(cls) -> "NanoModelRegistry":
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, Type[BaseNanoModel]] = {}
            cls._instance._configs: Dict[str, NanoModelConfig] = {}
        return cls._instance
    
    def register(
        self,
        model_id: str,
        model_class: Type[BaseNanoModel],
        config: Optional[NanoModelConfig] = None
    ) -> None:
        """
        Register a nano model class.
        
        Args:
            model_id: Unique identifier for the model (e.g., "NM-TRADE-001")
            model_class: The model class to register
            config: Optional default configuration for the model
        """
        if model_id in self._models:
            raise ValueError(f"Model {model_id} is already registered")
        
        self._models[model_id] = model_class
        if config:
            self._configs[model_id] = config
    
    def unregister(self, model_id: str) -> None:
        """
        Unregister a nano model.
        
        Args:
            model_id: The model identifier to unregister
        """
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._configs:
            del self._configs[model_id]
    
    def get_model_class(self, model_id: str) -> Type[BaseNanoModel]:
        """
        Get the model class for a given model ID.
        
        Args:
            model_id: The model identifier
            
        Returns:
            The registered model class
            
        Raises:
            KeyError: If model is not registered
        """
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} is not registered")
        return self._models[model_id]
    
    def get_model(
        self,
        model_id: str,
        config: Optional[NanoModelConfig] = None
    ) -> BaseNanoModel:
        """
        Get an instance of a registered model.
        
        Args:
            model_id: The model identifier
            config: Optional configuration override
            
        Returns:
            An instance of the registered model
            
        Raises:
            KeyError: If model is not registered
            ValueError: If no config provided and no default config exists
        """
        model_class = self.get_model_class(model_id)
        
        if config is None:
            config = self._configs.get(model_id)
        
        if config is None:
            raise ValueError(
                f"No configuration provided for model {model_id} "
                "and no default configuration exists"
            )
        
        return model_class(config)
    
    def get_config(self, model_id: str) -> Optional[NanoModelConfig]:
        """
        Get the default configuration for a model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            The default configuration or None if not set
        """
        return self._configs.get(model_id)
    
    def list_models(self) -> List[str]:
        """
        List all registered model IDs.
        
        Returns:
            List of registered model identifiers
        """
        return list(self._models.keys())
    
    def list_models_by_category(self, category: str) -> List[str]:
        """
        List models by category prefix.
        
        Args:
            category: Category prefix (e.g., "price_prediction", "volatility")
            
        Returns:
            List of model IDs matching the category
        """
        return [
            model_id for model_id in self._models.keys()
            if category.lower() in model_id.lower()
        ]
    
    def clear(self) -> None:
        """Clear all registered models. Use with caution."""
        self._models.clear()
        self._configs.clear()


# Global registry instance
registry = NanoModelRegistry()


def register_model(
    model_id: str,
    config: Optional[NanoModelConfig] = None
):
    """
    Decorator to register a nano model class.
    
    Example:
        @register_model("NM-TRADE-001", config=my_config)
        class PricePrediction1Min(BaseNanoModel):
            ...
    """
    def decorator(cls: Type[BaseNanoModel]) -> Type[BaseNanoModel]:
        registry.register(model_id, cls, config)
        return cls
    return decorator


def get_model(model_id: str, config: Optional[NanoModelConfig] = None) -> BaseNanoModel:
    """
    Convenience function to get a model from the global registry.
    
    Args:
        model_id: The model identifier
        config: Optional configuration override
        
    Returns:
        An instance of the registered model
    """
    return registry.get_model(model_id, config)


def list_models() -> List[str]:
    """
    Convenience function to list all registered models.
    
    Returns:
        List of registered model identifiers
    """
    return registry.list_models()
