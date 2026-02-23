"""
Strategy Registry for Trading Vertical

This module provides registration and discovery of trading strategies,
enabling consistent access patterns and lifecycle management.
"""

from typing import Dict, List, Optional, Type

from .base import BaseStrategy, StrategyConfig


class StrategyRegistry:
    """
    Registry for trading strategies.
    
    Provides centralized registration, discovery, and instantiation
    of trading strategies.
    
    Example:
        registry = StrategyRegistry()
        
        # Register a strategy class
        registry.register("simple_momentum", SimpleMomentumStrategy)
        
        # Get a strategy instance
        strategy = registry.get_strategy("simple_momentum", config)
        
        # List all registered strategies
        strategies = registry.list_strategies()
    """
    
    _instance: Optional["StrategyRegistry"] = None
    
    def __new__(cls) -> "StrategyRegistry":
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies: Dict[str, Type[BaseStrategy]] = {}
            cls._instance._configs: Dict[str, StrategyConfig] = {}
        return cls._instance
    
    def register(
        self,
        strategy_id: str,
        strategy_class: Type[BaseStrategy],
        config: Optional[StrategyConfig] = None
    ) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_class: The strategy class to register
            config: Optional default configuration for the strategy
        """
        if strategy_id in self._strategies:
            raise ValueError(f"Strategy {strategy_id} is already registered")
        
        self._strategies[strategy_id] = strategy_class
        if config:
            self._configs[strategy_id] = config
    
    def unregister(self, strategy_id: str) -> None:
        """
        Unregister a strategy.
        
        Args:
            strategy_id: The strategy identifier to unregister
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
        if strategy_id in self._configs:
            del self._configs[strategy_id]
    
    def get_strategy_class(self, strategy_id: str) -> Type[BaseStrategy]:
        """
        Get the strategy class for a given strategy ID.
        
        Args:
            strategy_id: The strategy identifier
            
        Returns:
            The registered strategy class
            
        Raises:
            KeyError: If strategy is not registered
        """
        if strategy_id not in self._strategies:
            raise KeyError(f"Strategy {strategy_id} is not registered")
        return self._strategies[strategy_id]
    
    def get_strategy(
        self,
        strategy_id: str,
        config: Optional[StrategyConfig] = None
    ) -> BaseStrategy:
        """
        Get an instance of a registered strategy.
        
        Args:
            strategy_id: The strategy identifier
            config: Optional configuration override
            
        Returns:
            An instance of the registered strategy
            
        Raises:
            KeyError: If strategy is not registered
            ValueError: If no config provided and no default config exists
        """
        strategy_class = self.get_strategy_class(strategy_id)
        
        if config is None:
            config = self._configs.get(strategy_id)
        
        if config is None:
            raise ValueError(
                f"No configuration provided for strategy {strategy_id} "
                "and no default configuration exists"
            )
        
        return strategy_class(config)
    
    def get_config(self, strategy_id: str) -> Optional[StrategyConfig]:
        """
        Get the default configuration for a strategy.
        
        Args:
            strategy_id: The strategy identifier
            
        Returns:
            The default configuration or None if not set
        """
        return self._configs.get(strategy_id)
    
    def list_strategies(self) -> List[str]:
        """
        List all registered strategy IDs.
        
        Returns:
            List of registered strategy identifiers
        """
        return list(self._strategies.keys())
    
    def list_strategies_by_type(self, strategy_type: str) -> List[str]:
        """
        List strategies by type prefix.
        
        Args:
            strategy_type: Type prefix (e.g., "momentum", "mean_reversion")
            
        Returns:
            List of strategy IDs matching the type
        """
        return [
            strategy_id for strategy_id in self._strategies.keys()
            if strategy_type.lower() in strategy_id.lower()
        ]
    
    def clear(self) -> None:
        """Clear all registered strategies. Use with caution."""
        self._strategies.clear()
        self._configs.clear()


# Global registry instance
registry = StrategyRegistry()


def register_strategy(
    strategy_id: str,
    config: Optional[StrategyConfig] = None
):
    """
    Decorator to register a strategy class.
    
    Example:
        @register_strategy("simple_momentum", config=my_config)
        class SimpleMomentumStrategy(BaseStrategy):
            ...
    """
    def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        registry.register(strategy_id, cls, config)
        return cls
    return decorator


def get_strategy(
    strategy_id: str,
    config: Optional[StrategyConfig] = None
) -> BaseStrategy:
    """
    Convenience function to get a strategy from the global registry.
    
    Args:
        strategy_id: The strategy identifier
        config: Optional configuration override
        
    Returns:
        An instance of the registered strategy
    """
    return registry.get_strategy(strategy_id, config)


def list_strategies() -> List[str]:
    """
    Convenience function to list all registered strategies.
    
    Returns:
        List of registered strategy identifiers
    """
    return registry.list_strategies()
