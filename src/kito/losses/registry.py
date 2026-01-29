"""
Loss Registry System.

Centralized registry for automatic loss instantiation from config.
"""
from typing import Dict, Type

import torch.nn as nn


class LossRegistry:
    """
    Central registry for loss functions.

    Features:
    - Automatic instantiation from config
    - Support for custom user-defined losses
    - Built-in PyTorch losses
    - Backward compatible with existing code

    Example:
        # Register custom loss
        @LossRegistry.register('my_loss')
        class MyLoss(nn.Module):
            def __init__(self, weight=1.0):
                super().__init__()
                self.weight = weight

            def forward(self, pred, target):
                return self.weight * torch.mean((pred - target) ** 2)

        # Create from config
        loss = LossRegistry.create('my_loss', weight=2.0)

        # List available losses
        print(LossRegistry.list_available())
    """

    _registry: Dict[str, Type[nn.Module]] = {}
    _builtin_registered: bool = False

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a loss function.

        Args:
            name: Unique identifier for the loss (case-insensitive)

        Returns:
            Decorator function

        Example:
            @LossRegistry.register('weighted_mse')
            class WeightedMSE(nn.Module):
                def __init__(self, weight=1.0):
                    super().__init__()
                    self.weight = weight

                def forward(self, pred, target):
                    return self.weight * F.mse_loss(pred, target)
        """

        def decorator(loss_class: Type[nn.Module]):
            # Normalize name to lowercase for case-insensitive matching
            normalized_name = name.lower()

            if normalized_name in cls._registry:
                raise ValueError(
                    f"Loss '{name}' (normalized: '{normalized_name}') already registered!"
                )

            cls._registry[normalized_name] = loss_class
            return loss_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """
        Get a loss class by name (case-insensitive).

        Args:
            name: Name of registered loss

        Returns:
            Loss class

        Raises:
            ValueError: If loss not found
        """
        cls._ensure_builtin_registered()

        # Normalize to lowercase for matching
        normalized_name = name.lower()

        if normalized_name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Loss '{name}' not found in registry.\n"
                f"Available losses: {available}\n"
                f"Tip: Register custom losses with @LossRegistry.register('name')"
            )

        return cls._registry[normalized_name]

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """
        Create a loss instance from registry.

        Args:
            name: Name of registered loss (case-insensitive)
            **kwargs: Arguments to pass to loss constructor

        Returns:
            Instantiated loss function

        Example:
            # Simple instantiation
            loss = LossRegistry.create('mse')

            # With parameters
            loss = LossRegistry.create('weighted_mse', weight=2.0)

            # Combined losses
            loss = LossRegistry.create('combined', losses=[
                {'name': 'mse', 'weight': 1.0},
                {'name': 'l1', 'weight': 0.5}
            ])
        """
        loss_class = cls.get(name)

        try:
            return loss_class(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate loss '{name}' with parameters {kwargs}.\n"
                f"Error: {e}\n"
                f"Hint: Check the __init__ signature of {loss_class.__name__}"
            )

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a loss is registered (case-insensitive)."""
        cls._ensure_builtin_registered()
        return name.lower() in cls._registry

    @classmethod
    def list_available(cls) -> list:
        """List all registered loss names."""
        cls._ensure_builtin_registered()
        return sorted(cls._registry.keys())

    @classmethod
    def _ensure_builtin_registered(cls):
        """Lazy registration of built-in losses."""
        if cls._builtin_registered:
            return

        # Import here to avoid circular imports
        from . import builtin  # noqa: F401
        cls._builtin_registered = True
