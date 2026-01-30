"""
Loss module for Kito framework.

Provides:
- Loss registry for automatic instantiation
- Built-in PyTorch losses
- Custom domain-specific losses
- Backward compatible get_loss() function
"""

# Import registry first
from .registry import LossRegistry

# Import built-in losses (triggers registration)
from . import builtin

# Import custom losses (triggers registration)
from . import custom

# Import utilities
from .utils import get_loss, discover_losses

# Public API
__all__ = [
    'LossRegistry',
    'get_loss',  # Backward compatible
    'discover_losses',   # Explicit discovery helpe
]

# Print available losses on import (optional - for debugging)
# print(f"Kito losses loaded. Available: {', '.join(LossRegistry.list_available())}")