"""
Loss utility functions with automatic discovery.

Users can simply place custom losses in losses/ directory,
and they will be automatically discovered!
"""
import importlib.util
import sys
from pathlib import Path
from typing import Union

import torch.nn as nn

from .registry import LossRegistry

_discovery_cache = set()  # Track what we've already searched


def discover_losses(directory: Union[str, Path] = './losses', verbose: bool = True):
    """
    Discover and register all losses in a directory.

    Searches for Python files with @LossRegistry.register() decorators.
    Can be called explicitly or happens automatically when loss is not found.

    Args:
        directory: Directory containing custom loss files (default: './losses')
        verbose: Print discovery messages (default: True)

    Returns:
        Number of new losses discovered
    """
    losses_dir = Path(directory)

    # Skip if already searched
    if str(losses_dir.resolve()) in _discovery_cache:
        return 0

    _discovery_cache.add(str(losses_dir.resolve()))

    if not losses_dir.exists():
        if verbose:
            print(f"Loss directory '{directory}' not found, skipping")
        return 0

    initial_count = len(LossRegistry.list_available())

    # Get package name from directory
    package_name = losses_dir.name

    # import individual .py files (don't rely on __init__.py, otherwise decorators do not get triggered)
    loaded = 0
    errors = []

    for py_file in sorted(losses_dir.glob('*.py')):
        # Skip __init__.py and private files
        if py_file.name.startswith('_'):
            continue

        module_name = f"{package_name}.{py_file.stem}"

        # Skip if already imported
        if module_name in sys.modules:
            continue

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                loaded += 1
        except Exception as e:
            errors.append((py_file.name, str(e)))
            if verbose:
                print(f"Warning: Could not load {py_file.name}: {e}")

    # Calculate how many losses were discovered
    new_count = len(LossRegistry.list_available()) - initial_count

    if verbose:
        if new_count > 0:
            print(f"Discovered {new_count} loss(es) from {directory}/ ({loaded} files loaded)")
        elif loaded > 0:
            print(f"Loaded {loaded} file(s) from {directory}/ but no losses registered")
            print(f"Make sure files use @LossRegistry.register('name') decorator")
        else:
            print(f"No Python files found in {directory}/")

    return new_count

def _auto_discover_loss(loss_name: str, verbose: bool = False) -> bool:
    """
    Try to auto-discover a specific loss from common directories.

    Called automatically when a loss is not found in registry.
    Searches in order:
    1. ./losses/
    2. ./custom_losses/
    3. ./lib/losses/

    Args:
        loss_name: Name of the loss to find
        verbose: Print discovery messages

    Returns:
        True if loss was found and registered, False otherwise
    """
    # Common directories to search
    search_dirs = [
        './losses',
        './custom_losses',
        './lib/losses',
    ]

    for directory in search_dirs:
        initial_registered = LossRegistry.is_registered(loss_name)

        if initial_registered:
            return True

        # Try to discover from this directory
        discover_losses(directory, verbose=False)

        # Check if loss is now registered
        if LossRegistry.is_registered(loss_name):
            if verbose:
                print(f"Auto-discovered '{loss_name}' from {directory}/")
            return True

    return False


def get_loss(loss_config):
    """
    Get loss function from config (backward compatible with old code).

    Supports:
    1. Simple string: 'mse'
    2. Dict with name: {'name': 'mse'}
    3. Dict with params: {'name': 'weighted_mse', 'params': {'weight': 2.0}}
    4. Dict with inline params: {'name': 'weighted_mse', 'weight': 2.0}
    5. Already instantiated loss: nn.Module instance.

    If a loss name is not found in the registry, automatically searches
    for it in ./losses/, ./custom_losses/, and ./lib/losses/ directories.

    This means users can simply create losses/my_loss.py with
    @LossRegistry.register('my_loss') and use it automatically.

    Args:
        loss_config: Loss configuration (string, dict or nn.Module instance).

    Returns:
        Loss function instance
    Raises:
        ValueError: If loss not found even after auto-discovery
        TypeError: If invalid config type

    Examples:
        # Simple string
        loss = get_loss('mse')

        # Dict with name
        loss = get_loss({'name': 'mse'})

        # Dict with params (new style)
        loss = get_loss({'name': 'weighted_mse', 'params': {'weight': 2.0}})

        # Dict with inline params (old style - still supported)
        loss = get_loss({'name': 'weighted_mse', 'weight': 2.0})

        # Already instantiated (passes through)
        from kito.losses import LossRegistry
        loss_instance = LossRegistry.create('mse')
        loss = get_loss(loss_instance)  # Returns same instance
    """
    # Handle already-instantiated loss
    if isinstance(loss_config, nn.Module):
        return loss_config

    # Extract loss name and params
    loss_name = None
    params = {}

    if isinstance(loss_config, str):
        loss_name = loss_config
        params = {}

    elif isinstance(loss_config, dict):
        if 'name' not in loss_config:
            raise ValueError(
                f"Loss config dict must have 'name' key. Got: {loss_config}"
            )

        loss_name = loss_config['name']

        # Check for 'params' key (new style)
        if 'params' in loss_config:
            params = loss_config['params']
        else:
            # Inline params (old style): extract all keys except 'name'
            params = {k: v for k, v in loss_config.items() if k != 'name'}

    else:
        raise TypeError(
            f"Invalid loss config type: {type(loss_config)}. "
            f"Expected string, dict, or nn.Module instance. "
            f"Got: {loss_config}"
        )

    # Try to create loss
    if not LossRegistry.is_registered(loss_name):
        # auto-discovery: Try to find and load the loss
        discovered = _auto_discover_loss(loss_name, verbose=True)

        if not discovered:
            # Still not found - give helpful error
            available = LossRegistry.list_available()
            available_str = ", ".join(sorted(available)[:15])
            if len(available) > 15:
                available_str += "..."

            raise ValueError(
                f"Loss '{loss_name}' not found in registry.\n"
                f"\n"
                f"Available losses: {available_str}\n"
                f"\n"
                f"Auto-discovery searched: ./losses/, ./custom_losses/, ./lib/losses/\n"
                f"\n"
                f"To add a custom loss:\n"
                f"1. Create losses/{loss_name}.py\n"
                f"2. Use @LossRegistry.register('{loss_name}') decorator\n"
                f"3. Use in config: loss: {{name: '{loss_name}', params: {{...}}}}\n"
                f"\n"
                f"Or import manually: import losses.{loss_name}\n"
                f"Or call explicitly: discover_losses('./losses')"
            )

    # Create and return loss instance
    try:
        return LossRegistry.create(loss_name, **params)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create loss '{loss_name}' with params {params}.\n"
            f"Error: {e}"
        ) from e
