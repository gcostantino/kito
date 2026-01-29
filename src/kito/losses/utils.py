"""
Backward compatible utility functions.

Wraps the old get_loss() function to work with the new registry.
"""
from .registry import LossRegistry


def get_loss(loss_config):
    """
    Get loss function from config (backward compatible with old code).

    Supports:
    1. Simple string: 'mse'
    2. Dict with name: {'name': 'mse'}
    3. Dict with params: {'name': 'weighted_mse', 'params': {'weight': 2.0}}
    4. Dict with inline params: {'name': 'weighted_mse', 'weight': 2.0}

    Args:
        loss_config: Loss configuration (string or dict)

    Returns:
        Loss function instance

    Examples:
        # Simple string
        loss = get_loss('mse')

        # Dict with name
        loss = get_loss({'name': 'mse'})

        # Dict with params (new style)
        loss = get_loss({'name': 'weighted_mse', 'params': {'weight': 2.0}})

        # Dict with inline params (old style - still supported)
        loss = get_loss({'name': 'weighted_mse', 'weight': 2.0})
    """
    # Case 1: Simple string
    if isinstance(loss_config, str):
        return LossRegistry.create(loss_config)

    # Case 2 & 3: Dict
    elif isinstance(loss_config, dict):
        if 'name' not in loss_config:
            raise ValueError(
                f"Loss config dict must have 'name' key. Got: {loss_config}"
            )

        name = loss_config['name']

        # Check if params are nested under 'params' key (new style)
        if 'params' in loss_config:
            params = loss_config['params']
        else:
            # Old style: params are inline (extract all keys except 'name')
            params = {k: v for k, v in loss_config.items() if k != 'name'}

        return LossRegistry.create(name, **params)

    else:
        raise TypeError(
            f"Invalid loss config type: {type(loss_config)}. "
            f"Expected string or dict, got: {loss_config}"
        )
