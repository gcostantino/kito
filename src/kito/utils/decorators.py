from functools import wraps


def require_mode(mode: str):
    """
    Decorator to enforce train_mode compatibility in Engine methods.

    Validates that the requested operation (train or inference) matches
    the config.training.train_mode setting.

    Args:
        mode: Either 'train' or 'inference'

    Raises:
        RuntimeError: If the operation conflicts with train_mode setting
        ValueError: If mode is not 'train' or 'inference'
    """
    if mode not in ('train', 'inference'):
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'.")

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get train_mode from config (default True for backward compatibility)
            config_mode = getattr(self.config.training, 'train_mode', True)

            # Check compatibility
            if mode == 'train' and not config_mode:
                raise RuntimeError(
                    f"Training not allowed: config.training.train_mode is set to False (inference mode).\n"
                    f"To enable training, set config.training.train_mode=True in your config."
                )

            if mode == 'inference' and config_mode:
                raise RuntimeError(
                    f"Inference not allowed: config.training.train_mode is set to True (training mode).\n"
                    f"To enable inference, set config.training.train_mode=False in your config."
                )

            # Mode is compatible - execute the original function
            return func(self, *args, **kwargs)

        return wrapper

    return decorator

def require_model_built(method):
    """Decorator to ensure the configuration is finalized before proceeding."""

    def wrapper(self, *args, **kwargs):
        if not self._model_built:
            raise RuntimeError("Model not built. Call module.build() before this operation.")
        return method(self, *args, **kwargs)
    return wrapper
