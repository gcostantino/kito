"""
Registry system for datasets and preprocessing.

Allows declarative configuration by registering classes with string names.

Example:
    @DATASETS.register('h5dataset')
    class H5Dataset:
        pass

    # Later
    dataset_cls = DATASETS.get('h5dataset')
    dataset = dataset_cls(path='data.h5')
"""


class Registry:
    """
    Simple registry for mapping string names to classes.

    Used for:
    - Dataset types ('h5dataset', 'memdataset')
    - Preprocessing types ('detrend', 'standardization')

    This enables config-based instantiation.
    """

    def __init__(self, name: str):
        """
        Initialize registry.

        Args:
            name: Registry name (for error messages)
        """
        self.name = name
        self._registry = {}

    def register(self, name: str):
        """
        Decorator to register a class.

        Args:
            name: String identifier for the class

        Returns:
            Decorator function

        Example:
            >>> @DATASETS.register('h5dataset')
            >>> class H5Dataset:
            ...     pass
        """

        def decorator(cls):
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self.name}. "
                    f"Existing: {self._registry[name]}, New: {cls}"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str):
        """
        Get a registered class by name.

        Args:
            name: String identifier

        Returns:
            Registered class

        Raises:
            KeyError: If name not registered
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_registered(self):
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str):
        """Check if name is registered."""
        return name in self._registry


# Global registries
DATASETS = Registry('DATASETS')
PREPROCESSING = Registry('PREPROCESSING')
