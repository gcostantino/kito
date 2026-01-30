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
import importlib.util
import sys
from pathlib import Path


class Registry:
    """
    Simple registry for mapping string names to classes.

    Used for:
    - Dataset types ('h5dataset', 'memdataset')
    - Preprocessing types ('detrend', 'standardization')

    This enables config-based instantiation.
    """

    def __init__(self, name: str, discovery_dirs=None):
        """
        Initialize registry.

        Args:
            name: Registry name (for error messages)
            discovery_dirs: List of directories to search for auto-discovery
                           If None, derives from name (e.g., 'PREPROCESSING' -> './preprocessing'
        """
        self.name = name
        self._registry = {}
        self._discovery_attempted = False

        # Auto-determine discovery directories from registry name
        if discovery_dirs is None:
            # Convert 'PREPROCESSING' -> ['./preprocessing', './custom_preprocessing']
            base_name = name.lower().rstrip('s')  # PREPROCESSING -> preprocessing
            self.discovery_dirs = [
                f'./{base_name}',
                f'./custom_{base_name}',
            ]
        else:
            self.discovery_dirs = discovery_dirs

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
        # check if already registered
        if name in self._registry:
            return self._registry[name]

        # try auto-discovery if not found
        if not self._discovery_attempted:
            self._discover(verbose=False)

            # check again after discovery
            if name in self._registry:
                return self._registry[name]

        # still not found - raise error
        available = ", ".join(sorted(self._registry.keys()))
        raise KeyError(
            f"'{name}' not found in {self.name}.\n"
            f"Available: {available}\n"
            f"\n"
            f"Auto-discovery searched: {', '.join(self.discovery_dirs)}\n"
            f"\n"
            f"To add a custom {self.name.lower()}:\n"
            f"1. Create {self.discovery_dirs[0]}/{name}.py\n"
            f"2. Use @{self.name}.register('{name}') decorator\n"
            f"3. Use in config or code"
        )

    def _discover(self, verbose: bool = False):
        """
        Discover and register classes from discovery directories.

        Searches for Python files with @registry.register() decorators.
        Only runs once per registry instance.

        Args:
            verbose: Print discovery messages
        """
        if self._discovery_attempted:
            return

        self._discovery_attempted = True

        for directory in self.discovery_dirs:
            dir_path = Path(directory)

            if not dir_path.exists():
                continue

            package_name = dir_path.name

            # Import all .py files in directory
            for py_file in sorted(dir_path.glob('*.py')):
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
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not load {py_file.name}: {e}")

    def create(self, name: str, **kwargs):
        """
        Create an instance of a registered class.

        Convenience method that combines get() + instantiation.

        Args:
            name: String identifier of registered class
            **kwargs: Arguments to pass to class constructor

        Returns:
            Instance of the registered class

        Raises:
            KeyError: If name not registered
            TypeError: If instantiation fails

        Example:
            >>> dataset = DATASETS.create('h5dataset', path='data.h5')
            >>> # Equivalent to:
            >>> dataset_cls = DATASETS.get('h5dataset')
            >>> dataset = dataset_cls(path='data.h5')
        """
        cls = self.get(name)

        try:
            return cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate {self.name}['{name}'] with params {kwargs}.\n"
                f"Error: {e}\n"
                f"Class: {cls}\n"
                f"Hint: Check the __init__ signature of {cls.__name__}"
            ) from e

    def list_registered(self):
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str):
        """Check if name is registered."""
        return name in self._registry


# Global registries with auto-discovery
DATASETS = Registry('DATASETS', discovery_dirs=['./datasets', './custom_datasets'])
PREPROCESSING = Registry('PREPROCESSING', discovery_dirs=['./preprocessing', './custom_preprocessing'])
