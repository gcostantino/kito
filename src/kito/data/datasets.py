import h5py

from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from kito.data.registry import DATASETS


class KitoDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.

    Provides common data loading and preprocessing pattern.
    Subclasses must implement _load_sample() to define how data is loaded.

    The standard workflow is:
    1. _load_sample(index) - Load raw data (subclass implements)
    2. Return data and labels

    Note: Preprocessing is now handled by PreprocessedDataset wrapper,
    so _preprocess_data() is removed from here.
    """

    @abstractmethod
    def _load_sample(self, index):
        """
        Load a single raw sample from the data source.

        Args:
            index: Sample index

        Returns:
            Tuple of (data, labels) as numpy arrays or tensors
        """
        pass

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Standard workflow:
        1. Load raw sample
        2. Return (preprocessing happens in PreprocessedDataset wrapper)
        """
        data, labels = self._load_sample(index)
        return data, labels

    @abstractmethod
    def __len__(self):
        """Get the total number of samples in the dataset."""
        pass


@DATASETS.register('h5dataset')
class H5Dataset(KitoDataset):
    """
    HDF5 dataset for PyTorch with lazy loading.

    Loads 'data' and 'labels' from HDF5 file with lazy loading for
    multiprocessing compatibility (DataLoader with num_workers > 0).

    Args:
        path: Path to HDF5 file

    HDF5 Structure Expected:
        - 'data': Input data array (N, ...)
        - 'labels': Target labels array (N, ...)

    Example:
        >>> dataset = H5Dataset("train.h5")
        >>> data, labels = dataset[0]
        >>> 
        >>> # Works with DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """

    def __init__(self, path: str):
        self.file_path = path

        # Lazy-loaded attributes (set in _lazy_load)
        self.dataset_data = None
        self.dataset_labels = None
        self.h5file = None

        # Get dataset length (without lazy loading)
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["data"])

    def _lazy_load(self):
        """
        Open HDF5 file and get dataset references.
        Called automatically in _load_sample().
        """
        if self.dataset_data is None or self.dataset_labels is None:
            try:
                self.h5file = h5py.File(self.file_path, 'r')
                self.dataset_data = self.h5file["data"]
                self.dataset_labels = self.h5file["labels"]
            except (OSError, KeyError) as e:
                raise RuntimeError(f"Failed to load H5 file '{self.file_path}': {e}")

    def _load_sample(self, index):
        """Load sample from HDF5 file with lazy loading."""
        self._lazy_load()
        return self.dataset_data[index], self.dataset_labels[index]

    def __len__(self):
        return self.dataset_len

    def __del__(self):
        """Close HDF5 file when object is destroyed."""
        if hasattr(self, 'h5file') and self.h5file is not None:
            self.h5file.close()

    def __getstate__(self):
        """
        Prepare object for pickling (needed for DataLoader with num_workers > 0).
        Remove non-picklable HDF5 file handles.
        """
        state = self.__dict__.copy()
        # Remove HDF5 file handle and dataset references
        state['dataset_data'] = None
        state['dataset_labels'] = None
        if 'h5file' in state:
            del state['h5file']
        return state

    def __setstate__(self, state):
        """Restore state after unpickling. File handles will be reloaded lazily."""
        self.__dict__.update(state)


@DATASETS.register('memdataset')
class MemDataset(KitoDataset):
    """
    In-memory dataset for PyTorch.

    Stores data and labels in memory (as numpy arrays or tensors).
    Useful when dataset fits in RAM for faster training.

    Args:
        x: Input data array (N, ...)
        y: Target labels array (N, ...)

    Example:
        >>> import numpy as np
        >>> x = np.random.randn(100, 10, 64, 64, 1)
        >>> y = np.random.randn(100, 10, 64, 64, 1)
        >>> dataset = MemDataset(x, y)
        >>> data, labels = dataset[0]
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Validate shapes
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length. Got {len(x)} and {len(y)}")

    def _load_sample(self, index):
        """Load sample from memory arrays."""
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
