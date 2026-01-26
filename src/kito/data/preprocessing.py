"""
Preprocessing classes for data transformation.

All preprocessing classes inherit from Preprocessing base class
and implement __call__ method.

Preprocessing can be:
- Composed using Pipeline
- Configured via config files
- Registered for factory instantiation
"""
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import torch

from src.kito.data.registry import PREPROCESSING


class Preprocessing(ABC):
    """
    Base class for all preprocessing operations.

    Preprocessing transforms (data, labels) → (processed_data, processed_labels)

    Subclasses must implement __call__ method.

    Example:
        class MyPreprocessing(Preprocessing):
            def __call__(self, data, labels):
                # Transform data
                return processed_data, labels
    """

    @abstractmethod
    def __call__(self, data, labels) -> Tuple:
        """
        Apply preprocessing.

        Args:
            data: Input data (numpy array or tensor)
            labels: Target labels (numpy array or tensor)

        Returns:
            Tuple of (processed_data, processed_labels)
        """
        pass


@PREPROCESSING.register('pipeline')
class Pipeline(Preprocessing):
    """
    Chain multiple preprocessing steps.

    Applies preprocessing steps sequentially.

    Args:
        steps: List of Preprocessing instances

    Example:
        >>> pipeline = Pipeline([
        ...     Detrend(),
        ...     Standardization(mean=0.5, std=0.2)
        ... ])
        >>> data, labels = pipeline(data, labels)
    """

    def __init__(self, steps: List[Preprocessing]):
        self.steps = steps

    def __call__(self, data, labels):
        for step in self.steps:
            data, labels = step(data, labels)
        return data, labels

    def __repr__(self):
        steps_repr = ', '.join([step.__class__.__name__ for step in self.steps])
        return f"Pipeline([{steps_repr}])"


@PREPROCESSING.register('normalize')
class Normalize(Preprocessing):
    """
    Min-max normalization: scale data to [min_val, max_val].

    Args:
        min_val: Minimum value after normalization
        max_val: Maximum value after normalization

    Example:
        >>> norm = Normalize(min_val=0, max_val=1)
        >>> data, labels = norm(data, labels)
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data, labels):
        data_min = data.min()
        data_max = data.max()

        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min)
            data = data * (self.max_val - self.min_val) + self.min_val

        return data, labels


@PREPROCESSING.register('standardization')
class Standardization(Preprocessing):
    """
    Standardize data: (data - mean) / std.

    Args:
        mean: Mean for standardization (None = compute from data)
        std: Standard deviation (None = compute from data)
        eps: Small constant to avoid division by zero

    Example:
        >>> # Compute mean/std from data
        >>> std = Standardization()
        >>> data, labels = std(data, labels)

        >>> # Use fixed mean/std
        >>> std = Standardization(mean=0.5, std=0.2)
        >>> data, labels = std(data, labels)
    """

    def __init__(self, mean: float = None, std: float = None, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps
        self._fitted = False

    def __call__(self, data, labels):
        # Compute mean/std on first call if not provided
        '''if self.mean is None and not self._fitted:
            self.mean = float(data.mean())
            self._fitted = True

        if self.std is None and not self._fitted:
            self.std = float(data.std())
            self._fitted = True'''
        if self.mean is None:
            if not self._fitted:
                self.mean = float(data.mean())
        if self.std is None:
            if not self._fitted:
                self.std = float(data.std())

        if not self._fitted:
            self._fitted = True

        # Standardize
        data = (data - self.mean) / (self.std + self.eps)

        return data, labels


@PREPROCESSING.register('clip_outliers')
class ClipOutliers(Preprocessing):
    """
    Clip outliers beyond n standard deviations.

    Args:
        n_std: Number of standard deviations for clipping

    Example:
        >>> clip = ClipOutliers(n_std=3)
        >>> data, labels = clip(data, labels)
    """

    def __init__(self, n_std: float = 3.0):
        self.n_std = n_std

    def __call__(self, data, labels):
        mean = data.mean()
        std = data.std()

        lower = mean - self.n_std * std
        upper = mean + self.n_std * std

        if isinstance(data, torch.Tensor):
            data = torch.clamp(data, lower, upper)
        else:
            data = np.clip(data, lower, upper)

        return data, labels


@PREPROCESSING.register('detrend')
class Detrend(Preprocessing):
    """
    Remove linear trend from data.

    Subtracts best-fit plane from each spatial slice.
    Useful for InSAR data with atmospheric gradients.

    Args:
        axis: Axis along which to detrend (None = all spatial axes)

    Example:
        >>> detrend = Detrend()
        >>> data, labels = detrend(data, labels)
    """

    def __init__(self, axis: int = None):
        self.axis = axis

    def __call__(self, data, labels):
        # Simple linear detrend (subtract mean along axis)
        # For more sophisticated detrending, override this method

        if self.axis is not None:
            mean = data.mean(axis=self.axis, keepdims=True)
        else:
            mean = data.mean()

        data = data - mean

        return data, labels


@PREPROCESSING.register('add_noise')
class AddNoise(Preprocessing):
    """
    Add Gaussian noise to data (for data augmentation).

    Args:
        std: Standard deviation of noise
        mean: Mean of noise

    Example:
        >>> noise = AddNoise(std=0.01)
        >>> data, labels = noise(data, labels)
    """

    def __init__(self, std: float = 0.01, mean: float = 0.0):
        self.std = std
        self.mean = mean

    def __call__(self, data, labels):
        if isinstance(data, torch.Tensor):
            noise = torch.randn_like(data) * self.std + self.mean
        else:
            noise = np.random.randn(*data.shape) * self.std + self.mean

        data = data + noise

        return data, labels


@PREPROCESSING.register('log_transform')
class LogTransform(Preprocessing):
    """
    Apply log transform: log(data + offset).

    Args:
        offset: Offset to ensure positivity
        base: Logarithm base (e, 10, 2)

    Example:
        >>> log = LogTransform(offset=1.0, base='e')
        >>> data, labels = log(data, labels)
    """

    def __init__(self, offset: float = 1.0, base: str = 'e'):
        self.offset = offset
        self.base = base

    def __call__(self, data, labels):
        data = data + self.offset

        if self.base == 'e':
            if isinstance(data, torch.Tensor):
                data = torch.log(data)
            else:
                data = np.log(data)
        elif self.base == '10':
            if isinstance(data, torch.Tensor):
                data = torch.log10(data)
            else:
                data = np.log10(data)
        elif self.base == '2':
            if isinstance(data, torch.Tensor):
                data = torch.log2(data)
            else:
                data = np.log2(data)

        return data, labels


@PREPROCESSING.register('to_tensor')
class ToTensor(Preprocessing):
    """
    Convert numpy arrays to PyTorch tensors.

    Args:
        dtype: Target dtype (e.g., torch.float32)

    Example:
        >>> to_tensor = ToTensor(dtype=torch.float32)
        >>> data, labels = to_tensor(data, labels)
    """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, data, labels):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).to(self.dtype)

        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels).to(self.dtype)

        return data, labels
