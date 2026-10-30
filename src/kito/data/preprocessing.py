"""
Preprocessing steps for Kito datasets.

A preprocessing step is a callable ``step(data, labels) -> (data, labels)`` that
works on ONE sample (NumPy arrays or torch tensors).

Design rules:

- Steps never estimate statistics from the sample they are given. Anything that
  depends on dataset statistics (mean, std, min, max) must be passed explicitly.
  Use :func:`compute_stats` on your training data to obtain these values.
- Steps change ``data`` only by default. Use ``apply_to='labels'`` or
  ``apply_to='both'`` when the targets need the same transformation
  (e.g. denoising, regression on images).
- Invalid parameters raise ``ValueError`` when the step is created, not in the
  middle of training.
- Steps marked ``training_only = True`` (e.g. :class:`AddNoise`) are skipped by
  ``Engine.predict`` and by the validation/test splits of ``GenericDataPipeline``.

Steps are registered by name and can be created from configuration:
``normalize``, ``standardization``, ``clip``, ``log_transform``, ``add_noise``,
``to_tensor``, ``pipeline``.
"""
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch

from kito.data.registry import PREPROCESSING

_APPLY_TO = ('data', 'labels', 'both')


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _check_apply_to(apply_to: str) -> str:
    if apply_to not in _APPLY_TO:
        raise ValueError(f"apply_to must be one of {_APPLY_TO}, got '{apply_to}'")
    return apply_to


def _to_float(x):
    """Return x as a floating-point array/tensor (integers become float32)."""
    if isinstance(x, torch.Tensor):
        return x if x.is_floating_point() else x.float()
    x = np.asarray(x)
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float32)


def _stat(value, x, channel_axis: Optional[int]):
    """
    Convert a statistic (scalar or 1-D per-channel values) to the type and dtype of ``x``.

    With ``channel_axis`` set, a 1-D statistic is reshaped so that it broadcasts along that axis
    of ``x``. Without it, the statistic is used as given (NumPy broadcasting rules).
    """
    arr = np.asarray(value, dtype=np.float64)
    if channel_axis is not None and arr.ndim == 1:
        shape = [1] * x.ndim
        shape[channel_axis] = arr.shape[0]
        arr = arr.reshape(shape)
    if isinstance(x, torch.Tensor):
        return torch.as_tensor(arr, dtype=x.dtype, device=x.device)
    return arr.astype(x.dtype)


def _resolve_dtype(dtype):
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown torch dtype '{dtype}'")
        return resolved
    return dtype


# ----------------------------------------------------------------------------
# Base classes
# ----------------------------------------------------------------------------

class Preprocessing(ABC):
    """Base class for all preprocessing steps."""

    #: If True, the step is only applied to training data.
    training_only = False

    @abstractmethod
    def __call__(self, data, labels) -> Tuple:
        """Transform one sample and return ``(data, labels)``."""


class _SampleTransform(Preprocessing):
    """Base for steps that apply the same function to data, labels, or both."""

    def __init__(self, apply_to: str = 'data'):
        self.apply_to = _check_apply_to(apply_to)

    @abstractmethod
    def _transform(self, x):
        """Transform a single array/tensor."""

    def __call__(self, data, labels):
        if self.apply_to in ('data', 'both'):
            data = self._transform(data)
        if self.apply_to in ('labels', 'both') and labels is not None:
            labels = self._transform(labels)
        return data, labels


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------

@PREPROCESSING.register('pipeline')
class Pipeline(Preprocessing):
    """
    Apply several steps in order. Nested pipelines are flattened.

    Args:
        steps: List of Preprocessing instances.
    """

    def __init__(self, steps: List[Preprocessing]):
        flat = []
        for step in steps:
            if isinstance(step, Pipeline):
                flat.extend(step.steps)
            elif isinstance(step, Preprocessing):
                flat.append(step)
            else:
                raise TypeError(f"Pipeline steps must be Preprocessing instances, got {type(step)}")
        self.steps = flat

    def __call__(self, data, labels):
        for step in self.steps:
            data, labels = step(data, labels)
        return data, labels

    def __repr__(self):
        return f"Pipeline([{', '.join(step.__class__.__name__ for step in self.steps)}])"


@PREPROCESSING.register('standardization')
class Standardization(_SampleTransform):
    """
    ``(x - mean) / std`` with fixed statistics.

    Args:
        mean: Scalar, or per-channel values (with ``channel_axis``).
        std: Scalar, or per-channel values. Must be > 0.
        channel_axis: Axis of one sample that ``mean``/``std`` refer to (e.g. 0 for (C, H, W)).
        apply_to: 'data', 'labels' or 'both'.

    Use :func:`compute_stats` to get ``mean`` and ``std`` from the training data.
    """

    def __init__(self, mean, std, channel_axis: Optional[int] = None, apply_to: str = 'data'):
        super().__init__(apply_to)
        if mean is None or std is None:
            raise ValueError("Standardization needs explicit 'mean' and 'std'. "
                             "Compute them on the training data with compute_stats().")
        if np.any(np.asarray(std, dtype=np.float64) <= 0):
            raise ValueError("Standardization: std must be > 0")
        self.mean = mean
        self.std = std
        self.channel_axis = channel_axis

    def _transform(self, x):
        x = _to_float(x)
        return (x - _stat(self.mean, x, self.channel_axis)) / _stat(self.std, x, self.channel_axis)

    def inverse(self, x):
        """Undo the standardization (e.g. on model predictions)."""
        x = _to_float(x)
        return x * _stat(self.std, x, self.channel_axis) + _stat(self.mean, x, self.channel_axis)


@PREPROCESSING.register('normalize')
class Normalize(_SampleTransform):
    """
    Linear rescaling of the range ``[data_min, data_max]`` to ``[min_val, max_val]``, with fixed bounds.
    Values outside the bounds are not clipped (combine with :class:`Clip` if needed).

    Args:
        data_min: Lower bound of the data (scalar or per-channel).
        data_max: Upper bound of the data. Must be > data_min.
        min_val: Target lower bound (default 0.0).
        max_val: Target upper bound (default 1.0).
        channel_axis: Axis that per-channel bounds refer to.
        apply_to: 'data', 'labels' or 'both'.
    """

    def __init__(self, data_min, data_max, min_val: float = 0.0, max_val: float = 1.0,
                 channel_axis: Optional[int] = None, apply_to: str = 'data'):
        super().__init__(apply_to)
        if data_min is None or data_max is None:
            raise ValueError("Normalize needs explicit 'data_min' and 'data_max'. "
                             "Compute them on the training data with compute_stats().")
        if np.any(np.asarray(data_max, dtype=np.float64) <= np.asarray(data_min, dtype=np.float64)):
            raise ValueError("Normalize: data_max must be > data_min")
        if not max_val > min_val:
            raise ValueError("Normalize: max_val must be > min_val")
        self.data_min = data_min
        self.data_max = data_max
        self.min_val = min_val
        self.max_val = max_val
        self.channel_axis = channel_axis

    def _transform(self, x):
        x = _to_float(x)
        lo = _stat(self.data_min, x, self.channel_axis)
        hi = _stat(self.data_max, x, self.channel_axis)
        return (x - lo) / (hi - lo) * (self.max_val - self.min_val) + self.min_val

    def inverse(self, x):
        """Undo the rescaling."""
        x = _to_float(x)
        lo = _stat(self.data_min, x, self.channel_axis)
        hi = _stat(self.data_max, x, self.channel_axis)
        return (x - self.min_val) / (self.max_val - self.min_val) * (hi - lo) + lo


@PREPROCESSING.register('clip')
class Clip(_SampleTransform):
    """
    Clip values to ``[min_value, max_value]``. Either bound may be None.

    Args:
        min_value: Lower bound or None.
        max_value: Upper bound or None.
        apply_to: 'data', 'labels' or 'both'.
    """

    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None,
                 apply_to: str = 'data'):
        super().__init__(apply_to)
        if min_value is None and max_value is None:
            raise ValueError("Clip needs at least one of min_value / max_value")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("Clip: min_value must be <= max_value")
        self.min_value = min_value
        self.max_value = max_value

    def _transform(self, x):
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, self.min_value, self.max_value)
        return np.clip(np.asarray(x), self.min_value, self.max_value)


_LOG_BASES = {
    'e': (np.log, torch.log, math.e),
    '10': (np.log10, torch.log10, 10.0),
    '2': (np.log2, torch.log2, 2.0),
}


@PREPROCESSING.register('log_transform')
class LogTransform(_SampleTransform):
    """
    ``log(x + offset)``. ``x + offset`` must be strictly positive, otherwise ``ValueError`` is raised
    when the sample is processed.

    Args:
        offset: Value added before the logarithm (default 1.0).
        base: 'e', '10' or '2'.
        apply_to: 'data', 'labels' or 'both'.
    """

    def __init__(self, offset: float = 1.0, base='e', apply_to: str = 'data'):
        super().__init__(apply_to)
        base = str(base)
        if base not in _LOG_BASES:
            raise ValueError(f"LogTransform: base must be one of {list(_LOG_BASES)}, got '{base}'")
        self.offset = offset
        self.base = base

    def _transform(self, x):
        x = _to_float(x) + self.offset
        if bool((x <= 0).any()):
            raise ValueError("LogTransform: data + offset must be strictly positive")
        np_fn, torch_fn, _ = _LOG_BASES[self.base]
        return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)

    def inverse(self, x):
        """Undo the transform: ``base ** x - offset``."""
        x = _to_float(x)
        base_value = _LOG_BASES[self.base][2]
        if isinstance(x, torch.Tensor):
            return torch.pow(torch.as_tensor(base_value, dtype=x.dtype, device=x.device), x) - self.offset
        return np.power(np.asarray(base_value, dtype=x.dtype), x) - self.offset


@PREPROCESSING.register('add_noise')
class AddNoise(_SampleTransform):
    """
    Add Gaussian noise ``N(mean, std)`` (data augmentation). Training only.
    The dtype of the input is preserved.

    Args:
        std: Noise standard deviation (default 0.01), must be >= 0.
        mean: Noise mean (default 0.0).
        apply_to: 'data', 'labels' or 'both'.
    """

    training_only = True

    def __init__(self, std: float = 0.01, mean: float = 0.0, apply_to: str = 'data'):
        super().__init__(apply_to)
        if std < 0:
            raise ValueError("AddNoise: std must be >= 0")
        self.std = std
        self.mean = mean

    def _transform(self, x):
        x = _to_float(x)
        if isinstance(x, torch.Tensor):
            return x + torch.randn_like(x) * self.std + self.mean
        noise = np.asarray(np.random.standard_normal(x.shape), dtype=x.dtype)
        return x + noise * self.std + self.mean


@PREPROCESSING.register('to_tensor')
class ToTensor(Preprocessing):
    """
    Convert data and labels to torch tensors.

    Args:
        dtype: dtype of the data (torch dtype or its name, e.g. 'float32').
        label_dtype: dtype of the labels. Defaults to ``dtype``. Use ``torch.long`` for class indices.
    """

    def __init__(self, dtype=torch.float32, label_dtype=None):
        self.dtype = _resolve_dtype(dtype)
        self.label_dtype = _resolve_dtype(label_dtype) if label_dtype is not None else self.dtype

    @staticmethod
    def _convert(x, dtype):
        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x))
        return x.to(dtype)

    def __call__(self, data, labels):
        return self._convert(data, self.dtype), self._convert(labels, self.label_dtype)


# ----------------------------------------------------------------------------
# Statistics helper
# ----------------------------------------------------------------------------

def compute_stats(dataset, n_samples: Optional[int] = None, channel_axis: Optional[int] = None,
                  which: str = 'data', seed: int = 0) -> dict:
    """
    Compute mean, std (population), min and max of a dataset, for use in
    :class:`Standardization` and :class:`Normalize`.

    Use the training data without preprocessing (the raw dataset, or a training subset).
    Statistics are accumulated sample by sample in float64 (Chan et al. merge), so memory use is
    one sample. Non-finite values raise ``ValueError``.

    Args:
        dataset: Object with ``len`` and ``dataset[i] -> (data, labels)``.
        n_samples: Use a random subset of this size (seeded). None uses every sample.
        channel_axis: Axis of one sample for per-channel statistics. None gives scalar statistics.
        which: 'data' or 'labels'.
        seed: Seed for the random subset.

    Returns:
        Dict with ``mean``, ``std``, ``min``, ``max`` (floats, or lists with ``channel_axis``)
        and ``count`` (number of values per channel).
    """
    if which not in ('data', 'labels'):
        raise ValueError("which must be 'data' or 'labels'")
    total = len(dataset)
    if total == 0:
        raise ValueError("compute_stats: dataset is empty")
    if n_samples is None or n_samples >= total:
        indices = range(total)
    else:
        indices = np.sort(np.random.default_rng(seed).choice(total, size=n_samples, replace=False))
    position = 0 if which == 'data' else 1

    count, mean, m2, lo, hi = 0, None, None, None, None
    for i in indices:
        x = dataset[int(i)][position]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"compute_stats: non-finite values in sample {int(i)}")
        if channel_axis is None:
            x2 = x.reshape(1, -1)
        else:
            x2 = np.moveaxis(x, channel_axis, 0).reshape(x.shape[channel_axis], -1)
        n_b = x2.shape[1]
        mean_b = x2.mean(axis=1)
        m2_b = ((x2 - mean_b[:, None]) ** 2).sum(axis=1)
        if mean is None:
            count, mean, m2 = n_b, mean_b, m2_b
            lo, hi = x2.min(axis=1), x2.max(axis=1)
        else:
            n = count + n_b
            delta = mean_b - mean
            m2 = m2 + m2_b + delta ** 2 * count * n_b / n
            mean = mean + delta * n_b / n
            count = n
            lo, hi = np.minimum(lo, x2.min(axis=1)), np.maximum(hi, x2.max(axis=1))

    std = np.sqrt(m2 / count)

    def _out(a):
        return float(a[0]) if channel_axis is None else [float(v) for v in a]

    return {'mean': _out(mean), 'std': _out(std), 'min': _out(lo), 'max': _out(hi), 'count': int(count)}
