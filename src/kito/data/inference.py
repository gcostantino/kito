"""
Inference data helpers.

`as_dataloader` turns whatever the user has (DataLoader, Dataset, data pipeline,
numpy array, tensor, (x, y) tuple) into a DataLoader ready for `Engine.predict`.
"""
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kito.config.moduleconfig import KitoModuleConfig
from kito.data.datapipeline import BaseDataPipeline
from kito.data.datasets import KitoDataset
from kito.data.preprocessed_dataset import PreprocessedDataset
from kito.data.preprocessing import Preprocessing


def _to_float32(a):
    """float64 -> float32 so external numpy data matches the model dtype."""
    if isinstance(a, np.ndarray) and a.dtype == np.float64:
        return a.astype(np.float32)
    if isinstance(a, torch.Tensor) and a.dtype == torch.float64:
        return a.float()
    return a


class InferenceDataset(KitoDataset):
    """
    In-memory dataset for inference. Labels are optional.

    Preprocessing steps are called as (data, labels), so when no labels are given
    the sample itself is used as a placeholder label. Prediction ignores labels.
    """

    def __init__(self, x, y=None):
        self.x = x if hasattr(x, 'shape') else np.asarray(x)
        self.y = y
        if y is not None and len(x) != len(y):
            raise ValueError(f"x and y must have same length. Got {len(x)} and {len(y)}")

    def _load_sample(self, index):
        x = _to_float32(self.x[index])
        y = x if self.y is None else _to_float32(self.y[index])
        return x, y

    def __len__(self):
        return len(self.x)


def _is_arraylike(obj) -> bool:
    return isinstance(obj, (np.ndarray, torch.Tensor, list))


def as_dataloader(
        data: Any,
        config: Optional[KitoModuleConfig] = None,
        *,
        batch_size: Optional[int] = None,
        preprocessing: Union[None, bool, list, Preprocessing] = None,
        num_workers: Optional[int] = None,
) -> DataLoader:
    """
    Build an inference DataLoader from flexible input.

    Args:
        data: DataLoader | data pipeline | Dataset | ndarray | Tensor | list | (x, y) tuple.
        config: Source of defaults (batch size, num_workers, data.preprocessing).
        batch_size: Overrides config.training.batch_size.
        preprocessing: None -> use config.data.preprocessing; False -> none;
            a Preprocessing instance or list of steps -> use these instead.
            Not applied to DataLoaders / pipelines (assumed ready).
        num_workers: Overrides config.data.num_workers (forced to 0 for in-memory data).
    """
    # Already ready to use
    if isinstance(data, DataLoader):
        return data
    if isinstance(data, BaseDataPipeline):
        if data.dataset is None:
            data.setup()
        return data.test_dataloader()  # raises EmptySplitError with a clear message

    # Build a dataset
    in_memory = False
    if isinstance(data, tuple) and len(data) == 2 and all(_is_arraylike(d) for d in data):
        dataset, in_memory = InferenceDataset(*data), True
    elif isinstance(data, Dataset):
        dataset = data
    elif _is_arraylike(data):
        dataset, in_memory = InferenceDataset(data), True
    else:
        raise TypeError(
            f"Unsupported data type for inference: {type(data)}. Expected DataLoader, "
            "data pipeline, Dataset, numpy array, tensor, list, or (x, y) tuple."
        )

    # Preprocessing (default: the one from config.data.preprocessing)
    if preprocessing is None and config is not None:
        preprocessing = list(config.data.preprocessing) or None
    if preprocessing is False or preprocessing is None:
        steps = None
    elif isinstance(preprocessing, Preprocessing):
        steps = [preprocessing]
    else:
        steps = list(preprocessing) or None

    if steps is not None:
        dataset = PreprocessedDataset(dataset, steps)
        # drop augmentation-only steps (e.g. AddNoise) at inference
        pipe = dataset.preprocessing
        if hasattr(pipe, 'steps'):
            pipe.steps = [s for s in pipe.steps if not getattr(s, 'training_only', False)]

    if batch_size is None:
        batch_size = config.training.batch_size if config is not None else 1
    if num_workers is None:
        num_workers = getattr(config.data, 'num_workers', 0) if config is not None else 0
    if in_memory:
        num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,  # every sample must get a prediction
        num_workers=num_workers,
    )
