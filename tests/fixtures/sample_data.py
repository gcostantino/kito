"""
Utilities for generating sample data for tests.
"""
import numpy as np
import torch
import h5py
from pathlib import Path


def generate_random_data(n_samples=100, shape=(1, 32, 32), dtype=np.float32):
    """
    Generate random numpy arrays for testing.

    Args:
        n_samples: Number of samples
        shape: Shape of each sample (C, H, W)
        dtype: Data type

    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    data = np.random.randn(n_samples, *shape).astype(dtype)
    labels = np.random.randn(n_samples, *shape).astype(dtype)
    return data, labels


def create_h5_dataset(path, n_samples=100, shape=(1, 32, 32)):
    """
    Create an H5 file with random data.

    Args:
        path: Path to save H5 file
        n_samples: Number of samples
        shape: Shape of each sample (C, H, W)
    """
    data, labels = generate_random_data(n_samples, shape)

    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('labels', data=labels)


def generate_torch_batch(batch_size=8, shape=(1, 32, 32)):
    """
    Generate a random PyTorch batch.

    Args:
        batch_size: Number of samples in batch
        shape: Shape of each sample (C, H, W)

    Returns:
        Tuple of (inputs, targets) as torch tensors
    """
    inputs = torch.randn(batch_size, *shape)
    targets = torch.randn(batch_size, *shape)
    return inputs, targets


class DummyModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
