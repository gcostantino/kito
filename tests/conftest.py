"""
Shared pytest fixtures for Kito tests.

Contains common fixtures used across multiple test files.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import h5py
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# Mock config classes (simplified version of old config)
@dataclass
class MockTrainingConfig:
    """Mock training configuration."""
    learning_rate: float = 1e-3
    n_train_epochs: int = 10
    batch_size: int = 8
    train_mode: bool = True
    distributed_training: bool = False
    master_gpu_id: int = 0


@dataclass
class MockModelConfig:
    """Mock model configuration."""
    loss: str = 'mean_squared_error'
    save_model_weights: bool = False
    train_codename: str = 'test_run'
    log_to_tensorboard: bool = False


@dataclass
class MockDataConfig:
    """Mock data configuration."""
    dataset_type: str = 'h5dataset'
    dataset_path: str = ''
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    total_samples: int = 100
    num_workers: int = 0  # 0 for testing (no multiprocessing)
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2


@dataclass
class MockWorkDirConfig:
    """Mock working directory configuration."""
    work_directory: str = '/tmp/kito_test'


@dataclass
class MockConfig:
    """Complete mock configuration."""
    training: MockTrainingConfig = field(default_factory=MockTrainingConfig)
    model: MockModelConfig = field(default_factory=MockModelConfig)
    data: MockDataConfig = field(default_factory=MockDataConfig)
    workdir: MockWorkDirConfig = field(default_factory=MockWorkDirConfig)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    return MockConfig()


@pytest.fixture
def device():
    """Provide a torch device (CPU for testing)."""
    return torch.device('cpu')


@pytest.fixture
def simple_model():
    """Provide a simple PyTorch model for testing."""
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, kernel_size=3, padding=1)
    )


@pytest.fixture
def sample_data():
    """Generate sample input/output data."""
    batch_size = 8
    channels = 1
    height, width = 32, 32

    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)

    return x, y


@pytest.fixture
def sample_numpy_data():
    """Generate sample numpy data."""
    n_samples = 100
    height, width = 32, 32

    x = np.random.randn(n_samples, 1, height, width).astype(np.float32)
    y = np.random.randn(n_samples, 1, height, width).astype(np.float32)

    return x, y


@pytest.fixture
def temp_h5_file(sample_numpy_data):
    """Create a temporary H5 file with sample data."""
    x, y = sample_numpy_data

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # Write data
    with h5py.File(temp_path, 'w') as f:
        f.create_dataset('data', data=x)
        f.create_dataset('labels', data=y)

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield temp_path

    # Cleanup
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    model = nn.Linear(10, 10)
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Reset after test
    torch.manual_seed(torch.initial_seed())
