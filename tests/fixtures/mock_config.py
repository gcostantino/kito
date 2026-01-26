"""
Mock configuration classes for testing.

Provides lightweight config objects similar to your dataclass structure
but simplified for testing purposes.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class KitoTrainingConfig:
    """Training configuration for tests."""
    learning_rate: float = 1e-3
    n_train_epochs: int = 10
    batch_size: int = 8
    train_mode: bool = True
    distributed_training: bool = False
    master_gpu_id: int = 0


@dataclass
class KitoModelConfig:
    """Model configuration for tests."""
    loss: str = 'mean_squared_error'
    save_model_weights: bool = False
    train_codename: str = 'test_run'
    log_to_tensorboard: bool = False


@dataclass
class KitoDataConfig:
    """Data configuration for tests."""
    dataset_type: str = 'h5dataset'
    dataset_path: str = ''
    dataset_init_args: Optional[Dict[str, Any]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    total_samples: Optional[int] = 100
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2


@dataclass
class KitoWorkDirConfig:
    """Working directory configuration for tests."""
    work_directory: str = '/tmp/kito_test'


@dataclass
class KitoModuleConfig:
    """Complete configuration for Kito module."""
    training: KitoTrainingConfig = field(default_factory=KitoTrainingConfig)
    model: KitoModelConfig = field(default_factory=KitoModelConfig)
    data: KitoDataConfig = field(default_factory=KitoDataConfig)
    workdir: KitoWorkDirConfig = field(default_factory=KitoWorkDirConfig)


def get_default_config() -> KitoModuleConfig:
    """Get default test configuration."""
    return KitoModuleConfig()


def get_ddp_config() -> KitoModuleConfig:
    """Get configuration for DDP testing."""
    config = KitoModuleConfig()
    config.training.distributed_training = True
    return config


def get_memory_dataset_config() -> KitoModuleConfig:
    """Get configuration for in-memory dataset."""
    config = KitoModuleConfig()
    config.data.dataset_type = 'memdataset'
    config.data.dataset_init_args = {'x': None, 'y': None}
    return config
