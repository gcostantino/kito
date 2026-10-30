"""
DataPipeline classes - Orchestrate data pipeline.

DataPipeline encapsulates:
- Dataset creation
- Preprocessing setup
- Train/val/test splitting
- DataLoader configuration

This separates data concerns from model training logic.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler

from kito.config.moduleconfig import KitoModuleConfig
from kito.data.datasets import KitoDataset
from kito.data.preprocessed_dataset import PreprocessedDataset
from kito.data.preprocessing import Preprocessing
from kito.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmptySplitError(ValueError):
    """Raised when a requested split (train/val/test) has no samples."""


class BaseDataPipeline(ABC):
    """
    Base class for data modules.

    A DataModule encapsulates all data-related logic:
    - How to load data (dataset)
    - How to preprocess data
    - How to split into train/val/test
    - How to create DataLoaders

    Usage:
        class MyDataModule(BaseDataModule):
            def setup(self):
                self.dataset = MyDataset(self.data_config.dataset_path)
                # ... setup preprocessing, splits, etc.

            def train_dataloader(self):
                return DataLoader(self.train_dataset, ...)

        # In training
        dm = MyDataModule(config)
        dm.setup()
        engine.fit(datamodule=dm)
    """

    def __init__(self, config: KitoModuleConfig):
        """
        Initialize DataModule.

        Args:
            config: Full module configuration (contains data config)
        """
        self.config = config
        self.data_config = config.data

        # Will be set by setup()
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    @abstractmethod
    def setup(self):
        """
        Setup datasets and preprocessing.

        This method should:
        1. Load raw dataset
        2. Apply preprocessing (if any)
        3. Create train/val/test splits
        4. Create DataLoaders

        Called once before training.
        """
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        pass


class GenericDataPipeline(BaseDataPipeline):
    """
    Generic DataModule that works with any registered dataset.

    It handles:
    - Loading dataset from registry
    - Applying preprocessing pipeline
    - Creating train/val/test splits
    - Setting up DataLoaders with DDP support

    Args:
        config: Full module configuration
        dataset: Pre-instantiated dataset (optional)
        preprocessing: Pre-instantiated preprocessing (optional)

    Example:
        >>> dm = GenericDataModule(config)
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
            self,
            config: KitoModuleConfig,
            dataset: Optional[KitoDataset] = None,
            preprocessing: Optional[Preprocessing] = None
    ):
        super().__init__(config)
        self._dataset = dataset  # Pre-instantiated dataset
        self._preprocessing = preprocessing  # Pre-instantiated preprocessing

    def setup(self):
        """
        Setup data pipeline.

        Steps:
        1. Use pre-instantiated dataset or load from registry
        2. Wrap with preprocessing if provided
        3. Create train/val/test splits
        4. Create DataLoaders
        """
        # 1. Get dataset
        if self._dataset is not None:
            raw_dataset = self._dataset
        else:
            from kito.data.registry import DATASETS
            cfg = self.data_config
            kwargs = dict(cfg.dataset_init_args) or {'path': cfg.dataset_path}
            if not kwargs.get('path') and not cfg.dataset_init_args:
                raise ValueError("Provide a dataset, config.data.dataset_init_args, or config.data.dataset_path")
            raw_dataset = DATASETS.create(cfg.dataset_type, **kwargs)

        # 2. Apply preprocessing
        steps = self._preprocessing if self._preprocessing is not None else (
                    list(self.data_config.preprocessing) or None)
        self.dataset = PreprocessedDataset(raw_dataset, steps) if steps else raw_dataset

        # 3. Create splits
        self._create_splits()

        # 4. Check for training-only preprocessing steps
        if steps and isinstance(self.dataset, PreprocessedDataset):
            eval_steps = [s for s in self.dataset.preprocessing.steps
                          if not getattr(s, 'training_only', False)]
            if len(eval_steps) != len(self.dataset.preprocessing.steps):
                eval_ds = PreprocessedDataset(raw_dataset, eval_steps or None)
                self.val_dataset = Subset(eval_ds, self.val_dataset.indices)
                self.test_dataset = Subset(eval_ds, self.test_dataset.indices)


        # 5. Create DataLoaders
        self._create_dataloaders()

    def _create_splits(self):
        """Create train/val/test subsets from train_ratio / val_ratio."""
        n = len(self.dataset)
        total = self.data_config.total_samples
        if total is None:
            total = n
        if total > n:
            raise ValueError(f"total_samples={total} exceeds dataset length {n}")

        tr, vr = self.data_config.train_ratio, self.data_config.val_ratio
        if not (0.0 <= tr <= 1.0 and 0.0 <= vr <= 1.0 and tr + vr <= 1.0 + 1e-9):
            raise ValueError(
                f"Invalid split ratios: train_ratio={tr}, val_ratio={vr} "
                "(each must be in [0, 1] and their sum <= 1)"
            )

        train_size = int(total * tr)
        val_size = int(total * vr)

        if getattr(self.data_config, 'shuffle_split', False):
            g = torch.Generator().manual_seed(getattr(self.data_config, 'split_seed', 0))
            indices = torch.randperm(total, generator=g).tolist()  # same on every rank
        else:
            indices = list(range(total))

        self.train_dataset = Subset(self.dataset, indices[:train_size])
        self.val_dataset = Subset(self.dataset, indices[train_size:train_size + val_size])
        self.test_dataset = Subset(self.dataset, indices[train_size + val_size:])

    def _create_dataloaders(self):
        """
        Create DataLoaders with proper settings.

        Handles:
            - Distributed training (DistributedSampler)
            - num_workers, pin_memory, etc.
            - Batch size from config

        Raises:
            ValueError: If all datasets are empty or None
        """
        batch_size = self.config.training.batch_size
        distributed = self.config.training.distributed_training

        # Check if DDP is ACTUALLY initialized (not just configured)
        ddp_initialized = dist.is_available() and dist.is_initialized()
        use_distributed_sampler = distributed and ddp_initialized

        # DataLoader kwargs
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': self.data_config.num_workers,
            'pin_memory': self.data_config.pin_memory,
            'persistent_workers': self.data_config.persistent_workers if self.data_config.num_workers > 0 else False,
            'prefetch_factor': self.data_config.prefetch_factor if self.data_config.num_workers > 0 else None,
        }

        # Train loader (with shuffling)
        if self.train_dataset is not None and len(self.train_dataset) > 0:
            if use_distributed_sampler:
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    sampler=train_sampler,
                    shuffle=False,  # Don't shuffle when using sampler
                    **loader_kwargs
                )
            else:
                self.train_loader = DataLoader(
                    self.train_dataset,
                    shuffle=True,
                    **loader_kwargs
                )
        else:
            self.train_loader = None
            if self.train_dataset is not None:
                logger.warning("Warning: Training dataset is empty. Skipping train_loader creation.")

        # Val loader (no shuffling)
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            if use_distributed_sampler:
                val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
                self.val_loader = DataLoader(
                    self.val_dataset,
                    sampler=val_sampler,
                    shuffle=False,
                    **loader_kwargs
                )
            else:
                self.val_loader = DataLoader(
                    self.val_dataset,
                    shuffle=False,
                    **loader_kwargs
                )
        else:
            self.val_loader = None
            if self.val_dataset is not None:
                logger.warning("Warning: Validation dataset is empty. Skipping val_loader creation.")

        # Test loader (no shuffling)
        if self.test_dataset is not None and len(self.test_dataset) > 0:
            if use_distributed_sampler:
                test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
                self.test_loader = DataLoader(
                    self.test_dataset,
                    sampler=test_sampler,
                    shuffle=False,
                    **loader_kwargs
                )
            else:
                self.test_loader = DataLoader(
                    self.test_dataset,
                    shuffle=False,
                    **loader_kwargs
                )
        else:
            self.test_loader = None
            if self.test_dataset is not None:
                logger.warning("Warning: Test dataset is empty. Skipping test_loader creation.")

        # validation: at least one loader must exist
        if self.train_loader is None and self.val_loader is None and self.test_loader is None:
            raise ValueError(
                "All dataloaders are None! At least one of (train_dataset, val_dataset, test_dataset) "
                "must be provided and non-empty.")

    def _get_loader(self, name: str) -> DataLoader:
        loader = getattr(self, f'{name}_loader')
        if loader is None:
            if self.dataset is None:
                raise RuntimeError("Call setup() before accessing dataloaders")
            raise EmptySplitError(
                f"The '{name}' split is empty "
                f"(train_ratio={self.data_config.train_ratio}, val_ratio={self.data_config.val_ratio})"
            )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._get_loader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_loader('val')

    def test_dataloader(self) -> DataLoader:
        return self._get_loader('test')
