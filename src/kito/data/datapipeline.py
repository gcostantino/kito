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

from torch.utils.data import DataLoader, Subset, DistributedSampler

from kito.config.moduleconfig import KitoModuleConfig
from kito.data.datasets import KitoDataset
from kito.data.preprocessed_dataset import PreprocessedDataset
from kito.data.preprocessing import Preprocessing


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
            raise ValueError("Dataset not provided to GenericDataModule")

        # 2. Apply preprocessing
        if self._preprocessing is not None:
            self.dataset = PreprocessedDataset(raw_dataset, self._preprocessing)
        else:
            self.dataset = raw_dataset

        # 3. Create splits
        self._create_splits()

        # 4. Create DataLoaders
        self._create_dataloaders()

    def _create_splits(self):
        """
        Create train/val/test splits using Subset.

        Splits based on data_config.train_ratio and data_config.val_ratio.
        """
        total_samples = self.data_config.total_samples
        if total_samples is None:
            total_samples = len(self.dataset)

        # Calculate split indices
        train_size = int(total_samples * self.data_config.train_ratio)
        val_size = int(total_samples * self.data_config.val_ratio)
        test_size = total_samples - train_size - val_size

        # Create subsets
        self.train_dataset = Subset(
            self.dataset,
            list(range(0, train_size))
        )
        self.val_dataset = Subset(
            self.dataset,
            list(range(train_size, train_size + val_size))
        )
        self.test_dataset = Subset(
            self.dataset,
            list(range(train_size + val_size, total_samples))
        )

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
            if distributed:
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
            self.train_loader = None  # no loader if dataset is empty
            if self.train_dataset is not None:
                print("Warning: Training dataset is empty. Skipping train_loader creation.")

        # Val loader (no shuffling)
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            if distributed:
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
                print("Warning: Validation dataset is empty. Skipping val_loader creation.")

        # Test loader (no shuffling)
        if self.test_dataset is not None and len(self.test_dataset) > 0:
            if distributed:
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
                print("Warning: Test dataset is empty. Skipping test_loader creation.")

            # validation: at least one loader must exist
            if self.train_loader is None and self.val_loader is None and self.test_loader is None:
                raise ValueError(
                    "All dataloaders are None! At least one of (train_dataset, val_dataset, test_dataset) "
                    "must be provided and non-empty.")

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        if self.train_loader is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        if self.val_loader is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        if self.test_loader is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self.test_loader
