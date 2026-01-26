"""
Integration tests for data pipeline.

Tests the full data pipeline workflow.
"""
import pytest
import torch
from kito.data.datasets import MemDataset
from kito.data.preprocessing import Pipeline, Normalize, ToTensor
from kito.data.preprocessed_dataset import PreprocessedDataset
from kito.data.datapipeline import GenericDataPipeline
from tests.fixtures.mock_config import get_default_config, get_memory_dataset_config
from tests.fixtures.sample_data import generate_random_data


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Test complete data pipeline workflows."""

    def test_data_pipeline_with_preprocessing(self, sample_numpy_data):
        """Test data pipeline with preprocessing."""
        x, y = sample_numpy_data

        # Create dataset
        dataset = MemDataset(x, y)

        # Create preprocessing pipeline
        preprocessing = Pipeline([
            Normalize(min_val=0.0, max_val=1.0),
            ToTensor()
        ])

        # Create config
        config = get_memory_dataset_config()
        config.data.total_samples = len(x)

        # Create data pipeline
        pipeline = GenericDataPipeline(
            config=config,
            dataset=dataset,
            preprocessing=preprocessing
        )
        pipeline.setup()

        # Get dataloaders
        train_loader = pipeline.train_dataloader()
        val_loader = pipeline.val_dataloader()

        # Verify data
        for batch_x, batch_y in train_loader:
            assert isinstance(batch_x, torch.Tensor)
            assert isinstance(batch_y, torch.Tensor)
            assert batch_x.min() >= 0.0
            assert batch_x.max() <= 1.0
            break

    def test_data_pipeline_splits(self, sample_numpy_data):
        """Test that data pipeline creates correct splits."""
        x, y = sample_numpy_data
        dataset = MemDataset(x, y)

        config = get_memory_dataset_config()
        config.data.train_ratio = 0.7
        config.data.val_ratio = 0.2
        config.data.total_samples = len(x)

        pipeline = GenericDataPipeline(config=config, dataset=dataset)
        pipeline.setup()

        train_loader = pipeline.train_dataloader()
        val_loader = pipeline.val_dataloader()
        test_loader = pipeline.test_dataloader()

        # Check approximate split sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        total = train_size + val_size + test_size
        assert total == len(x)

        # Check ratios are approximately correct
        assert abs(train_size / total - 0.7) < 0.1
        assert abs(val_size / total - 0.2) < 0.1
