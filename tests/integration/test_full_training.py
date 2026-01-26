"""
Integration tests for full training workflows.

Tests end-to-end training pipelines.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from kito.engine import Engine
from kito.module import KitoModule
from tests.fixtures.mock_config import get_default_config


class SimpleRegressionModule(KitoModule):
    """Simple regression module for integration tests."""

    def build_inner_model(self):
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.model_input_size = (10,)
        self.standard_data_shape = (10,)

    def bind_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )


@pytest.mark.integration
class TestFullTrainingWorkflow:
    """Test complete training workflows."""

    def test_simple_training_run(self, device):
        """Test a simple end-to-end training run."""
        # Generate synthetic data
        x_train = torch.randn(100, 10)
        y_train = torch.randn(100, 10)
        x_val = torch.randn(20, 10)
        y_val = torch.randn(20, 10)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=16,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=16
        )

        # Create config
        config = get_default_config()
        config.training.n_train_epochs = 5
        config.training.learning_rate = 1e-3
        config.training.batch_size = 16

        # Create module and engine
        module = SimpleRegressionModule('TestModel', device, config)
        engine = Engine(module, config)

        # Train
        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=5,
            callbacks=[],
            train_verbosity_level=0,
            val_verbosity_level=0
        )

        # Verify training completed
        assert module.is_built
        assert module.is_optimizer_set
        assert engine.current_epoch == 5

    def test_training_with_explicit_build(self, device):
        """Test training with explicit model build."""
        # Data
        x = torch.randn(50, 10)
        y = torch.randn(50, 10)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=16)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=16)

        # Config
        config = get_default_config()

        # Build explicitly
        module = SimpleRegressionModule('TestModel', device, config)
        module.build()
        module.associate_optimizer()

        # Create engine and train
        engine = Engine(module, config)
        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=3,
            callbacks=[],
            train_verbosity_level=0,
            val_verbosity_level=0
        )

        assert engine.current_epoch == 3

    def test_training_loss_decreases(self, device, random_seed):
        """Test that training loss generally decreases."""
        # Generate simple learnable pattern
        x = torch.randn(100, 10)
        y = x * 2.0  # Simple linear relationship

        train_loader = DataLoader(TensorDataset(x, y), batch_size=16)
        val_loader = DataLoader(TensorDataset(x[:20], y[:20]), batch_size=16)

        config = get_default_config()
        config.training.learning_rate = 1e-2

        module = SimpleRegressionModule('TestModel', device, config)
        engine = Engine(module, config)

        # Track initial loss
        initial_losses = []

        class LossTracker:
            def __init__(self):
                self.losses = []

            def on_epoch_end(self, epoch, logs, **kwargs):
                self.losses.append(logs['train_loss'])

        tracker = LossTracker()

        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=10,
            callbacks=[tracker],
            train_verbosity_level=0,
            val_verbosity_level=0
        )

        # Loss should generally decrease
        assert len(tracker.losses) == 10
        assert tracker.losses[-1] < tracker.losses[0]  # Final < initial
