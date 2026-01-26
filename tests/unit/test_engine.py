"""
Unit tests for Engine.

Tests the training orchestration engine (with mocking).
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch, call
from torch.utils.data import DataLoader, TensorDataset
from kito.engine import Engine
from kito.module import KitoModule
from tests.fixtures.mock_config import get_default_config


class MockModule(KitoModule):
    """Mock module for engine tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_step_called = 0
        self._validation_step_called = 0

    def build_inner_model(self):
        self.model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        self.model_input_size = (10,)
        self.standard_data_shape = (10,)

    def bind_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def training_step(self, batch, pbar_handler=None):
        self._training_step_called += 1
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss}

    def validation_step(self, batch, pbar_handler=None):
        self._validation_step_called += 1
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets,
            'inputs': inputs
        }


class TestEngineInitialization:
    """Test Engine initialization."""

    def test_engine_creation(self, device, mock_config):
        """Test creating an engine."""
        module = MockModule('TestModel', device, mock_config)
        engine = Engine(module, mock_config)

        assert engine.module == module
        assert engine.config == mock_config
        assert engine.current_epoch == 0
        assert engine.stop_training == False

    def test_engine_device_assignment(self, device, mock_config):
        """Test engine device assignment."""
        module = MockModule('TestModel', device, mock_config)
        engine = Engine(module, mock_config)

        assert engine.device.type == 'cpu'  # In tests we use CPU

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.get_rank')
    def test_engine_ddp_initialization(self, mock_rank, mock_init, device):
        """Test engine initialization with DDP."""
        mock_init.return_value = True
        mock_rank.return_value = 0

        config = get_default_config()
        config.training.distributed_training = True

        module = MockModule('TestModel', device, config)
        engine = Engine(module, config)

        assert engine.distributed_training == True
        assert engine.driver_device == True  # Rank 0


class TestEngineAutoSetup:
    """Test engine auto-setup functionality."""

    def test_ensure_model_ready_builds_model(self, device, mock_config):
        """Test that engine auto-builds model if needed."""
        module = MockModule('TestModel', device, mock_config)
        engine = Engine(module, mock_config)

        assert not module.is_built

        engine._ensure_model_ready_for_training()

        assert module.is_built
        assert module.is_optimizer_set

    def test_ensure_model_ready_skips_if_built(self, device, mock_config):
        """Test that engine doesn't rebuild if already built."""
        module = MockModule('TestModel', device, mock_config)
        module.build()
        module.associate_optimizer()

        engine = Engine(module, mock_config)

        # Should not raise or rebuild
        engine._ensure_model_ready_for_training()

        assert module.is_built
        assert module.is_optimizer_set


class TestEngineFit:
    """Test engine fit method."""

    def test_fit_with_dataloaders(self, device, mock_config):
        """Test fit with train and val dataloaders."""
        # Create simple datasets
        x = torch.randn(32, 10)
        y = torch.randn(32, 10)
        train_dataset = TensorDataset(x, y)
        val_dataset = TensorDataset(x, y)

        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        # Create module and engine
        module = MockModule('TestModel', device, mock_config)
        engine = Engine(module, mock_config)

        # Run fit
        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=2,
            callbacks=[],
            train_verbosity_level=0,
            val_verbosity_level=0
        )

        # Check training happened
        assert module._training_step_called > 0
        assert module._validation_step_called > 0
        assert engine.current_epoch == 2

    def test_fit_auto_builds_model(self, device, mock_config):
        """Test that fit auto-builds model."""
        x = torch.randn(16, 10)
        y = torch.randn(16, 10)
        train_dataset = TensorDataset(x, y)
        val_dataset = TensorDataset(x, y)

        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        module = MockModule('TestModel', device, mock_config)
        engine = Engine(module, mock_config)

        assert not module.is_built

        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=1,
            callbacks=[],
            train_verbosity_level=0,
            val_verbosity_level=0
        )

        assert module.is_built
        assert module.is_optimizer_set

    def test_fit_uses_config_epochs_if_not_specified(self, device, mock_config):
        """Test that fit uses config epochs when max_epochs=None."""
        x = torch.randn(16, 10)
        y = torch.randn(16, 10)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=8)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=8)

        mock_config.training.n_train_epochs = 3

        module = MockModule('TestModel', device, mock_config)
        engine = Engine(module, mock_config)

        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=None,  # Use config value
            callbacks=[],
            train_verbosity_level=0,
            val_verbosity_level=0
        )

        assert engine.current_epoch == 3


class TestEngineLoadWeights:
    """Test engine weight loading."""

    def test_load_weights_convenience_method(self, device, mock_config, temp_dir):
        """Test engine's load_weights convenience method."""
        import os

        # Create and save weights
        module1 = MockModule('TestModel', device, mock_config)
        module1.build()
        weight_path = os.path.join(temp_dir, 'weights.pt')
        module1.save_weights(weight_path)

        # Load weights through engine
        module2 = MockModule('TestModel', device, mock_config)
        module2.build()
        engine = Engine(module2, mock_config)

        engine.load_weights(weight_path)

        assert module2.is_weights_loaded


class TestEngineCallbacks:
    """Test engine callback system."""

    def test_fit_calls_callback_hooks(self, device, mock_config):
        """Test that callbacks are called during training."""
        x = torch.randn(16, 10)
        y = torch.randn(16, 10)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=8)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=8)

        # Create mock callback
        mock_callback = Mock()
        mock_callback.on_train_begin = Mock()
        mock_callback.on_epoch_begin = Mock()
        mock_callback.on_epoch_end = Mock()
        mock_callback.on_train_end = Mock()

        module = MockModule('TestModel', device, mock_config)
        engine = Engine(module, mock_config)

        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=2,
            callbacks=[mock_callback],
            train_verbosity_level=0,
            val_verbosity_level=0
        )

        # Verify callbacks were called
        mock_callback.on_train_begin.assert_called_once()
        assert mock_callback.on_epoch_begin.call_count == 2
        assert mock_callback.on_epoch_end.call_count == 2
        mock_callback.on_train_end.assert_called_once()
