"""
Unit tests for KitoModule.

Tests the base module class that users extend.
"""
import pytest
import torch
import torch.nn as nn
from kito.module import KitoModule
from tests.fixtures.mock_config import get_default_config


class SimpleTestModule(KitoModule):
    """Simple test module for unit tests."""

    def build_inner_model(self):
        """Build a simple sequential model."""
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        self.model_input_size = (1, 32, 32)
        self.standard_data_shape = (1, 32, 32)

    def bind_optimizer(self):
        """Bind a simple optimizer."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )


class TestKitoModule:
    """Test KitoModule base class."""

    def test_module_initialization(self, device, mock_config):
        """Test module initialization."""
        module = SimpleTestModule('TestModel', device, mock_config)

        assert module.model_name == 'TestModel'
        assert module.device == device
        assert module.config == mock_config
        assert module.learning_rate == mock_config.training.learning_rate
        assert module.batch_size == mock_config.training.batch_size

    def test_module_initialization_no_config(self, device):
        """Test module initialization without config."""
        module = SimpleTestModule('TestModel', device, None)

        assert module.model_name == 'TestModel'
        assert module.device == device
        assert module.config is None
        assert module.learning_rate is None
        assert module.batch_size is None

    def test_module_build(self, device, mock_config):
        """Test building the model."""
        module = SimpleTestModule('TestModel', device, mock_config)

        assert not module.is_built
        assert module.model is None

        module.build()

        assert module.is_built
        assert module.model is not None
        assert isinstance(module.model, nn.Module)

    def test_module_build_moves_to_device(self, device, mock_config):
        """Test that build moves model to device."""
        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()

        # Check that model is on correct device
        for param in module.model.parameters():
            assert param.device == device

    def test_module_associate_optimizer(self, device, mock_config):
        """Test setting up optimizer."""
        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()

        assert not module.is_optimizer_set
        assert module.optimizer is None

        module.associate_optimizer()

        assert module.is_optimizer_set
        assert module.optimizer is not None
        assert isinstance(module.optimizer, torch.optim.Optimizer)

    def test_module_associate_optimizer_before_build_fails(self, device, mock_config):
        """Test that setting optimizer before build raises error."""
        module = SimpleTestModule('TestModel', device, mock_config)

        with pytest.raises(RuntimeError, match="Must call build\\(\\) before"):
            module.associate_optimizer()

    def test_module_training_step(self, device, mock_config):
        """Test training step."""
        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()
        module.associate_optimizer()

        # Create a batch
        inputs = torch.randn(4, 1, 32, 32).to(device)
        targets = torch.randn(4, 1, 32, 32).to(device)
        batch = (inputs, targets)

        # Run training step
        output = module.training_step(batch)

        assert 'loss' in output
        assert isinstance(output['loss'], torch.Tensor)
        assert output['loss'].ndim == 0  # Scalar

    def test_module_validation_step(self, device, mock_config):
        """Test validation step."""
        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()

        # Create a batch
        inputs = torch.randn(4, 1, 32, 32).to(device)
        targets = torch.randn(4, 1, 32, 32).to(device)
        batch = (inputs, targets)

        # Run validation step
        with torch.no_grad():
            output = module.validation_step(batch)

        assert 'loss' in output
        assert 'outputs' in output
        assert 'targets' in output
        assert 'inputs' in output

    def test_module_prediction_step(self, device, mock_config):
        """Test prediction step."""
        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()

        # Create input batch
        inputs = torch.randn(4, 1, 32, 32).to(device)
        batch = (inputs,)

        # Run prediction step
        with torch.no_grad():
            outputs = module.prediction_step(batch)

        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape[0] == 4  # Batch size

    def test_module_load_weights(self, device, mock_config, temp_dir):
        """Test loading weights."""
        import os

        # Build and save weights
        module1 = SimpleTestModule('TestModel', device, mock_config)
        module1.build()
        weight_path = os.path.join(temp_dir, 'test_weights.pt')
        module1.save_weights(weight_path)

        # Create new module and load weights
        module2 = SimpleTestModule('TestModel', device, mock_config)
        module2.build()

        assert not module2.is_weights_loaded

        module2.load_weights(weight_path)

        assert module2.is_weights_loaded

    def test_module_load_weights_nonexistent_file(self, device, mock_config):
        """Test loading weights from nonexistent file raises error."""
        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()

        with pytest.raises(FileNotFoundError):
            module.load_weights('/nonexistent/weights.pt')

    def test_module_save_weights(self, device, mock_config, temp_dir):
        """Test saving weights."""
        import os

        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()

        weight_path = os.path.join(temp_dir, 'saved_weights.pt')
        module.save_weights(weight_path)

        assert os.path.exists(weight_path)

    def test_module_get_sample_input(self, device, mock_config):
        """Test getting sample input."""
        module = SimpleTestModule('TestModel', device, mock_config)
        module.build()

        sample = module.get_sample_input()

        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (1, 1, 32, 32)  # (batch=1, channels, H, W)
        assert sample.device == device

    def test_module_get_sample_input_before_build_fails(self, device, mock_config):
        """Test that getting sample input before build raises error."""
        module = SimpleTestModule('TestModel', device, mock_config)

        with pytest.raises(ValueError, match="model_input_size not set"):
            module.get_sample_input()

    def test_module_state_properties(self, device, mock_config):
        """Test module state properties."""
        module = SimpleTestModule('TestModel', device, mock_config)

        # Initial state
        assert not module.is_built
        assert not module.is_optimizer_set
        assert not module.is_weights_loaded

        # After build
        module.build()
        assert module.is_built
        assert not module.is_optimizer_set

        # After optimizer
        module.associate_optimizer()
        assert module.is_optimizer_set


class TestKitoModuleAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_build_inner_model_not_implemented(self, device, mock_config):
        """Test that build_inner_model must be implemented."""

        class IncompleteModule(KitoModule):
            def bind_optimizer(self):
                pass

        module = IncompleteModule('Test', device, mock_config)

        with pytest.raises(NotImplementedError):
            module.build()

    def test_bind_optimizer_not_implemented(self, device, mock_config):
        """Test that bind_optimizer must be implemented."""

        class IncompleteModule(KitoModule):
            def build_inner_model(self):
                self.model = nn.Linear(10, 10)
                self.model_input_size = (10,)

        module = IncompleteModule('Test', device, mock_config)
        module.build()

        with pytest.raises(NotImplementedError):
            module.associate_optimizer()
