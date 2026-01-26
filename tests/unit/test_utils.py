"""
Unit tests for utility functions.

Tests gpu_utils and loss_utils.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from kito.utils.gpu_utils import assign_device
from kito.utils.loss_utils import get_loss


class TestGPUUtils:
    """Test GPU utility functions."""

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_assign_device_cuda_available(self, mock_mps, mock_cuda):
        """Test device assignment when CUDA is available."""
        mock_cuda.return_value = True
        mock_mps.return_value = False

        device = assign_device(gpu_id=0)

        assert device.type == 'cuda'
        assert device.index == 0

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_assign_device_mps_available(self, mock_mps, mock_cuda):
        """Test device assignment when MPS is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = True

        device = assign_device(gpu_id=0)

        assert device.type == 'mps'

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_assign_device_cpu_fallback(self, mock_mps, mock_cuda):
        """Test device assignment falls back to CPU."""
        mock_cuda.return_value = False
        mock_mps.return_value = False

        device = assign_device(gpu_id=0)

        assert device.type == 'cpu'

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_assign_device_multiple_gpus(self, mock_mps, mock_cuda):
        """Test device assignment with multiple GPUs."""
        mock_cuda.return_value = True
        mock_mps.return_value = False

        device0 = assign_device(gpu_id=0)
        device1 = assign_device(gpu_id=1)

        assert device0.index == 0
        assert device1.index == 1


class TestLossUtils:
    """Test loss utility functions."""

    def test_get_loss_mse(self):
        """Test getting MSE loss."""
        loss_fn = get_loss('mean_squared_error')

        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_mae(self):
        """Test getting MAE loss."""
        loss_fn = get_loss('mean_absolute_error')

        assert isinstance(loss_fn, nn.L1Loss)

    def test_get_loss_cross_entropy(self):
        """Test getting cross entropy loss."""
        loss_fn = get_loss('cross_entropy_loss')

        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_loss_bce(self):
        """Test getting binary cross entropy loss."""
        loss_fn = get_loss('binary_cross_entropy_loss')

        assert isinstance(loss_fn, nn.BCELoss)

    def test_get_loss_case_insensitive(self):
        """Test that loss names are case-insensitive."""
        loss_fn1 = get_loss('mean_squared_error')
        loss_fn2 = get_loss('MEAN_SQUARED_ERROR')
        loss_fn3 = get_loss('Mean_Squared_Error')

        assert type(loss_fn1) == type(loss_fn2) == type(loss_fn3)

    def test_get_loss_ssim(self):
        """Test getting SSIM loss (custom loss)."""
        loss_fn = get_loss('ssim_loss')

        # Should return a callable
        assert callable(loss_fn)

    def test_get_loss_invalid(self):
        """Test that invalid loss name raises error."""
        with pytest.raises(ValueError, match="Loss .* not valid"):
            get_loss('invalid_loss_name')

    def test_get_loss_empty_string(self):
        """Test that empty string raises error."""
        with pytest.raises(AssertionError):
            get_loss('')

    def test_loss_functions_work(self):
        """Test that returned loss functions actually work."""
        loss_fn = get_loss('mean_squared_error')

        pred = torch.randn(10, 5)
        target = torch.randn(10, 5)

        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # MSE is non-negative
