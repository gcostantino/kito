"""
Unit tests for preprocessing classes.

Tests all preprocessing transformations.
"""
import pytest
import numpy as np
import torch
from kito.data.preprocessing import (
    Preprocessing,
    Pipeline,
    Normalize,
    Standardization,
    ClipOutliers,
    Detrend,
    AddNoise,
    LogTransform,
    ToTensor
)


class TestNormalize:
    """Test Normalize preprocessing."""

    def test_normalize_default_range(self):
        """Test normalization to [0, 1]."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0, 1, 2, 3, 4])

        norm = Normalize(min_val=0.0, max_val=1.0)
        data_norm, labels_norm = norm(data, labels)

        assert data_norm.min() == 0.0
        assert data_norm.max() == 1.0
        assert np.array_equal(labels, labels_norm)  # Labels unchanged

    def test_normalize_custom_range(self):
        """Test normalization to custom range."""
        data = np.array([0.0, 10.0])
        labels = np.array([0, 1])

        norm = Normalize(min_val=-1.0, max_val=1.0)
        data_norm, _ = norm(data, labels)

        assert data_norm.min() == -1.0
        assert data_norm.max() == 1.0

    def test_normalize_constant_data(self):
        """Test normalization with constant data."""
        data = np.ones(10)
        labels = np.zeros(10)

        norm = Normalize()
        data_norm, _ = norm(data, labels)

        # Should remain unchanged (no division by zero)
        assert np.array_equal(data, data_norm)


class TestStandardization:
    """Test Standardization preprocessing."""

    def test_standardization_compute_from_data(self):
        """Test computing mean/std from data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0, 1, 2, 3, 4])

        std = Standardization()
        data_std, _ = std(data, labels)

        # After standardization, mean should be ~0, std should be ~1
        assert np.abs(data_std.mean()) < 1e-6
        assert np.abs(data_std.std() - 1.0) < 1e-6

    def test_standardization_with_fixed_params(self):
        """Test standardization with fixed mean/std."""
        data = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 1, 2])

        std = Standardization(mean=2.0, std=1.0)
        data_std, _ = std(data, labels)

        expected = (data - 2.0) / 1.0
        assert np.allclose(data_std, expected)

    def test_standardization_preserves_labels(self):
        """Test that labels are not modified."""
        data = np.random.randn(10, 5)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        std = Standardization()
        _, labels_out = std(data, labels)

        assert np.array_equal(labels, labels_out)


class TestClipOutliers:
    """Test ClipOutliers preprocessing."""

    def test_clip_outliers_default(self):
        """Test clipping outliers at 3 std."""
        # Create data with outliers
        data = np.array([1.0, 2.0, 3.0, 100.0, -100.0])
        labels = np.zeros(5)

        clip = ClipOutliers(n_std=3.0)
        data_clipped, _ = clip(data, labels)

        # Outliers should be clipped
        assert data_clipped.max() < 100.0
        assert data_clipped.min() > -100.0

    def test_clip_outliers_torch(self):
        """Test clipping with torch tensors."""
        data = torch.tensor([1.0, 2.0, 3.0, 100.0, -100.0])
        labels = torch.zeros(5)

        clip = ClipOutliers(n_std=2.0)
        data_clipped, _ = clip(data, labels)

        assert isinstance(data_clipped, torch.Tensor)
        assert data_clipped.max() < 100.0


class TestDetrend:
    """Test Detrend preprocessing."""

    def test_detrend_removes_mean(self):
        """Test that detrend removes mean."""
        data = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        labels = np.zeros(5)

        detrend = Detrend()
        data_detrended, _ = detrend(data, labels)

        # Mean should be close to zero
        assert np.abs(data_detrended.mean()) < 1e-10

    def test_detrend_with_axis(self):
        """Test detrending along specific axis."""
        data = np.random.randn(10, 5, 5) + 10.0  # Add offset
        labels = np.zeros((10, 5, 5))

        detrend = Detrend(axis=0)
        data_detrended, _ = detrend(data, labels)

        # Check shape preserved
        assert data_detrended.shape == data.shape


class TestAddNoise:
    """Test AddNoise preprocessing."""

    def test_add_noise_changes_data(self):
        """Test that noise is added to data."""
        data = np.zeros((10, 5))
        labels = np.zeros((10, 5))

        noise = AddNoise(std=0.1, mean=0.0)
        data_noisy, _ = noise(data, labels)

        # Data should no longer be all zeros
        assert not np.array_equal(data, data_noisy)

    def test_add_noise_torch(self):
        """Test adding noise to torch tensors."""
        data = torch.zeros(10, 5)
        labels = torch.zeros(10, 5)

        noise = AddNoise(std=0.1)
        data_noisy, _ = noise(data, labels)

        assert isinstance(data_noisy, torch.Tensor)
        assert not torch.equal(data, data_noisy)


class TestLogTransform:
    """Test LogTransform preprocessing."""

    def test_log_transform_natural(self):
        """Test natural log transform."""
        data = np.array([1.0, 2.0, 3.0])
        labels = np.zeros(3)

        log_tf = LogTransform(offset=0.0, base='e')
        data_log, _ = log_tf(data, labels)

        expected = np.log(data)
        assert np.allclose(data_log, expected)

    def test_log_transform_base10(self):
        """Test log10 transform."""
        data = np.array([1.0, 10.0, 100.0])
        labels = np.zeros(3)

        log_tf = LogTransform(offset=0.0, base='10')
        data_log, _ = log_tf(data, labels)

        expected = np.log10(data)
        assert np.allclose(data_log, expected)

    def test_log_transform_with_offset(self):
        """Test log transform with offset."""
        data = np.array([0.0, 1.0, 2.0])
        labels = np.zeros(3)

        log_tf = LogTransform(offset=1.0, base='e')
        data_log, _ = log_tf(data, labels)

        expected = np.log(data + 1.0)
        assert np.allclose(data_log, expected)


class TestToTensor:
    """Test ToTensor preprocessing."""

    def test_to_tensor_from_numpy(self):
        """Test converting numpy to tensor."""
        data = np.random.randn(10, 5).astype(np.float32)
        labels = np.random.randn(10, 5).astype(np.float32)

        to_tensor = ToTensor(dtype=torch.float32)
        data_tensor, labels_tensor = to_tensor(data, labels)

        assert isinstance(data_tensor, torch.Tensor)
        assert isinstance(labels_tensor, torch.Tensor)
        assert data_tensor.dtype == torch.float32

    def test_to_tensor_already_tensor(self):
        """Test that tensors pass through unchanged."""
        data = torch.randn(10, 5)
        labels = torch.randn(10, 5)

        to_tensor = ToTensor()
        data_out, labels_out = to_tensor(data, labels)

        assert isinstance(data_out, torch.Tensor)
        assert torch.equal(data, data_out)


class TestPipeline:
    """Test Pipeline preprocessing."""

    def test_pipeline_empty(self):
        """Test empty pipeline."""
        data = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 1, 2])

        pipeline = Pipeline([])
        data_out, labels_out = pipeline(data, labels)

        assert np.array_equal(data, data_out)
        assert np.array_equal(labels, labels_out)

    def test_pipeline_single_step(self):
        """Test pipeline with single step."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.zeros(5)

        pipeline = Pipeline([Normalize()])
        data_out, _ = pipeline(data, labels)

        assert data_out.min() == 0.0
        assert data_out.max() == 1.0

    def test_pipeline_multiple_steps(self):
        """Test pipeline with multiple steps."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.zeros(5)

        pipeline = Pipeline([
            Normalize(),
            Standardization(mean=0.5, std=0.2),
            ToTensor()
        ])
        data_out, labels_out = pipeline(data, labels)

        assert isinstance(data_out, torch.Tensor)
        assert isinstance(labels_out, torch.Tensor)

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        pipeline = Pipeline([
            Normalize(),
            Standardization(),
            ToTensor()
        ])

        repr_str = repr(pipeline)
        assert 'Pipeline' in repr_str
        assert 'Normalize' in repr_str
        assert 'Standardization' in repr_str
        assert 'ToTensor' in repr_str
