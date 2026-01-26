"""
Unit tests for dataset classes.

Tests H5Dataset, MemDataset, and PreprocessedDataset.
"""
import pytest
import numpy as np
import torch
from kito.data.datasets import H5Dataset, MemDataset, KitoDataset
from kito.data.preprocessed_dataset import PreprocessedDataset
from kito.data.preprocessing import Normalize


class TestMemDataset:
    """Test MemDataset class."""

    def test_create_memdataset(self, sample_numpy_data):
        """Test creating a MemDataset."""
        x, y = sample_numpy_data
        dataset = MemDataset(x, y)

        assert len(dataset) == len(x)

    def test_memdataset_getitem(self, sample_numpy_data):
        """Test getting items from MemDataset."""
        x, y = sample_numpy_data
        dataset = MemDataset(x, y)

        data, label = dataset[0]
        assert data.shape == x[0].shape
        assert label.shape == y[0].shape
        assert np.array_equal(data, x[0])
        assert np.array_equal(label, y[0])

    def test_memdataset_length_mismatch(self):
        """Test that mismatched x and y raise error."""
        x = np.random.randn(100, 1, 32, 32)
        y = np.random.randn(50, 1, 32, 32)  # Different length

        with pytest.raises(ValueError, match="x and y must have same length"):
            MemDataset(x, y)

    def test_memdataset_iteration(self, sample_numpy_data):
        """Test iterating over MemDataset."""
        x, y = sample_numpy_data
        dataset = MemDataset(x, y)

        count = 0
        for data, label in dataset:
            count += 1
            assert data.shape == x[0].shape

        assert count == len(dataset)


class TestH5Dataset:
    """Test H5Dataset class."""

    def test_create_h5dataset(self, temp_h5_file):
        """Test creating an H5Dataset."""
        dataset = H5Dataset(temp_h5_file)
        assert dataset.dataset_len > 0

    def test_h5dataset_getitem(self, temp_h5_file):
        """Test getting items from H5Dataset."""
        dataset = H5Dataset(temp_h5_file)

        data, label = dataset[0]
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)

    def test_h5dataset_lazy_loading(self, temp_h5_file):
        """Test that H5Dataset uses lazy loading."""
        dataset = H5Dataset(temp_h5_file)

        # Before accessing data, datasets should be None
        assert dataset.dataset_data is None
        assert dataset.dataset_labels is None

        # After accessing, should be loaded
        _, _ = dataset[0]
        assert dataset.dataset_data is not None
        assert dataset.dataset_labels is not None

    def test_h5dataset_len(self, temp_h5_file):
        """Test H5Dataset length."""
        dataset = H5Dataset(temp_h5_file)
        assert len(dataset) == 100  # From fixture

    def test_h5dataset_nonexistent_file(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(Exception):  # h5py raises OSError
            dataset = H5Dataset('/nonexistent/file.h5')
            _ = dataset[0]  # Trigger lazy loading

    def test_h5dataset_pickling(self, temp_h5_file):
        """Test H5Dataset can be pickled (for multiprocessing)."""
        import pickle

        dataset = H5Dataset(temp_h5_file)

        # Should be picklable
        pickled = pickle.dumps(dataset)
        unpickled = pickle.loads(pickled)

        assert unpickled.file_path == dataset.file_path


class TestPreprocessedDataset:
    """Test PreprocessedDataset wrapper."""

    def test_preprocessed_dataset_no_preprocessing(self, sample_numpy_data):
        """Test PreprocessedDataset without preprocessing."""
        x, y = sample_numpy_data
        base_dataset = MemDataset(x, y)
        dataset = PreprocessedDataset(base_dataset, preprocessing=None)

        data, label = dataset[0]
        base_data, base_label = base_dataset[0]

        assert np.array_equal(data, base_data)
        assert np.array_equal(label, base_label)

    def test_preprocessed_dataset_with_preprocessing(self, sample_numpy_data):
        """Test PreprocessedDataset with preprocessing."""
        x, y = sample_numpy_data
        base_dataset = MemDataset(x, y)
        preprocessing = Normalize(min_val=0.0, max_val=1.0)
        dataset = PreprocessedDataset(base_dataset, preprocessing)

        data, label = dataset[0]

        # Data should be normalized
        assert data.min() >= 0.0
        assert data.max() <= 1.0

    def test_preprocessed_dataset_len(self, sample_numpy_data):
        """Test PreprocessedDataset length."""
        x, y = sample_numpy_data
        base_dataset = MemDataset(x, y)
        dataset = PreprocessedDataset(base_dataset, Normalize())

        assert len(dataset) == len(base_dataset)

    def test_preprocessed_dataset_repr(self, sample_numpy_data):
        """Test PreprocessedDataset string representation."""
        x, y = sample_numpy_data
        base_dataset = MemDataset(x, y)
        preprocessing = Normalize()
        dataset = PreprocessedDataset(base_dataset, preprocessing)

        repr_str = repr(dataset)
        assert 'PreprocessedDataset' in repr_str
