"""
Unit tests for Registry pattern.

Tests the registry system used for datasets and preprocessing.
"""
import pytest

from kito.data.registry import Registry, DATASETS, PREPROCESSING


class TestRegistry:
    """Test Registry class."""

    def test_registry_creation(self):
        """Test creating a registry."""
        reg = Registry('TEST_REGISTRY')
        assert reg.name == 'TEST_REGISTRY'
        assert len(reg._registry) == 0

    def test_register_class(self):
        """Test registering a class."""
        reg = Registry('TEST')

        @reg.register('myclass')
        class MyClass:
            pass

        assert 'myclass' in reg
        assert reg.get('myclass') == MyClass

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate name raises error."""
        reg = Registry('TEST')

        @reg.register('duplicate')
        class Class1:
            pass

        with pytest.raises(ValueError, match="'duplicate' already registered"):
            @reg.register('duplicate')
            class Class2:
                pass

    def test_get_nonexistent_raises_error(self):
        """Test that getting non-existent name raises error."""
        reg = Registry('TEST')

        with pytest.raises(KeyError, match="'nonexistent' not found"):
            reg.get('nonexistent')

    def test_list_registered(self):
        """Test listing all registered names."""
        reg = Registry('TEST')

        @reg.register('class1')
        class Class1:
            pass

        @reg.register('class2')
        class Class2:
            pass

        registered = reg.list_registered()
        assert 'class1' in registered
        assert 'class2' in registered
        assert len(registered) == 2

    def test_contains(self):
        """Test __contains__ method."""
        reg = Registry('TEST')

        @reg.register('exists')
        class ExistsClass:
            pass

        assert 'exists' in reg
        assert 'notexists' not in reg


class TestGlobalRegistries:
    """Test global DATASETS and PREPROCESSING registries."""

    def test_datasets_registry_exists(self):
        """Test that DATASETS registry is initialized."""
        assert DATASETS.name == 'DATASETS'
        assert 'h5dataset' in DATASETS
        assert 'memdataset' in DATASETS

    def test_preprocessing_registry_exists(self):
        """Test that PREPROCESSING registry is initialized."""
        assert PREPROCESSING.name == 'PREPROCESSING'
        assert 'normalize' in PREPROCESSING
        assert 'standardization' in PREPROCESSING
        assert 'pipeline' in PREPROCESSING

    def test_can_get_registered_datasets(self):
        """Test retrieving registered datasets."""
        from kito.data.datasets import H5Dataset, MemDataset

        assert DATASETS.get('h5dataset') == H5Dataset
        assert DATASETS.get('memdataset') == MemDataset

    def test_can_get_registered_preprocessing(self):
        """Test retrieving registered preprocessing."""
        from kito.data.preprocessing import Normalize, Standardization

        assert PREPROCESSING.get('normalize') == Normalize
        assert PREPROCESSING.get('standardization') == Standardization
