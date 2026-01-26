"""
PreprocessedDataset - Wraps a dataset and applies preprocessing.

Separates data loading from preprocessing for flexibility.

Example:
    raw_dataset = H5Dataset('data.h5')
    preprocessing = Pipeline([Detrend(), Standardization()])
    dataset = PreprocessedDataset(raw_dataset, preprocessing)
"""
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    Wraps a base dataset and applies preprocessing on-the-fly.

    This allows:
    - Keeping datasets "dumb" (just load data)
    - Composable preprocessing
    - Easy experimentation (swap preprocessing without changing dataset)

    Args:
        base_dataset: Underlying dataset (H5Dataset, MemDataset, etc.)
        preprocessing: Preprocessing instance or None

    Example:
        >>> raw_dataset = H5Dataset('data.h5')
        >>> preprocessing = Standardization(mean=0, std=1)
        >>> dataset = PreprocessedDataset(raw_dataset, preprocessing)
        >>> data, labels = dataset[0]  # Automatically preprocessed
    """

    def __init__(self, base_dataset: Dataset, preprocessing=None):
        self.base_dataset = base_dataset
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        # Load raw data
        data, labels = self.base_dataset[index]

        # Apply preprocessing if specified
        if self.preprocessing is not None:
            data, labels = self.preprocessing(data, labels)

        return data, labels

    def __len__(self):
        return len(self.base_dataset)

    def __repr__(self):
        return (
            f"PreprocessedDataset(\n"
            f"  base_dataset={self.base_dataset},\n"
            f"  preprocessing={self.preprocessing}\n"
            f")"
        )
