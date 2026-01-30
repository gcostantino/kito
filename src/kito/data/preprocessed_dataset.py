"""
PreprocessedDataset - Wraps a dataset and applies preprocessing.

Separates data loading from preprocessing for flexibility.

Example:
    raw_dataset = H5Dataset('data.h5')
    preprocessing = Pipeline([Detrend(), Standardization()])
    dataset = PreprocessedDataset(raw_dataset, preprocessing)
"""
from typing import Union, List

from torch.utils.data import Dataset

from kito.data.preprocessing import Preprocessing, PREPROCESSING, Pipeline
from kito.config.moduleconfig import PreprocessingStepConfig


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

    def __init__(self, base_dataset: Dataset, preprocessing=Union[List[PreprocessingStepConfig], List[Preprocessing]]):
        self.base_dataset = base_dataset
        self.preprocessing = preprocessing

        # build preprocessing pipeline from configs
        if preprocessing is not None:
            self.preprocessing = self._build_pipeline(preprocessing)
        else:
            self.preprocessing = None

    def _build_pipeline(self, preprocessing_configs):
        """
        Convert list of PreprocessingStepConfig to Pipeline.

        Args:
            preprocessing_configs: List of PreprocessingStepConfig or dicts

        Returns:
            Pipeline instance with instantiated preprocessing steps
        """
        steps = []

        for config in preprocessing_configs:
            if isinstance(config, PreprocessingStepConfig):
                # create instance from config using registry
                step_instance = PREPROCESSING.create(
                    config.name,
                    **config.params
                )
                steps.append(step_instance)

            elif isinstance(config, dict):
                # Also support dict format
                name = config['name']
                params = config.get('params', {})
                step_instance = PREPROCESSING.create(name, **params)
                steps.append(step_instance)

            elif isinstance(config, Preprocessing):
                # Already an instance
                steps.append(config)

            else:
                raise TypeError(f"Invalid preprocessing config: {type(config)}")

        # return Pipeline with instantiated steps
        return Pipeline(steps)

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
