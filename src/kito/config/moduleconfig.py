"""
Base configuration system for KitoModule framework.

Users can extend these base configs with their own custom parameters.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union


@dataclass
class PreprocessingStepConfig:
    """
    Configuration for a single preprocessing step.

    Args:
        type: Name of preprocessing class (e.g., 'detrend', 'standardization')
        params: Dictionary of parameters to pass to preprocessing class

    Example:
        >>> step = PreprocessingStepConfig(
        ...     type='standardization',
        ...     params={'mean': 0.5, 'std': 0.2}
        ... )
    """
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """
    Data loading and preprocessing configuration.

    This config defines:
    - What dataset to use (H5, memory, custom)
    - Where data is located
    - How to initialize the dataset (flexible args)
    - Memory loading strategy
    - How to split data (train/val/test)
    - What preprocessing to apply
    - DataLoader settings
    """

    # Dataset configuration
    dataset_type: str = 'h5dataset'  # 'h5dataset', 'memdataset', or custom

    # Simple path (backward compatible)
    dataset_path: str = ''

    # Flexible initialization args (for custom datasets)
    # If provided, takes precedence over dataset_path
    # Allows any constructor signature: Dataset(**dataset_init_args)
    dataset_init_args: Dict[str, Any] = field(default_factory=dict)

    # Memory management
    load_into_memory: bool = False  # Load entire dataset into RAM for faster training

    # Splitting ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is implicit: 1 - train_ratio - val_ratio

    # Total samples to use (None = use all available)
    total_samples: Optional[int] = None

    # Preprocessing pipeline
    # List of preprocessing steps applied in order
    preprocessing: List[PreprocessingStepConfig] = field(default_factory=list)

    # DataLoader settings
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclass
class TrainingConfig:
    """Core training parameters required by KitoModule."""

    # Essential training parameters
    learning_rate: float
    n_train_epochs: int
    batch_size: int
    train_mode: bool  # True for training, False for inference

    # Verbosity (0=silent, 1=progress bar, 2=detailed)
    train_verbosity_level: int = 2
    val_verbosity_level: int = 2
    test_verbosity_level: int = 2

    # Distributed training
    distributed_training: bool = False
    master_gpu_id: int = 0  # Only used if not distributed

    # Weight initialization
    initialize_model_with_saved_weights: bool = False

    # device type initialization
    device_type: str = "cuda"  # "cuda", "mps", or "cpu"

    def __post_init__(self):
        """Validate device_type after initialization."""
        valid_devices = {"cuda", "mps", "cpu"}

        if self.device_type not in valid_devices:
            raise ValueError(
                f"Invalid device_type: '{self.device_type}'. "
                f"Must be one of {valid_devices}."
            )

        # Normalize to lowercase (user-friendly)
        self.device_type = self.device_type.lower()


@dataclass
class ModelConfig:
    """Core model parameters required by KitoModule.

    The loss definition supports:
    1. Simple string: 'mse'
    2. Dict with name: {'name': 'mse'}
    3. Dict with params: {'name': 'weighted_mse', 'params': {'weight': 2.0}}
    4. Dict with inline params: {'name': 'weighted_mse', 'weight': 2.0}"""

    # Data dimensions
    input_data_size: Tuple[int, ...]  # Flexible shape

    # Loss and optimization
    loss: Union[str, dict] = field(default_factory=dict)

    # Callbacks and logging
    #log_to_tensorboard: bool = False
    #save_model_weights: bool = False
    #text_logging: bool = False
    #csv_logging: bool = False
    train_codename: str = "experiment"

    # Weights
    weight_load_path: str = ""

    # Inference
    save_inference_to_disk: bool = False
    inference_filename: str = ""

    # TensorBoard visualization (optional)
    tensorboard_img_id: str = "training_viz"
    batch_idx_viz: List[int] = field(default_factory=lambda: [0])


@dataclass
class WorkDirConfig:
    """Working directory configuration."""
    work_directory: str


@dataclass
class CallbacksConfig:
    """
    Configuration for Kito's built-in default callbacks.

    For custom callbacks, use instead:
        callbacks = engine.get_default_callbacks()
        callbacks.append(MyCustomCallback())
        engine.fit(..., callbacks=callbacks)
    """

    # === CSV Logger ===
    enable_csv_logger: bool = True

    # === Text Logger ===
    enable_text_logger: bool = True

    # === Model Checkpoint ===
    enable_model_checkpoint: bool = True
    checkpoint_monitor: str = 'val_loss'
    checkpoint_mode: str = 'min'  # 'min' or 'max'
    checkpoint_save_best_only: bool = True
    checkpoint_verbose: bool = False

    # === TensorBoard ===
    enable_tensorboard: bool = False  # Master switch
    tensorboard_scalars: bool = True
    tensorboard_histograms: bool = True
    tensorboard_histogram_freq: int = 5
    tensorboard_graph: bool = True
    tensorboard_images: bool = False
    tensorboard_image_freq: int = 1
    tensorboard_batch_indices: List[int] = field(default_factory=lambda: [0])


@dataclass
class KitoModuleConfig:
    """
    Base configuration container for KitoModule.

    Contains all configuration sections:
    - training: Training parameters
    - model: Model architecture and settings
    - workdir: Output directories
    - data: Dataset and preprocessing
    """
    training: TrainingConfig
    model: ModelConfig
    workdir: WorkDirConfig
    data: DataConfig
    callbacks: CallbacksConfig
