import datetime
import os
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.callbacks.callback_base import Callback, CallbackList
from src.callbacks.ddp_aware_callback import DDPAwareCallback
from src.data.datapipeline import GenericDataPipeline
from src.module import KitoModule
from src.strategies.logger_strategy import DDPLogger, DefaultLogger
from src.strategies.progress_bar_strategy import (
    StandardProgressBarHandler,
    DDPProgressBarHandler
)
from src.strategies.readiness_validator import ReadinessValidator
from src.utils.gpu_utils import assign_device


class Engine:
    """
    Engine for training, validation, and inference.

    HYBRID APPROACH:
    - Auto-builds model if not built (user can also build explicitly)
    - Auto-sets optimizer if not set (user can also set explicitly)

    This provides:
    1. Simplicity for beginners (just works)
    2. Control for advanced users (explicit calls)
    3. Flexibility for power users (custom args, modifications)

    The Engine:
    1. Manages device assignment and DDP
    2. Iterates over batches and epochs
    3. Calls module's single-batch methods (training_step, validation_step, prediction_step)
    4. Manages callbacks (creates defaults, registers hooks)
    5. Handles logging and progress bars

    Args:
        module: BaseModule instance
        config: Configuration object (for device, DDP, callbacks, etc.)

    Example (Simple - auto-build):
        module = MyModel('MyModel', device, config)
        engine = Engine(module, config)
        engine.fit(train_loader, val_loader, max_epochs=100)  # Auto-builds and sets optimizer

    Example (Advanced - explicit control):
        module = MyModel('MyModel', device, config)
        module.build(custom_layers=64)  # Custom build
        module.summary()  # Inspect
        module.associate_optimizer()  # Custom optimizer setup

        engine = Engine(module, config)
        engine.fit(train_loader, val_loader, max_epochs=100)  # Uses pre-built model
    """

    def __init__(self, module: KitoModule, config):
        """
        Initialize Engine.

        Args:
            module: BaseModule instance (can be built or not)
            config: Configuration object
        """
        self.module = module
        self.config = config

        # Extract config values
        self.distributed_training = config.training.distributed_training
        self.work_directory = config.workdir.work_directory

        # Device and DDP setup
        self.gpu_id = (
            dist.get_rank()
            if self.distributed_training
            else config.training.master_gpu_id
        )
        self.driver_device = (
            self.gpu_id == 0
            if self.distributed_training
            else True
        )
        self.device = assign_device(self.gpu_id)

        # Logger
        self.logger = DDPLogger() if self.distributed_training else DefaultLogger()

        # Progress bars
        self.train_pbar = (
            DDPProgressBarHandler()
            if self.distributed_training
            else StandardProgressBarHandler()
        )
        self.val_pbar = (
            DDPProgressBarHandler()
            if self.distributed_training
            else StandardProgressBarHandler()
        )
        self.inference_pbar = StandardProgressBarHandler()

        # Training state
        self.current_epoch = 0
        self.stop_training = False

        # First batch flag for data shape checking
        self._first_train_batch = True

    # ========================================================================
    # AUTO-SETUP (Hybrid approach)
    # ========================================================================

    def _ensure_model_ready_for_training(self):
        """
        Ensure model is ready for training.

        Auto-builds and auto-sets optimizer if needed.
        Logs when doing so for transparency.
        """
        # Auto-build if needed
        if not self.module.is_built:
            self.logger.log_info(
                f"Model '{self.module.model_name}' not built. Building automatically..."
            )
            self.module.build()
            self.logger.log_info("✓ Model built successfully.")

        # Auto-setup optimizer if needed
        if not self.module.is_optimizer_set:
            self.logger.log_info(
                f"Optimizer not set for '{self.module.model_name}'. Setting up automatically..."
            )
            self.module.associate_optimizer()
            self.logger.log_info(
                f"✓ Optimizer configured: {self.module.optimizer.__class__.__name__}"
            )

    def _ensure_model_ready_for_inference(self, weight_path: Optional[str] = None):
        """
        Ensure model is ready for inference.

        Auto-builds if needed.
        Does NOT auto-load weights (user must do this explicitly).
        """
        # Auto-build if needed
        if not self.module.is_built:
            self.logger.log_info(
                f"Model '{self.module.model_name}' not built. Building automatically..."
            )
            self.module.build()
            self.logger.log_info("✓ Model built successfully.")

        # Check weights are loaded (don't auto-load, user must be explicit)
        if not self.module.is_weights_loaded:
            self.logger.log_warning(
                f"⚠ Weights not loaded for '{self.module.model_name}'. "
                "Call module.load_weights() or engine.load_weights() before inference."
            )

    # ========================================================================
    # FIT - Training + Validation
    # ========================================================================

    def fit(
            self,
            train_loader: DataLoader = None,
            val_loader: DataLoader = None,
            data_pipeline: GenericDataPipeline = None,
            max_epochs: Optional[int] = None,
            callbacks: Optional[List[Callback]] = None,
            train_verbosity_level: int = 1,
            val_verbosity_level: int = 1
    ):
        """
        Train the module.

        HYBRID: Auto-builds model and sets optimizer if not already done.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            data_pipeline: GenericDataPipeline instance
            max_epochs: Maximum epochs (None = use config value)
            callbacks: List of callbacks (None = create smart defaults)
            train_verbosity_level: Training progress bar verbosity
            val_verbosity_level: Validation progress bar verbosity

        Example (Simple):
            engine.fit(train_loader, val_loader, max_epochs=100)
            # Auto-builds and sets optimizer

        Example (Advanced):
            module.build(custom_layers=64)
            module.associate_optimizer()
            engine.fit(train_loader, val_loader, max_epochs=100)
            # Uses pre-built model
        """
        # ===== HYBRID: Auto-setup if needed =====
        self._ensure_model_ready_for_training()

        if data_pipeline is not None:
            train_loader = data_pipeline.train_dataloader()
            val_loader = data_pipeline.val_dataloader()

        if train_loader is None and data_pipeline is None:
            raise ValueError("Must provide either train_loader or data_pipeline")

        # Validate readiness (should always pass after auto-setup)
        ReadinessValidator.check_for_training(self.module)
        ReadinessValidator.check_data_loaders(train_loader, val_loader)

        # Get max_epochs
        if max_epochs is None:
            max_epochs = self.config.training.n_train_epochs

        # Wrap model for DDP if needed
        if self.distributed_training:
            self._wrap_model_ddp()

        # Setup callbacks
        if callbacks is None:
            callbacks = self._create_default_callbacks()

        # Wrap callbacks for DDP
        if self.distributed_training:
            callbacks = [DDPAwareCallback(cb) for cb in callbacks]

        callbacks = CallbackList(callbacks)

        # Log training info
        self._log_training_info(max_epochs, len(callbacks.callbacks))

        # Reset first batch flag
        self._first_train_batch = True

        # ===== HOOK: on_train_begin =====
        callbacks.on_train_begin(engine=self, model=self.module.model)

        try:
            for epoch in range(max_epochs):
                self.current_epoch = epoch + 1

                # ===== HOOK: on_epoch_begin =====
                callbacks.on_epoch_begin(
                    epoch=self.current_epoch,
                    engine=self,
                    model=self.module.model
                )

                # Train epoch
                train_loss = self._train_epoch(
                    train_loader,
                    train_verbosity_level
                )

                # Validate epoch
                val_loss, val_data, val_outputs = self._validate_epoch(
                    val_loader,
                    val_verbosity_level
                )

                # Prepare logs
                logs = {
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }

                # ===== HOOK: on_epoch_end =====
                callbacks.on_epoch_end(
                    epoch=self.current_epoch,
                    engine=self,
                    model=self.module.model,
                    logs=logs,
                    val_data=val_data,
                    val_outputs=val_outputs
                )

                # Check early stopping
                if self.stop_training:
                    self.logger.log_info(f"\nStopping training at epoch {self.current_epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.log_info(f"\nTraining interrupted at epoch {self.current_epoch}")

        finally:
            # ===== HOOK: on_train_end =====
            callbacks.on_train_end(engine=self, model=self.module.model)

            # Cleanup DDP
            if self.distributed_training:
                dist.destroy_process_group()

        self.logger.log_info(f"\nTraining of {self.module.model_name} completed.")

    def _train_epoch(self, train_loader, verbosity_level):
        """
        Train for one epoch.

        Iterates over batches and calls module.training_step(batch).
        """
        self.module.model.train()

        # Set epoch for DDP sampler
        if self.distributed_training:
            train_loader.sampler.set_epoch(self.current_epoch)

        # Init progress bar
        self.train_pbar.init(
            len(train_loader),
            verbosity_level,
            message=f"Epoch {self.current_epoch}"
        )

        # Accumulate loss
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Check data shape on first batch
            if self._first_train_batch:
                self.module._check_data_shape()
                self._first_train_batch = False

            # ===== Call module's training_step =====
            step_output = self.module.training_step(batch, self.train_pbar)

            # Extract loss
            loss = step_output['loss']
            running_loss += loss.item()

            # Update progress bar
            self.train_pbar.step(
                batch_idx + 1,
                [("train_loss: ", float(f'{loss.item():.4f}'))]
            )

        # Return average loss
        return running_loss / len(train_loader)

    def _validate_epoch(self, val_loader, verbosity_level):
        """
        Validate for one epoch.

        Iterates over batches and calls module.validation_step(batch).
        """
        self.module.model.eval()

        # Init progress bar
        self.val_pbar.init(
            len(val_loader) + 1,
            verbosity_level
        )

        # Accumulate metrics
        running_loss = 0.0
        last_inputs = None
        last_targets = None
        last_outputs = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # ===== Call module's validation_step =====
                step_output = self.module.validation_step(batch, self.val_pbar)

                # Extract metrics
                loss = step_output['loss']
                running_loss += loss.item()

                # Store last batch for callbacks
                last_outputs = step_output.get('outputs')
                last_targets = step_output.get('targets')
                last_inputs = step_output.get('inputs')

                # Update progress bar
                self.val_pbar.step(
                    batch_idx + 1,
                    [("val_loss: ", float(f'{loss.item():.4f}'))]
                )

        # Average loss
        avg_loss = running_loss / len(val_loader)

        # Package validation data for callbacks
        val_data = (last_inputs, last_targets)

        return avg_loss, val_data, last_outputs

    # ========================================================================
    # PREDICT - Inference
    # ========================================================================

    def predict(
            self,
            test_loader,
            save_to_disk: bool = False,
            output_path: Optional[str] = None,
            verbosity_level: int = 1
    ):
        """
        Run inference on test data.

        HYBRID: Auto-builds model if not already built.
        Does NOT auto-load weights (user must load explicitly).

        Args:
            test_loader: Test DataLoader
            save_to_disk: Save predictions to HDF5 file
            output_path: Path to save predictions (if save_to_disk=True)
            verbosity_level: Progress bar verbosity

        Returns:
            numpy.ndarray: Predictions (if save_to_disk=False)
            None: If save_to_disk=True

        Example:
            # Load weights first (explicit)
            module.load_weights('weights/best.pt')

            # Predict (auto-builds if needed)
            predictions = engine.predict(test_loader)
        """
        # ===== HYBRID: Auto-setup if needed =====
        self._ensure_model_ready_for_inference()

        # Validate (will warn if weights not loaded)
        ReadinessValidator.check_for_inference(self.module)
        ReadinessValidator.check_data_loaders(test_loader=test_loader)

        if save_to_disk and output_path is None:
            raise ValueError("output_path required when save_to_disk=True")

        # Log inference info
        self._log_inference_info(len(test_loader), save_to_disk, output_path)

        # Prepare storage
        storage = self._prepare_prediction_storage(
            test_loader,
            save_to_disk,
            output_path
        )

        # Run inference
        self.module.model.eval()

        self.inference_pbar.init(
            len(test_loader),
            verbosity_level,
            message='Evaluating model in inference mode'
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # ===== Call module's prediction_step =====
                outputs = self.module.prediction_step(batch, self.inference_pbar)

                # Store predictions
                self._store_predictions(
                    storage,
                    batch_idx,
                    outputs,
                    test_loader.batch_size,
                    len(test_loader.dataset)
                )

                # Update progress bar
                self.inference_pbar.step(batch_idx + 1, values=None)

        # Finalize storage
        result = self._finalize_prediction_storage(storage, save_to_disk, output_path)

        self.logger.log_info(f"\nInference of {self.module.model_name} completed.")

        return result

    def _prepare_prediction_storage(self, test_loader, save_to_disk, output_path):
        """Prepare storage for predictions."""
        # Infer output shape from first batch
        sample_batch = next(iter(test_loader))

        with torch.no_grad():
            sample_output = self.module.prediction_step(sample_batch, None)

        # Determine shapes
        batch_size = sample_output.shape[0]
        output_shape = sample_output.shape[1:]
        total_samples = len(test_loader.dataset)
        full_shape = (total_samples,) + output_shape

        # Use module's standard_data_shape if set, otherwise infer
        if self.module.standard_data_shape is not None:
            output_shape = self.module.standard_data_shape
            full_shape = (total_samples,) + output_shape

        if save_to_disk:
            # Validate output path
            self._check_inference_save_path_valid(output_path)

            # Create HDF5 file
            h5_file = h5py.File(output_path, 'w')
            h5_dataset = h5_file.create_dataset(
                'predictions',
                shape=full_shape,
                dtype='float32',
                chunks=(batch_size,) + output_shape
            )

            self.logger.log_info(f"Created HDF5 dataset '{output_path}' to store predictions.")

            return {'type': 'disk', 'file': h5_file, 'dataset': h5_dataset}
        else:
            # Allocate memory
            predictions = np.zeros(full_shape, dtype=np.float32)
            self.logger.log_info(f"Created tensor in memory to store predictions.")
            return {'type': 'memory', 'data': predictions}

    def _store_predictions(self, storage, batch_idx, outputs, batch_size, dataset_len):
        """Store predictions for a batch."""
        batch_data = outputs.cpu().detach().numpy()

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, dataset_len)

        if storage['type'] == 'disk':
            storage['dataset'][start_idx:end_idx] = batch_data
        else:
            storage['data'][start_idx:end_idx] = batch_data

    def _finalize_prediction_storage(self, storage, save_to_disk, output_path):
        """Finalize storage and return result."""
        if storage['type'] == 'disk':
            storage['file'].close()
            self.logger.log_info(f"Saved predictions to '{output_path}'")
            return None
        else:
            return storage['data']

    def _check_inference_save_path_valid(self, inference_filename):
        """Validate inference output path."""
        file_name, file_extension = os.path.splitext(inference_filename)

        if os.path.isdir(inference_filename):
            raise IsADirectoryError(
                f"ERROR: '{os.path.abspath(inference_filename)}' is a directory."
            )

        if file_extension != '.h5':
            raise ValueError(
                f"ERROR: '{os.path.abspath(inference_filename)}' must have .h5 extension."
            )

        if os.path.exists(inference_filename):
            raise FileExistsError(
                f"ERROR: '{os.path.abspath(inference_filename)}' already exists."
            )

    # ========================================================================
    # DDP
    # ========================================================================

    def _wrap_model_ddp(self):
        """Wrap model with DistributedDataParallel."""
        from torch.nn.parallel import DistributedDataParallel

        if not isinstance(self.module.model, DistributedDataParallel):
            self.module.model = DistributedDataParallel(
                self.module.model,
                device_ids=[self.gpu_id]
            )
            self.module.model.to(self.device)
            self.logger.log_info("Model wrapped in DistributedDataParallel.")

    # ========================================================================
    # DEFAULT CALLBACKS
    # ========================================================================

    def _create_default_callbacks(self):
        """Create smart default callbacks based on config."""
        '''from torchmodel.callbacks.callbacks_core import (
            ModelCheckpoint,
            CSVLogger,
            TextLogger,
            TensorBoardScalars,
            TensorBoardHistograms,
            TensorBoardGraph
        )'''  # vedi se tenerlo cosi' + forse fai interfacce in __init__.py
        from src.callbacks.modelcheckpoint import ModelCheckpoint
        from src.callbacks.csv_logger import CSVLogger
        from src.callbacks.txt_logger import TextLogger
        from src.callbacks.tensorboard_callbacks import TensorBoardScalars, TensorBoardHistograms, TensorBoardGraph

        from src.callbacks.tensorboard_callback_images import SimpleImagePlotter

        callbacks = []

        # Setup paths
        timestamp = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
        work_dir = Path(os.path.expandvars(self.work_directory))
        model_name = self.module.model_name
        train_codename = self.config.model.train_codename

        # === CSV Logger (always) ===
        csv_dir = work_dir / "logs" / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{model_name}_{timestamp}_{train_codename}.csv"
        callbacks.append(CSVLogger(str(csv_path)))

        # === Text Logger (always) ===
        log_dir = work_dir / "logs" / "text"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model_name}_{timestamp}_{train_codename}.log"
        callbacks.append(TextLogger(str(log_path)))

        # === Model Checkpoint (if configured) ===
        if self.config.model.save_model_weights:
            weight_dir = work_dir / "weights" / model_name
            weight_dir.mkdir(parents=True, exist_ok=True)
            weight_path = weight_dir / f"best_{model_name}_{timestamp}_{train_codename}.pt"

            callbacks.append(
                ModelCheckpoint(
                    filepath=str(weight_path),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=True
                )
            )

        # === TensorBoard (if configured) ===
        if self.config.model.log_to_tensorboard:
            tb_dir = work_dir / "logs" / "tensorboard" / model_name / timestamp / train_codename
            tb_dir.mkdir(parents=True, exist_ok=True)

            # Scalars
            callbacks.append(TensorBoardScalars(str(tb_dir)))

            # Histograms
            callbacks.append(TensorBoardHistograms(str(tb_dir), freq=5))

            # Model graph
            callbacks.append(
                TensorBoardGraph(
                    str(tb_dir),
                    input_to_model=lambda: self.module.get_sample_input()
                )
            )

            # Image plotting (if configured)
            if hasattr(self.config.model, 'tensorboard_img_id'):
                img_dir = tb_dir / 'images'
                callbacks.append(
                    SimpleImagePlotter(
                        log_dir=str(img_dir),
                        tag=self.config.model.tensorboard_img_id,
                        freq=1,
                        batch_indices=getattr(self.config.model, 'batch_idx_viz', [0])
                    )
                )

        return callbacks

    # ========================================================================
    # WEIGHT LOADING (Convenience method)
    # ========================================================================

    def load_weights(self, weight_path: str, strict: bool = True):
        """
        Load weights into module.

        Convenience method that calls module.load_weights().

        Args:
            weight_path: Path to weight file
            strict: Strict loading
        """
        self.module.load_weights(weight_path, strict)
        self.logger.log_info(f"Loaded weights from '{weight_path}' successfully.")

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log_training_info(self, max_epochs, num_callbacks):
        """Log training configuration."""
        attrib_to_log = {
            'model_name': self.module.model_name,
            'optimizer': self.module.optimizer.__class__.__name__,
            'batch_size': self.module.batch_size,
            'n_train_epochs': max_epochs,
            'learning_rate': self.module.learning_rate,
            'work_directory': self.work_directory,
            'save_model_weights': self.config.model.save_model_weights,
            'distributed_training': self.distributed_training,
            'callbacks': num_callbacks
        }

        self.logger.log_info(
            'Model being used with the following parameters:\n\n' +
            '\n '.join(f"{k} -> {v}" for k, v in attrib_to_log.items()) + '\n'
        )

    def _log_inference_info(self, num_batches, save_to_disk, output_path):
        """Log inference configuration."""
        self.logger.log_info(
            f"\nRunning inference for {self.module.model_name}\n"
            f"Batches: {num_batches}\n"
            f"Device: {self.device}\n"
            f"Save to disk: {save_to_disk}\n"
            + (f"Output: {output_path}\n" if save_to_disk else "Output: In-memory\n")
        )
