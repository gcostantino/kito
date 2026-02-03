import datetime
import os
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from kito.callbacks.callback_base import Callback, CallbackList
from kito.callbacks.ddp_aware_callback import DDPAwareCallback
from kito.config.moduleconfig import CallbacksConfig, KitoModuleConfig
from kito.data.datapipeline import GenericDataPipeline
from kito.module import KitoModule
from kito.strategies.logger_strategy import DDPLogger, DefaultLogger
from kito.strategies.progress_bar_strategy import (
    StandardProgressBarHandler,
    DDPProgressBarHandler
)
from kito.strategies.readiness_validator import ReadinessValidator
from kito.utils.decorators import require_mode
from kito.utils.gpu_utils import assign_device, get_available_devices


def _ddp_worker_fn(rank, worker_args, world_size):
    """
    Worker function that runs in each spawned process.

    This recreates the Engine and runs training.
    """
    try:
        # Set environment variables
        os.environ['KITO_WORKER_RANK'] = str(rank)
        os.environ['KITO_WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # Recreate module and engine in this process
        config = worker_args['config']
        module_class = worker_args['module_class']

        # Instantiate module (user's class)
        module = module_class(config)

        # Create engine
        engine = Engine(module, config)

        # Get data loaders
        data_pipeline = worker_args['data_pipeline']
        if data_pipeline is not None:
            # Recreate data pipeline in this process
            data_pipeline.setup()
            train_loader = data_pipeline.train_dataloader()
            val_loader = data_pipeline.val_dataloader()
        else:
            raise ValueError("DataPipeline required for auto-spawned DDP")

        # Run training (will skip spawning since _is_spawned_worker=True)
        engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=worker_args['max_epochs'],
            callbacks=worker_args['callbacks']
        )

    except KeyboardInterrupt:
        print(f"Worker {rank}: Interrupted")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


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

    def __init__(self, module: KitoModule, config: KitoModuleConfig):
        """
        Initialize Engine.

        Args:
            module: KitoModule instance (can be built or not)
            config: Configuration object
        """
        self.max_epochs = None
        self.module = module
        self.config = config

        # Extract config values
        self.distributed_training = config.training.distributed_training
        self.work_directory = config.workdir.work_directory

        # ===== AUTO-INITIALIZE DDP =====
        # Track if we're in a spawned worker process OR launched with torchrun
        self._is_spawned_worker = os.environ.get('KITO_WORKER_RANK') is not None
        self._is_torchrun = (
                os.environ.get('RANK') is not None and
                os.environ.get('LOCAL_RANK') is not None and
                os.environ.get('WORLD_SIZE') is not None
        )

        self.logger = DDPLogger() if self.distributed_training else DefaultLogger()

        # Initialize DDP if we're in a worker process (spawn OR torchrun)
        if self.distributed_training and (self._is_spawned_worker or self._is_torchrun):
            self._auto_init_ddp()

        self._setup_devices(config)

        # Assign to module
        self.module._move_to_device(self.device)

        # Progress bars - only use DDP handlers when DDP is actually running
        ddp_initialized = dist.is_available() and dist.is_initialized()  # this could become a class attribute

        self.train_pbar = (
            DDPProgressBarHandler()
            if (self.distributed_training and ddp_initialized)
            else StandardProgressBarHandler()
        )
        self.val_pbar = (
            DDPProgressBarHandler()
            if (self.distributed_training and ddp_initialized)
            else StandardProgressBarHandler()
        )
        self.inference_pbar = StandardProgressBarHandler()

        self.timestamp = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
        self.train_run_id = self.timestamp  # Alias for clarity

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
                f"Model '{self.module.module_name}' not built. Building automatically..."
            )
            self.module.build()
            self.module._move_to_device(self.device)  # Then move to device
            self.logger.log_info("Model built successfully.")

        # load pretrained weights if specified (for transfer learning / fine-tuning)
        if self.config.training.initialize_model_with_saved_weights:
            self.load_weights()

        # Auto-setup optimizer if needed
        if not self.module.is_optimizer_set:
            self.logger.log_info(
                f"Optimizer not set for '{self.module.module_name}'. Setting up automatically..."
            )
            self.module.associate_optimizer()
            self.logger.log_info(
                f"Optimizer configured: {self.module.optimizer.__class__.__name__}"
            )

    def _ensure_model_ready_for_inference(self):
        """
        Ensure model is ready for inference.

        Auto-builds if needed.
        Auto-loads weights (user can also do this explicitly).
        """
        # Auto-build if needed
        if not self.module.is_built:
            self.logger.log_info(
                f"Model '{self.module.module_name}' not built. Building automatically..."
            )
            self.module.build()
            self.logger.log_info("Model built successfully.")

        #  Auto-load weights if not already loaded and config specifies a path
        if not self.module.is_weights_loaded:
            if self.config.model.weight_load_path is not None:
                self.logger.log_info("Loading weights automatically...")
                self.load_weights()
            else:
                self.logger.log_warning(
                    f"Weights not loaded for '{self.module.module_name}'. "
                    "Set config.model.weight_load_path or call module.load_weights() before inference."
                )

    # ========================================================================
    # Device handling
    # ========================================================================
    def _setup_devices(self, config):
        """
        Setups devices and DistributedDataParallel configuration.
        """
        # Check if DDP is actually initialized (not just config flag)
        ddp_initialized = dist.is_available() and dist.is_initialized()

        if self.distributed_training and ddp_initialized:
            # ===== DDP WORKER MODE (spawned or torchrun) =====
            # Global rank (0 to world_size-1)
            self.rank = dist.get_rank()

            # Local rank (0 to num_gpus_per_node-1)
            local_rank_str = os.environ.get('LOCAL_RANK')
            if local_rank_str is None:
                # Try KITO_WORKER_RANK (from spawn)
                local_rank_str = os.environ.get('KITO_WORKER_RANK')

            if local_rank_str is None:
                raise RuntimeError(
                    "LOCAL_RANK environment variable not found. "
                    "This should be set by torchrun or Kito spawn."
                )

            self.local_rank = int(local_rank_str)
            self.gpu_id = self.local_rank

            # Only global rank 0 is the master/driver
            self.is_master = (self.rank == 0)
            self.driver_device = self.is_master
            self.world_size = dist.get_world_size()

        elif self.distributed_training and not ddp_initialized:
            # ===== MAIN PROCESS (before spawning) =====
            # DDP config enabled but not yet spawned - use defaults for main process
            self.rank = 0
            self.local_rank = config.training.device_id  # Use config default
            self.gpu_id = config.training.device_id
            self.is_master = True
            self.driver_device = True
            self.world_size = torch.cuda.device_count()  # it only works with CUDA. To be extended...

            self.logger.log_info(
                f"Main process: DDP will spawn {self.world_size} workers"
            )

        else:
            # ===== SINGLE GPU MODE =====
            self.rank = 0
            self.local_rank = config.training.device_id
            self.gpu_id = config.training.device_id
            self.is_master = True
            self.driver_device = True
            self.world_size = 1

        device_type = config.training.device_type
        self.device = assign_device(device_type, self.gpu_id)

        # Log device info
        available = get_available_devices()
        self.logger.log_info(
            f"Device configuration:\n"
            f"  Requested: {device_type}\n"
            f"  Assigned: {self.device}\n"
            f"  Available: {available}"
            f"  Rank: {self.rank}/{self.world_size}"
        )

    # ========================================================================
    # FIT - Training + Validation
    # ========================================================================
    @require_mode('train')
    def fit(
            self,
            train_loader: DataLoader = None,
            val_loader: DataLoader = None,
            data_pipeline: GenericDataPipeline = None,
            max_epochs: Optional[int] = None,
            callbacks: Optional[List[Callback]] = None,
    ):
        """
        Train the module.

        Auto-builds model and sets optimizer if not already done.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            data_pipeline: GenericDataPipeline instance
            max_epochs: Maximum epochs (None = use config value)
            callbacks: List of callbacks (None = create smart defaults)

        Example (Simple):
            engine.fit(train_loader, val_loader, max_epochs=100)
            # Auto-builds and sets optimizer

        Example (Advanced):
            module.build(custom_layers=64)
            module.associate_optimizer()
            engine.fit(train_loader, val_loader, max_epochs=100)
            # Uses pre-built model
        Example (DDP, Advanced)
        Strategy #1 (recommended) - use torchrun:
            torchrun --nproc_per_node=4 train.py
        Strategy #2: transparent (handled by Engine), less control
            python train.py
        """
        # ===== AUTO-SPAWN FOR DDP =====
        # Only spawn if DDP enabled AND not already in a worker (spawn or torchrun)
        if self.distributed_training and not self._is_spawned_worker and not self._is_torchrun:
            # We're in main process - spawn workers
            return self._spawn_and_train(
                train_loader, val_loader, data_pipeline, max_epochs, callbacks
            )

        self._ensure_model_ready_for_training()

        if data_pipeline is not None:
            train_loader = data_pipeline.train_dataloader()
            val_loader = data_pipeline.val_dataloader()

        if train_loader is None and data_pipeline is None:
            raise ValueError("Must provide either train_loader or data_pipeline")

        # Validate readiness (should always pass after auto-setup)
        ReadinessValidator.check_for_training(self.module)
        ReadinessValidator.check_data_loaders(train_loader=train_loader, val_loader=val_loader)

        # Get max_epochs
        if max_epochs is None:
            max_epochs = self.config.training.n_train_epochs
        self.max_epochs = max_epochs

        # Wrap model for DDP if needed
        if self.distributed_training:
            self._wrap_model_ddp()

        # create defaults
        default_callbacks = self._create_default_callbacks()

        # merge with user callbacks
        if callbacks is not None:
            if isinstance(callbacks, Callback):
                callbacks = [callbacks]
            callbacks = default_callbacks + callbacks
        else:
            callbacks = default_callbacks

        # autoconfigure callbacks with Engine context
        callbacks = self._setup_callbacks(callbacks)

        # Wrap callbacks for DDP
        if self.distributed_training:
            callbacks = [DDPAwareCallback(cb, master_rank=0) for cb in callbacks]

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
                    self.config.training.train_verbosity_level
                )

                # Validate epoch
                val_loss, val_data, val_outputs = self._validate_epoch(
                    val_loader,
                    self.config.training.val_verbosity_level
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

        self.logger.log_info(f"\nTraining of {self.module.module_name} completed.")

    def _train_epoch(self, train_loader, verbosity_level):
        """
        Train for one epoch.

        Iterates over batches and calls module.training_step(batch).
        """
        self.module.model.train()

        # Set epoch for DDP sampler

        if self.distributed_training:
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(self.current_epoch)
            else:
                self.logger.log_warning(
                    "DataLoader sampler does not have set_epoch method. "
                    "For DDP, use DistributedSampler to ensure proper shuffling."
                )

        # Init progress bar
        self.train_pbar.init(
            len(train_loader),
            verbosity_level,
            message=f"Epoch {self.current_epoch}/{self.max_epochs}"
        )

        # Accumulate loss
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Check data shape on first batch
            if self._first_train_batch:
                self.module._check_data_shape(batch)
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
            len(val_loader),
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
    @require_mode('inference')
    def predict(
            self,
            test_loader: DataLoader = None,
            data_pipeline: GenericDataPipeline = None
    ):
        """
        Run inference on test data.

        This method auto-loads weights (or user must load explicitly).

        IMPORTANT: prediction_step is called once on a sample batch to infer output shape.

        Args:
            test_loader: Test DataLoader
            data_pipeline: Data pipeline for inference

        Returns:
            numpy.ndarray: Predictions (if config.model.save_inference_to_disk=False)
            None: If config.model.save_inference_to_disk=True

        Example:
            # Load weights first (explicit)
            module.load_weights('weights/best.pt')

            # Predict (auto-builds if needed)
            predictions = engine.predict(test_loader)
        """
        self._ensure_model_ready_for_inference()

        if data_pipeline is not None:
            test_loader = data_pipeline.test_dataloader()

        if test_loader is None and data_pipeline is None:
            raise ValueError("Must provide either test_loader or data_pipeline")

        # Validate (will warn if weights not loaded)
        ReadinessValidator.check_for_inference(self.module)
        ReadinessValidator.check_data_loaders(test_loader=test_loader)

        save_to_disk = self.config.model.save_inference_to_disk
        output_path = self.config.model.inference_filename

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
            self.config.training.test_verbosity_level,
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

        self.logger.log_info(f"\nInference of {self.module.module_name} completed.")

        return result

    def _prepare_prediction_storage(self, test_loader, save_to_disk, output_path):
        """Prepare storage for predictions."""
        # Infer output shape from first batch
        sample_batch = next(iter(test_loader))

        with torch.no_grad():
            sample_output = self.module.prediction_step(sample_batch, None)

        # Validate prediction_step returns a tensor
        if not isinstance(sample_output, torch.Tensor):
            raise TypeError(
                f"prediction_step must return a torch.Tensor, got {type(sample_output)}. "
                "If returning multiple outputs, return only the main prediction tensor."
            )

        # Determine shapes
        output_shape = sample_output.shape[1:]
        total_samples = len(test_loader.dataset)
        full_shape = (total_samples,) + output_shape

        # Use module's standard_data_shape if set, otherwise infer
        if self.module.standard_data_shape is not None:
            output_shape = self.module.standard_data_shape
            full_shape = (total_samples,) + output_shape

        # Use a reasonable chunk size (not tied to actual batch size)
        chunk_size = min(test_loader.batch_size, 32)  # Cap at 32 for HDF5 efficiency

        if save_to_disk:
            # Validate output path
            self._check_inference_save_path_valid(output_path)

            # Create HDF5 file
            h5_file = h5py.File(output_path, 'w')
            h5_dataset = h5_file.create_dataset(
                'predictions',
                shape=full_shape,
                dtype='float32',
                chunks=(chunk_size,) + output_shape
            )

            self.logger.log_info(f"Created HDF5 dataset '{output_path}' to store predictions.")

            return {'type': 'disk', 'file': h5_file, 'dataset': h5_dataset}
        else:
            # Allocate memory
            predictions = np.zeros(full_shape, dtype=np.float32)
            self.logger.log_info(f"Created tensor in memory to store predictions.")
            return {'type': 'memory', 'data': predictions}

    def _store_predictions(self, storage, batch_idx, outputs, nominal_batch_size, dataset_len):
        """Store predictions for a batch."""
        batch_data = outputs.cpu().detach().numpy()

        # Use actual batch size from output, not nominal batch size
        actual_batch_size = batch_data.shape[0]

        start_idx = batch_idx * nominal_batch_size
        end_idx = min(start_idx + actual_batch_size, dataset_len)

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

    def _spawn_and_train(self, train_loader, val_loader, data_pipeline, max_epochs, callbacks):
        """
        Spawn worker processes for DDP training.

        This is called only once from the main process.
        """
        if data_pipeline is None and (train_loader is not None or val_loader is not None):
            raise ValueError(
                "When using automatic DDP spawning, you must provide a data_pipeline "
                "(not pre-created dataloaders). Example:\n"
                "  engine.fit(data_pipeline=pipeline, max_epochs=100)\n"
                "Alternatively, launch with torchrun and pass dataloaders directly."
            )

        import torch.multiprocessing as mp

        nprocs = self.world_size

        if nprocs < 2:
            raise ValueError(
                f"Distributed training enabled but only {nprocs} GPU(s) available. "
                "Need at least 2 GPUs or disable distributed training."
            )

        self.logger.log_info(f"Spawning {nprocs} processes for DDP training...")

        # Set master address
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

        # Prepare args to pass to workers
        worker_args = {
            'module_class': type(self.module),
            'module_name': self.module.module_name,
            'config': self.config,
            'data_pipeline': data_pipeline,
            'max_epochs': max_epochs,
            'callbacks': callbacks,
        }

        # Spawn workers
        try:
            mp.spawn(
                _ddp_worker_fn,
                args=(worker_args, nprocs),
                nprocs=nprocs,
                join=True
            )
        except KeyboardInterrupt:
            self.logger.log_info("Training interrupted by user.")

        self.logger.log_info("DDP training completed.")

    def _auto_init_ddp(self):
        """
        Automatically initialize DDP if not already initialized.

        Supports three launch modes:
        1. torchrun (recommended): User runs `torchrun --nproc_per_node=N script.py
        2. Transparent spawn: Engine.fit() spawns workers automatically
        3. Already initialized: Skip initialization
        """
        # Check if already initialized
        if dist.is_available() and dist.is_initialized():
            self.logger.log_info("DDP already initialized, skipping setup.")
            return

        # Mode 1: Check for Kito spawn (KITO_WORKER_RANK set by _ddp_worker_fn)
        kito_rank = os.environ.get('KITO_WORKER_RANK')
        kito_world_size = os.environ.get('KITO_WORLD_SIZE')

        if kito_rank is not None and kito_world_size is not None:
            # Spawned by Engine._spawn_and_train()
            rank = int(kito_rank)
            world_size = int(kito_world_size)

            self.logger.log_info(
                f"Detected Kito spawn launch (rank={rank}, world_size={world_size})"
            )

            # Set defaults for master address if not set
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '12355'

            # Initialize process group
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)

            self.logger.log_info(
                f"DDP initialized via spawn: backend={backend}, "
                f"rank={dist.get_rank()}, world_size={dist.get_world_size()}"
            )
            return

        # Mode 2: Check for torchrun (RANK/LOCAL_RANK/WORLD_SIZE set by torchrun)
        rank = os.environ.get('RANK')
        local_rank = os.environ.get('LOCAL_RANK')
        world_size = os.environ.get('WORLD_SIZE')

        if rank is not None and local_rank is not None and world_size is not None:
            # Launched with torchrun
            self.logger.log_info(
                f"Detected torchrun launch (rank={rank}, local_rank={local_rank}, "
                f"world_size={world_size})"
            )

            # Set defaults for master address if not set
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '12355'

            # Initialize process group (torchrun already set rank/world_size)
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)

            self.logger.log_info(
                f"DDP initialized via torchrun: backend={backend}, "
                f"rank={dist.get_rank()}, world_size={dist.get_world_size()}"
            )
            return

        # Mode 3: No DDP environment detected - should not happen if spawn is working
        # This case is handled by fit() which spawns if needed
        self.logger.log_warning(
            "DDP initialization called but no distributed environment detected. "
            "This should not happen - check your launch configuration."
        )

    def _wrap_model_ddp(self):
        """Wrap model with DistributedDataParallel."""
        from torch.nn.parallel import DistributedDataParallel

        if not isinstance(self.module.model, DistributedDataParallel):
            # Model should already be on correct device before wrapping
            # DDP will handle device placement internally
            self.module.model = DistributedDataParallel(
                self.module.model,
                device_ids=[self.gpu_id]
            )
            # self.module._move_to_device(self.device)
            self.logger.log_info("Model wrapped in DistributedDataParallel.")

            # Note: Do NOT call _move_to_device after DDP wrapping
            # DDP already manages device placement via device_ids

    # ========================================================================
    # DEFAULT CALLBACKS
    # ========================================================================

    def _setup_callbacks(self, callbacks: List[Callback]) -> List[Callback]:
        """
        Setup callbacks with Engine context.

        Passes shared timestamp and other Engine attributes to callbacks.
        This ensures all callbacks use the same parameters (e.g., timestamp) for the training run.

        Args:
            callbacks: List of callback instances

        Returns:
            Configured callbacks
        """
        # create context dict with shared values
        context = {
            'timestamp': self.timestamp,
            'work_directory': self.work_directory,
            'module_name': self.module.module_name,
            'train_codename': self.config.model.train_codename,
        }

        for callback in callbacks:
            if hasattr(callback, 'setup'):
                callback.setup(engine=self, **context)

        return callbacks

    def _create_default_callbacks(self):
        """Create smart default callbacks based on config."""
        from kito.callbacks.modelcheckpoint import ModelCheckpoint
        from kito.callbacks.csv_logger import CSVLogger
        from kito.callbacks.txt_logger import TextLogger
        from kito.callbacks.tensorboard_callbacks import (
            TensorBoardScalars,
            TensorBoardHistograms,
            TensorBoardGraph
        )

        callbacks = []

        # Get callbacks config (with defaults if not provided)
        cb_config = getattr(self.config, 'callbacks', CallbacksConfig())

        # Setup paths
        work_dir = Path(os.path.expandvars(self.work_directory))
        module_name = self.module.module_name
        train_codename = self.config.model.train_codename

        # === CSV Logger ===
        if cb_config.enable_csv_logger:
            csv_dir = work_dir / "logs" / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"{module_name}_{self.timestamp}_{train_codename}.csv"
            callbacks.append(CSVLogger(str(csv_path)))

        # === Text Logger ===
        if cb_config.enable_text_logger:
            log_dir = work_dir / "logs" / "text"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{module_name}_{self.timestamp}_{train_codename}.log"
            callbacks.append(TextLogger(str(log_path)))

        # === Model Checkpoint ===
        if cb_config.enable_model_checkpoint:
            weight_dir = work_dir / "weights" / module_name
            weight_dir.mkdir(parents=True, exist_ok=True)
            weight_path = weight_dir / f"best_{module_name}_{self.timestamp}_{train_codename}.pt"

            callbacks.append(
                ModelCheckpoint(
                    filepath=str(weight_path),
                    monitor=cb_config.checkpoint_monitor,
                    save_best_only=cb_config.checkpoint_save_best_only,
                    mode=cb_config.checkpoint_mode,
                    verbose=cb_config.checkpoint_verbose
                )
            )

        # === TensorBoard ===
        if cb_config.enable_tensorboard:
            tb_dir = work_dir / "logs" / "tensorboard" / module_name / self.timestamp / train_codename
            tb_dir.mkdir(parents=True, exist_ok=True)

            # Scalars
            if cb_config.tensorboard_scalars:
                callbacks.append(TensorBoardScalars(str(tb_dir)))

            # Histograms
            if cb_config.tensorboard_histograms:
                callbacks.append(
                    TensorBoardHistograms(
                        str(tb_dir),
                        freq=cb_config.tensorboard_histogram_freq
                    )
                )

            # Model graph
            if cb_config.tensorboard_graph:
                callbacks.append(
                    TensorBoardGraph(
                        str(tb_dir),
                        input_to_model=lambda: self.module.get_sample_input()
                    )
                )

            # Image plotting
            if cb_config.tensorboard_images:
                img_dir = tb_dir / 'images'

                # use user-specified plotter class or auto-detect
                plotter_class = self._get_image_plotter_class(cb_config)
                plotter = plotter_class(
                    log_dir=str(img_dir),  # log_dir can also be auto-configured by setup()
                    tag=getattr(self.config.model, 'tensorboard_img_id', None),
                    freq=cb_config.tensorboard_image_freq,
                    batch_indices=cb_config.tensorboard_batch_indices
                )

                callbacks.append(plotter)

        return callbacks

    def _get_image_plotter_class(self, cb_config):
        """
        Get the image plotter class to use.

        Priority:
        1. User-specified class in config
        2. Module's preferred plotter (if module defines one)
        3. SimpleImagePlotter (default fallback)

        Args:
            cb_config: CallbacksConfig instance

        Returns:
            Image plotter class
        """
        # priority 1: explicitly specified in config
        if cb_config.image_plotter_class is not None:
            return cb_config.image_plotter_class

        # priority 2: module recommends a plotter
        if hasattr(self.module, 'get_preferred_image_plotter'):
            return self.module.get_preferred_image_plotter()

        # priority 3: default fallback
        from kito.callbacks.tensorboard_callback_images import SimpleImagePlotter
        return SimpleImagePlotter

    def get_default_callbacks(self):
        """
        Get default callbacks configured from config.callbacks.

        Useful when you want to extend defaults with custom callbacks.

        Returns:
            List of callback instances

        Example:
            >>> # Get defaults
            >>> callbacks = engine.get_default_callbacks()
            >>>
            >>> # Modify or extend
            >>> callbacks.append(MyCustomCallback(param=10))
            >>>
            >>> # Or modify existing
            >>> for cb in callbacks:
            ...     if isinstance(cb, ModelCheckpoint):
            ...         cb.verbose = False
            >>>
            >>> # Use them
            >>> engine.fit(train_loader, val_loader, callbacks=callbacks)
        """
        return self._create_default_callbacks()

    # ========================================================================
    # WEIGHT LOADING (Convenience method)
    # ========================================================================

    def load_weights(self, weight_path: str = None, strict: bool = True):
        """
        Load weights into module.

        Convenience method that calls module.load_weights().

        Args:
            weight_path: Path to weight file
            strict: Strict loading
        """
        if weight_path is None:
            weight_path = self.config.model.weight_load_path
        ReadinessValidator.check_model_built(self.module)
        ReadinessValidator.check_weights_config(weight_path)
        try:
            self.module.load_weights(weight_path, strict)
            self.logger.log_info(
                f"Successfully loaded weights from: {weight_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained weights from '{weight_path}'\n"
                f"Error: {e}"
            ) from e

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log_training_info(self, max_epochs, num_callbacks):
        """Log training configuration."""
        attrib_to_log = {
            'module_name': self.module.module_name,
            'optimizer': self.module.optimizer.__class__.__name__,
            'batch_size': self.module.batch_size,
            'n_train_epochs': max_epochs,
            'learning_rate': self.module.learning_rate,
            'work_directory': self.work_directory,
            'model_checkpointing': self.config.callbacks.enable_model_checkpoint,
            'distributed_training': self.distributed_training,
            'callbacks': num_callbacks  # this list could be extended
        }

        self.logger.log_info(
            'Model being used with the following parameters:\n\n' +
            '\n '.join(f"{k} -> {v}" for k, v in attrib_to_log.items()) + '\n'
        )

    def _log_inference_info(self, num_batches, save_to_disk, output_path):
        """Log inference configuration."""
        self.logger.log_info(
            f"\nRunning inference for {self.module.module_name}\n"
            f"Batches: {num_batches}\n"
            f"Device: {self.device}\n"
            f"Save to disk: {save_to_disk}\n"
            + (f"Output: {output_path}\n" if save_to_disk else "Output: In-memory\n")
        )
