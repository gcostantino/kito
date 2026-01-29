import os
from abc import ABC, abstractmethod

import torch
from packaging.version import parse
from torchsummary import summary as model_summary

from kito.config.moduleconfig import KitoModuleConfig


class KitoModule(ABC):
    """
    Base module for PyTorch models.

    Focused on model definition and single-batch operations.
    The Engine handles all iteration, callbacks, and orchestration.

    Usage:
        class MyModel(KitoModule):
            def build_inner_model(self):
                self.model = nn.Sequential(...)
                self.model_input_size = (3, 64, 64)
                self.standard_data_shape = (3, 64, 64)  # For inference

            def bind_optimizer(self):
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.learning_rate
                )

        # Use with Engine
        module = MyModel('MyModel', device, config)
        module.build()
        module.associate_optimizer()

        engine = Engine(module)
        engine.fit(train_loader, val_loader, max_epochs=100)
    """

    def __init__(self, model_name: str, config: KitoModuleConfig = None):
        """
        Initialize BaseModule.

        Args:
            model_name: Name of the model
            config: Optional config object for future extensibility
        """
        self.model_name = model_name
        self.config = config

        # Extract useful config values if provided
        if config is not None:
            self.learning_rate = config.training.learning_rate
            self.batch_size = config.training.batch_size
        else:
            self.learning_rate = None
            self.batch_size = None

        # Model components
        self.model = None
        self.device = None  # set by Engine
        self.model_input_size = None
        self.standard_data_shape = None  # For inference (set by subclass)
        self.optimizer = None

        # Loss function (set by subclass or from config)
        if config is not None:
            from kito.utils.loss_utils import get_loss
            self.loss = get_loss(config.model.loss)
        else:
            self.loss = None

        # State flags
        self._model_built = False
        self._optimizer_bound = False
        self._weights_loaded = False

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================
    @abstractmethod
    def build_inner_model(self, *args, **kwargs):
        """
        Build the model architecture.

        Must set:
        - self.model: The PyTorch model
        - self.model_input_size: Tuple of input shape (C, H, W) or (C, H, W, D)
        - self.standard_data_shape: Output shape for inference (optional)

        Example:
            def build_inner_model(self):
                self.model = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 3, 3, padding=1)
                )
                self.model_input_size = (3, 64, 64)
                self.standard_data_shape = (3, 64, 64)
        """
        pass

    @abstractmethod
    def bind_optimizer(self, *args, **kwargs):
        """
        Setup the optimizer.

        Must set:
        - self.optimizer: The PyTorch optimizer

        Example:
            def bind_optimizer(self):
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.learning_rate
                )
        """
        pass

    def _check_data_shape(self):
        """
        Check data shape on first batch.

        Optional - implement if you need to validate input shape.
        Called by Engine on first training batch.

        Example:
            def _check_data_shape(self):
                # Validate that data matches expected shape
                pass
        """
        pass  # Default: no checking

    # ========================================================================
    # SETUP METHODS
    # ========================================================================

    def _move_to_device(self, device: torch.device):
        """
        Internal method called by Engine to move model to device.

        Called AFTER build() by Engine.
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)

    def build(self, *args, **kwargs):
        """Build model and move to device."""
        self.build_inner_model(*args, **kwargs)
        # self.model.to(self.device)
        self._model_built = True

    def associate_optimizer(self, *args, **kwargs):
        """Setup optimizer."""
        if not self._model_built:
            raise RuntimeError("Must call build() before associate_optimizer()")
        self.bind_optimizer(*args, **kwargs)
        self._optimizer_bound = True

    # ========================================================================
    # SINGLE-BATCH OPERATIONS (Called by Engine)
    # ========================================================================

    def training_step(self, batch, pbar_handler=None):
        """
        Perform one training step on a single batch.

        Called by Engine for each training batch.

        Args:
            batch: Tuple of (inputs, targets) from DataLoader
            pbar_handler: Progress bar handler (optional, provided by Engine)

        Returns:
            dict: {'loss': tensor} - Must contain at least 'loss'

        Override this for custom training logic (freeze layers, gradient accumulation, etc.).

        Default implementation:
            1. Move data to device
            2. Zero gradients
            3. Forward pass
            4. Compute loss
            5. Backward pass
            6. Optimizer step
        """
        inputs, targets = batch

        # Move to device
        inputs = self.send_data_to_device(inputs)
        targets = targets.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.pass_data_through_model(inputs)

        # Compute loss
        loss = self.compute_loss((inputs, targets), outputs)

        # Backward pass
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        return {'loss': loss}

    def validation_step(self, batch, pbar_handler=None):
        """
        Perform one validation step on a single batch.

        Called by Engine for each validation batch.

        Args:
            batch: Tuple of (inputs, targets) from DataLoader
            pbar_handler: Progress bar handler (optional, provided by Engine)

        Returns:
            dict: Must contain 'loss', 'outputs', 'inputs', 'targets'

        Override this for custom validation logic.
        """
        inputs, targets = batch

        # Move to device
        inputs = self.send_data_to_device(inputs)
        targets = targets.to(self.device)

        # Forward pass (no gradients)
        outputs = self.pass_data_through_model(inputs)

        # Compute loss
        loss = self.compute_loss((inputs, targets), outputs)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets,
            'inputs': inputs
        }

    def prediction_step(self, batch, pbar_handler=None):
        """
        Perform one prediction step on a single batch.

        Called by Engine for each inference batch.

        Args:
            batch: Input data from DataLoader (can be tuple or tensor)
            pbar_handler: Progress bar handler (optional, provided by Engine)

        Returns:
            tensor: Model predictions

        Override this for custom prediction logic.
        """
        # Handle different batch formats
        '''if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch  # here in the else case errors might produce...'''
        if isinstance(batch, (tuple, list)):
            inputs = batch[0] if len(batch) > 0 else batch
        elif isinstance(batch, torch.Tensor):
            inputs = batch
        elif isinstance(batch, dict):
            # Handle dict batches (common in HuggingFace)
            inputs = batch.get('input', batch.get('data', batch))
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # Move to device
        inputs = self.send_data_to_device(inputs)

        # Forward pass
        outputs = self.pass_data_through_model(inputs)

        # Handle model outputs (for multi-output scenarios)
        outputs = self.handle_model_outputs(outputs)

        return outputs

    # ========================================================================
    # CORE OPERATIONS (Can be overridden)
    # ========================================================================

    def pass_data_through_model(self, data):
        """
        Forward pass through model.

        Override for multi-input models.

        Args:
            data: Input tensor(s)

        Returns:
            Output tensor(s)
        """
        return self.model(data)

    def compute_loss(self, data_pair, y_pred, **kwargs):
        """
        Compute loss from data pair and predictions.

        Override for custom loss computation or multi-output scenarios.

        Args:
            data_pair: Tuple of (inputs, targets)
            y_pred: Model predictions
            **kwargs: Additional arguments (e.g., epoch)

        Returns:
            Loss tensor
        """
        y_true = data_pair[1].to(self.device)
        return self.apply_loss(y_pred, y_true, **kwargs)

    def apply_loss(self, y_pred, y_true, **kwargs):
        """
        Apply loss function.

        Override for custom loss scenarios.

        Args:
            y_pred: Model predictions
            y_true: Ground truth
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        return self.loss(y_pred, y_true)

    def send_data_to_device(self, data):
        """
        Move data to device.

        Override for complex data structures (dict, nested tuples, etc.).

        Args:
            data: Data to move

        Returns:
            Data on device
        """
        return data.to(self.device)

    def handle_model_outputs(self, outputs):
        """
        Handle model outputs.

        Override for multi-output scenarios.

        Args:
            outputs: Raw model outputs

        Returns:
            Processed outputs
        """
        return outputs

    # ========================================================================
    # WEIGHTS (Model-specific operations)
    # ========================================================================

    def load_weights(self, weight_path: str, strict: bool = True):
        """
        Load model weights.

        Called by Engine but can be overridden for custom loading logic.

        Args:
            weight_path: Path to weight file
            strict: Strict state dict loading
        """
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        # Check file extension
        _, file_extension = os.path.splitext(weight_path)
        if file_extension != '.pt':
            raise ValueError(f"Invalid weight file: {weight_path}. Must be .pt file.")

        # Load weights
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=strict)

        self._weights_loaded = True

    def save_weights(self, weight_path: str):
        """
        Save model weights.

        Args:
            weight_path: Path to save weights
        """
        os.makedirs(os.path.dirname(weight_path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), weight_path)

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def get_sample_input(self):
        """Get sample input tensor (for summaries, TensorBoard graphs, etc.)."""
        if self.model_input_size is None:
            raise ValueError("model_input_size not set. Call build() first.")
        return torch.randn(1, *self.model_input_size).to(self.device)

    def set_model_input_size(self, *args, **kwargs):
        """
        Hook for setting model input size in complex scenarios.

        Optional - most models set this in build_inner_model().
        """
        raise NotImplementedError("Subclasses can implement set_model_input_size() if needed.")

    def summary(self, summary_depth: int = 3):
        """Print model summary."""
        if self.model_input_size is None:
            raise ValueError("model_input_size not set. Call build() first.")

        torch_version = torch.__version__.split('+')[0]
        if parse(torch_version) < parse('2.6.0'):
            model_summary(self.model, self.model_input_size, batch_dim=0, depth=summary_depth)
        else:
            model_summary(self.model, self.model_input_size)

    # ========================================================================
    # STATE PROPERTIES
    # ========================================================================

    @property
    def is_built(self):
        """Check if model is built."""
        return self._model_built

    @property
    def is_optimizer_set(self):
        """Check if optimizer is set."""
        return self._optimizer_bound

    @property
    def is_weights_loaded(self):
        """Check if weights are loaded."""
        return self._weights_loaded
