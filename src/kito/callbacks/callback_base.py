"""
Modern callback system for BaseModule.

Inspired by Keras and PyTorch Lightning callback patterns.
Each callback is independent and handles a single concern.
"""
from abc import ABC


class Callback(ABC):
    """
    Base class for all callbacks.

    Callbacks allow you to customize the training loop behavior
    by hooking into specific events (epoch start/end, batch start/end, etc.).

    All methods have default no-op implementations, so you only need to
    override the ones you care about.
    """

    def setup(self, engine, **kwargs):
        """
        Setup callback with Engine context.

        Called by Engine before training starts.
        Override this to auto-configure your callback.

        Args:
            engine: Engine instance
            **kwargs: Additional context

        Example:
            class MyCallback(Callback):
                def __init__(self, output_dir=None):
                    self.output_dir = output_dir

                def setup(self, engine, **kwargs):
                    if self.output_dir is None:
                        # Auto-configure from Engine
                        self.output_dir = engine.work_directory + '/outputs'
        """
        pass

    def on_train_begin(self, engine, model, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, engine, model, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch, engine, model, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number (1-indexed)
            engine: Reference to the Engine
            model: The PyTorch model
            logs: Dictionary of metrics (e.g., {'train_loss': 0.5, 'val_loss': 0.3})
            **kwargs: Additional context (val_data, val_outputs, etc.)
        """
        pass

    def on_train_batch_begin(self, batch, engine, model, **kwargs):
        """Called at the beginning of each training batch."""
        pass

    def on_train_batch_end(self, batch, engine, model, logs=None, **kwargs):
        """Called at the end of each training batch."""
        pass

    def on_validation_begin(self, epoch, engine, model, **kwargs):
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, epoch, engine, model, logs=None, **kwargs):
        """Called at the end of validation."""
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.

    Iterates through all callbacks and calls the appropriate method.
    """

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def append(self, callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, engine, model, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(engine, model, **kwargs)

    def on_train_end(self, engine, model, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(engine, model, **kwargs)

    def on_epoch_begin(self, epoch, engine, model, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, engine, model, **kwargs)

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, engine, model, logs, **kwargs)

    def on_train_batch_begin(self, batch, engine, model, **kwargs):
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, engine, model, **kwargs)

    def on_train_batch_end(self, batch, engine, model, logs=None, **kwargs):
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, engine, model, logs, **kwargs)

    def on_validation_begin(self, epoch, engine, model, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_begin(epoch, engine, model, **kwargs)

    def on_validation_end(self, epoch, engine, model, logs=None, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_end(epoch, engine, model, logs, **kwargs)
