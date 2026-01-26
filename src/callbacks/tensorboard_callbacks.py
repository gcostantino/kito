from typing import Optional, Callable

from src.callbacks.callback_base import Callback
from torch.utils.tensorboard import SummaryWriter


class TensorBoardScalars(Callback):
    """
    Log scalar metrics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs

    Example:
        tb_scalars = TensorBoardScalars('logs/tensorboard')
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self, engine, model, **kwargs):
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Log scalars to TensorBoard."""
        if logs is None:
            return

        for key, value in logs.items():
            self.writer.add_scalar(key, value, epoch)

        self.writer.flush()

    def on_train_end(self, engine, model, **kwargs):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class TensorBoardHistograms(Callback):
    """
    Log model parameter histograms to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs
        freq: Frequency of histogram logging (every N epochs)

    Example:
        tb_histograms = TensorBoardHistograms('logs/tensorboard', freq=5)
    """

    def __init__(self, log_dir: str, freq: int = 1):
        self.log_dir = log_dir
        self.freq = freq
        self.writer = None

    def on_train_begin(self, engine, model, **kwargs):
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Log parameter histograms."""
        if epoch % self.freq != 0:
            return

        # Handle DDP
        model_to_log = model.module if hasattr(model, 'module') else model

        for name, param in model_to_log.named_parameters():
            self.writer.add_histogram(name, param, epoch)

        self.writer.flush()

    def on_train_end(self, engine, model, **kwargs):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class TensorBoardGraph(Callback):
    """
    Log model graph to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs
        input_to_model: Function that returns sample input for the model

    Example:
        def get_input():
            return torch.randn(1, 3, 64, 64)

        tb_graph = TensorBoardGraph('logs/tensorboard', input_to_model=get_input)
    """

    def __init__(
            self,
            log_dir: str,
            input_to_model: Optional[Callable] = None
    ):
        self.log_dir = log_dir
        self.input_to_model = input_to_model
        self.writer = None
        self.logged = False

    def on_train_begin(self, engine, model, **kwargs):
        """Initialize TensorBoard writer and log graph."""
        if self.logged:
            return

        self.writer = SummaryWriter(self.log_dir)

        # Get sample input
        if self.input_to_model:
            sample_input = self.input_to_model()
        elif hasattr(engine, 'get_sample_input'):
            sample_input = engine.get_sample_input()
        else:
            return  # Can't log graph without input

        # Handle DDP
        model_to_log = model.module if hasattr(model, 'module') else model

        self.writer.add_graph(model_to_log, sample_input)
        self.writer.flush()
        self.logged = True

    def on_train_end(self, engine, model, **kwargs):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
