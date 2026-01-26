import datetime
import os

from src.kito.callbacks.callback_base import Callback


class TextLogger(Callback):
    """
    Log training metrics to a text file.

    Args:
        filename: Path to log file
        append: Append to existing file or overwrite

    Example:
        text_logger = TextLogger('logs/training.log')
    """

    def __init__(
            self,
            filename: str,
            append: bool = True
    ):
        self.filename = filename
        self.append = append
        self.file = None

        # Create directory
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    def on_train_begin(self, engine, model, **kwargs):
        """Open log file."""
        mode = 'a' if self.append else 'w'
        self.file = open(self.filename, mode)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"{'=' * 60}")
        self.file.write(f"Training started at {timestamp}")
        self.file.write(f"{'=' * 60}")
        self.file.flush()

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Write metrics to log file."""
        if logs is None:
            return

        self.file.write(f"Epoch {epoch}: ")
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.file.write(metrics_str + "\n")
        self.file.flush()

    def on_train_end(self, engine, model, **kwargs):
        """Close log file."""
        if self.file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.file.write(f"\nTraining ended at {timestamp}\n")
            self.file.close()
