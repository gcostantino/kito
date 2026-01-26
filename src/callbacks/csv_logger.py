import csv
import os

from src.callbacks.callback_base import Callback


class CSVLogger(Callback):
    """
    Log training metrics to a CSV file.

    Args:
        filename: Path to CSV file
        separator: Column separator (default: ',')
        append: Append to existing file or overwrite

    Example:
        csv_logger = CSVLogger('logs/training_log.csv')
    """

    def __init__(
            self,
            filename: str,
            separator: str = ',',
            append: bool = False
    ):
        self.filename = filename
        self.separator = separator
        self.append = append
        self.writer = None
        self.file = None
        self.keys = None

        # Create directory
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    def on_train_begin(self, engine, model, **kwargs):
        """Open CSV file and write header."""
        mode = 'a' if self.append else 'w'
        self.file = open(self.filename, mode, newline='')
        self.writer = csv.writer(self.file, delimiter=self.separator)

        # If not appending, we'll write header on first epoch
        if not self.append:
            self.keys = None

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Write metrics to CSV."""
        if logs is None:
            return

        # Add epoch to logs
        row_dict = {'epoch': epoch, **logs}

        # Write header if first time
        if self.keys is None:
            self.keys = list(row_dict.keys())
            self.writer.writerow(self.keys)

        # Write values
        self.writer.writerow([row_dict.get(k, '') for k in self.keys])
        self.file.flush()

    def on_train_end(self, engine, model, **kwargs):
        """Close CSV file."""
        if self.file:
            self.file.close()
